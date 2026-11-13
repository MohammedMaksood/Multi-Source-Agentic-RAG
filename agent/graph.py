"""Phase 1 agent: rewrite, retrieve, grade, then generate, with a capped retry loop and per-thread memory."""
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.schemas import Answer, Grade, RewrittenQuery

MAX_RETRIES = 2
TOP_K = 5
HISTORY_MESSAGES = 6
FALLBACK_ANSWER = "I could not find enough evidence in the corpus to answer this question."


class State(TypedDict, total=False):
    messages: Annotated[list[dict], add]
    question: str
    query: str
    documents: list[dict]
    sufficient: bool
    reason: str
    retries: int
    answer: str
    sources: list[str]


def format_history(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[-HISTORY_MESSAGES:])


def format_passages(documents: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(documents, start=1))


def finish(question: str, answer: str, sources: list[str]) -> dict:
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return {"answer": answer, "sources": sources, "messages": messages}


def build_graph(llm, retriever, mode: str, checkpointer):
    def rewrite(state: State) -> dict:
        history = format_history(state.get("messages", []))
        if not history:
            return {"query": state["question"], "retries": 0}
        rewritten = llm.structured("rewrite", 1, RewrittenQuery, history=history, question=state["question"])
        return {"query": rewritten.query.strip() or state["question"], "retries": 0}

    def retrieve(state: State) -> dict:
        chunks = retriever.retrieve(state["query"], mode, k=TOP_K)
        return {"documents": [{"paper_id": c.paper_id, "index": c.index, "text": c.text} for c in chunks]}

    def grade(state: State) -> dict:
        graded = llm.structured("grade", 1, Grade, question=state["question"], passages=format_passages(state["documents"]))
        return {"sufficient": graded.sufficient, "reason": graded.reason}

    def after_grade(state: State) -> str:
        if state["sufficient"]:
            return "generate"
        return "reformulate" if state["retries"] < MAX_RETRIES else "fallback"

    def reformulate(state: State) -> dict:
        new = llm.structured(
            "reformulate", 1, RewrittenQuery,
            question=state["question"], previous_query=state["query"], reason=state["reason"],
        )
        return {"query": new.query.strip() or state["question"], "retries": state["retries"] + 1}

    def generate(state: State) -> dict:
        documents = state["documents"]
        out = llm.structured("generate", 1, Answer, question=state["question"], passages=format_passages(documents))
        cited = [n for n in dict.fromkeys(out.citations) if 1 <= n <= len(documents)]
        sources = [f"{documents[n - 1]['paper_id']}#{documents[n - 1]['index']}" for n in cited]
        return finish(state["question"], out.answer, sources)

    def fallback(state: State) -> dict:
        return finish(state["question"], FALLBACK_ANSWER, [])

    graph = StateGraph(State)
    for node in (rewrite, retrieve, grade, reformulate, generate, fallback):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "reformulate", "fallback"])
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile(checkpointer=checkpointer).with_config(tags=[f"mode:{mode}"], metadata={"retrieval_mode": mode})


def ask(graph, question: str, thread_id: str) -> dict:
    return graph.invoke({"question": question}, config={"configurable": {"thread_id": thread_id}})
