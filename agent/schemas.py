"""Pydantic schemas for the structured output of each LLM step."""
from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    query: str


class Grade(BaseModel):
    sufficient: bool
    reason: str


class Answer(BaseModel):
    answer: str
    citations: list[int]
