from typing import Dict, List
from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str
    filename: str

    title: str = ""
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = Field(default_factory=list)

    sections: Dict[str, str] = Field(default_factory=dict)

    references: List[str] = Field(default_factory=list)

    full_text: str = ""