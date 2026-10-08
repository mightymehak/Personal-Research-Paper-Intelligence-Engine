from pydantic import BaseModel


class PaperChunk(BaseModel):
    chunk_id: str
    paper_id: str

    paper_title: str
    section: str

    chunk_index: int
    text: str
    word_count: int