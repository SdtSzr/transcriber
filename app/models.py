from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    segments: list[Segment] | None = None
