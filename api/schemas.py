from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    location_boost_factor: float = Field(default=1.8, ge=1.0, le=5.0)
    explain: bool = False


class SuggestionResponse(BaseModel):
    suggestions: list[str]

