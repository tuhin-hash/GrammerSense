"""
Pydantic API request and response validation schemas.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class CorrectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Input text to correct")
    num_candidates: int = Field(default=3, ge=1, le=5, description="Number of candidate generations to return")
    num_beams: int = Field(default=4, ge=1, le=8, description="Number of beams for beam search")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")


class CandidateSchema(BaseModel):
    candidate_text: str
    rank: int
    confidence: float
    edit_distance: int
    rank_score: float
    is_selected: bool


class DetectedErrorSchema(BaseModel):
    error_id: str
    start_char: int
    end_char: int
    original_text: str
    replacement_text: str
    category: str
    verification_tier: Literal["DIRECTLY_IDENTIFIED", "INFERRED", "CORRECT"]
    confidence: float
    explanation: str
    rule_id: Optional[str] = None


class DiffChunkSchema(BaseModel):
    type: Literal["equal", "delete", "insert"]
    text: str


class MetricsSchema(BaseModel):
    original_word_count: int
    corrected_word_count: int
    error_count: int
    directly_identified_count: int
    inferred_count: int
    edit_density_per_100_words: float
    error_free_token_ratio: float
    model_confidence: float


class CorrectResponse(BaseModel):
    original_text: str
    corrected_text: str
    candidates: List[CandidateSchema]
    detected_errors: List[DetectedErrorSchema]
    diff_chunks: List[DiffChunkSchema]
    metrics: MetricsSchema


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Input text to analyze for errors")


class DetectResponse(BaseModel):
    original_text: str
    detected_errors: List[DetectedErrorSchema]
    error_count: int
    has_errors: bool


class CandidatesRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Input text")
    num_candidates: int = Field(default=4, ge=1, le=10, description="Number of candidates")


class CandidatesResponse(BaseModel):
    original_text: str
    candidates: List[CandidateSchema]


class PairSchema(BaseModel):
    source: str
    target: str
    id: Optional[str] = None


class EvaluateRequest(BaseModel):
    dataset_name: Optional[Literal["demo", "test", "val"]] = "demo"
    custom_pairs: Optional[List[PairSchema]] = None
    max_samples: Optional[int] = Field(default=None, ge=1, le=1000)


class EvaluateResponse(BaseModel):
    reproducibility: Dict[str, Any]
    metrics: Dict[str, Any]
    sample_results: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str
    timestamp: str
