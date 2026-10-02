"""
FastAPI REST routes for grammar error detection, correction, candidate generation,
evaluation, and system health checks.
"""

from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
import logging

from backend.app.config import settings
from backend.app.models.grammar_model import GrammarModel
from backend.app.services.correction_service import CorrectionService
from backend.app.services.evaluation_service import EvaluationService
from backend.app.api.schemas import (
    CorrectRequest, CorrectResponse,
    DetectRequest, DetectResponse,
    CandidatesRequest, CandidatesResponse,
    EvaluateRequest, EvaluateResponse,
    HealthResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Grammar Error Correction"])


def get_model() -> GrammarModel:
    model = GrammarModel.get_instance()
    if not model.is_loaded:
        success = model.load()
        if not success:
            logger.warning("Model could not be loaded into memory.")
    return model


def get_correction_service(model: GrammarModel = Depends(get_model)) -> CorrectionService:
    return CorrectionService(model=model)


def get_evaluation_service(model: GrammarModel = Depends(get_model)) -> EvaluationService:
    return EvaluationService(model=model)


@router.get("/health", response_model=HealthResponse)
def health_check(model: GrammarModel = Depends(get_model)):
    """Health check reporting system status, active model, and device execution mode."""
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        model_name=model.active_checkpoint,
        device=model.device,
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.post("/correct", response_model=CorrectResponse)
def correct_text(
    request: CorrectRequest,
    service: CorrectionService = Depends(get_correction_service)
):
    """
    Primary endpoint: Detects errors, generates candidate corrections, ranks candidates,
    localizes edit spans, and classifies errors into directly verified and inferred categories.
    """
    try:
        result = service.process_text(
            text=request.text,
            num_candidates=request.num_candidates,
            num_beams=request.num_beams,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        logger.error(f"Error in /correct endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")


@router.post("/detect", response_model=DetectResponse)
def detect_errors(
    request: DetectRequest,
    service: CorrectionService = Depends(get_correction_service)
):
    """
    Detection-only endpoint: Identifies incorrect words or structures,
    returning localized spans and error category tiers.
    """
    try:
        result = service.process_text(text=request.text, num_candidates=1)
        errors = result["detected_errors"]
        return DetectResponse(
            original_text=request.text,
            detected_errors=errors,
            error_count=len(errors),
            has_errors=len(errors) > 0
        )
    except Exception as e:
        logger.error(f"Error in /detect endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/candidates", response_model=CandidatesResponse)
def get_candidates(
    request: CandidatesRequest,
    service: CorrectionService = Depends(get_correction_service)
):
    """
    Generates and returns ranked alternative correction candidates with sequence confidences.
    """
    try:
        result = service.process_text(
            text=request.text,
            num_candidates=request.num_candidates
        )
        return CandidatesResponse(
            original_text=request.text,
            candidates=result["candidates"]
        )
    except Exception as e:
        logger.error(f"Error in /candidates endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Candidate generation failed: {str(e)}")


@router.post("/evaluate", response_model=EvaluateResponse)
def run_evaluation(
    request: EvaluateRequest,
    eval_service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Evaluates the active model on a held-out test dataset or custom evaluation pairs.
    Returns real GLEU, BLEU, and F0.5 metrics with full reproducibility metadata.
    """
    try:
        if request.custom_pairs:
            pairs = [{"source": p.source, "target": p.target} for p in request.custom_pairs]
            result = eval_service.evaluate_pairs(pairs)
            return EvaluateResponse(
                reproducibility={
                    "model_name_or_path": eval_service.model.active_checkpoint,
                    "device": eval_service.model.device,
                    "random_seed": settings.RANDOM_SEED,
                    "num_samples": len(pairs),
                },
                metrics={
                    "corpus_gleu": result["corpus_gleu"],
                    "corpus_bleu": result["corpus_bleu"],
                    "exact_match_accuracy_percent": result["exact_match_accuracy"],
                    "token_precision_percent": result["precision"],
                    "token_recall_percent": result["recall"],
                    "token_f0_5_score_percent": result["f0_5_score"],
                },
                sample_results=result["pairs"]
            )
        else:
            # File-based dataset evaluation
            dataset_file_map = {
                "demo": settings.DATA_DIR / "demo" / "demo_pairs.jsonl",
                "test": settings.DATA_DIR / "processed" / "test.jsonl",
                "val": settings.DATA_DIR / "processed" / "val.jsonl",
            }
            target_path = dataset_file_map.get(request.dataset_name)
            if not target_path or not target_path.exists():
                # Fallback to demo if processed test set doesn't exist yet
                target_path = settings.DATA_DIR / "demo" / "demo_pairs.jsonl"

            result = eval_service.evaluate_dataset(
                dataset_path=target_path,
                max_samples=request.max_samples
            )
            return EvaluateResponse(
                reproducibility=result["reproducibility"],
                metrics=result["metrics"],
                sample_results=result.get("sample_results", [])
            )
    except Exception as e:
        logger.error(f"Error in /evaluate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
