"""
Destinations Router — /api/v1/destinations
Browse, filter, and get details for all destinations.
"""
from fastapi import APIRouter, Query
from typing import Optional, List
import logging

from services.rag_service import rag_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/destinations")
async def list_destinations(
    place_type: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    min_rating: float = Query(default=0.0, ge=0, le=5),
    max_budget: Optional[int] = Query(None)
):
    """List all destinations with optional filters"""
    results = rag_service.knowledge_base

    if place_type:
        results = [d for d in results if d["place_type"].lower() == place_type.lower()]
    if state:
        results = [d for d in results if state.lower() in d["state"].lower()]
    if min_rating:
        results = [d for d in results if d["rating"] >= min_rating]
    if max_budget:
        results = [d for d in results if d["avg_budget_per_day"] <= max_budget]

    return {
        "total": len(results),
        "destinations": results
    }


@router.get("/destinations/{destination_id}")
async def get_destination(destination_id: str):
    """Get full details for a specific destination"""
    dest = next((d for d in rag_service.knowledge_base if d["id"] == destination_id), None)
    if not dest:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Destination '{destination_id}' not found")
    return dest


@router.get("/destinations/stats/summary")
async def get_stats():
    """Dashboard stats"""
    kb = rag_service.knowledge_base
    return {
        "total_destinations": len(kb),
        "avg_rating": round(sum(d["rating"] for d in kb) / len(kb), 2),
        "states_covered": len(set(d["state"] for d in kb)),
        "place_types": list(set(d["place_type"] for d in kb)),
        "budget_range": {
            "min": min(d["avg_budget_per_day"] for d in kb),
            "max": max(d["avg_budget_per_day"] for d in kb)
        }
    }