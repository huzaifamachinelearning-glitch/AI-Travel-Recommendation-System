"""
Recommend Router — /api/v1/recommend

Real world mein har feature ka apna router hota hai.
Sab kuch main.py mein nahi likhte — wo spaghetti code hai.
"""
from fastapi import APIRouter, HTTPException
import logging

from models.schemas import RecommendationRequest, RecommendationResponse, Destination
from services.rag_service import rag_service
from services.llm_service import llm_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Main recommendation endpoint.

    Flow:
    1. Validate input (Pydantic does this automatically)
    2. Retrieve relevant destinations (RAG - Retrieval)
    3. Build context for LLM (RAG - Augment)
    4. Generate AI insights (RAG - Generation)
    5. Return structured response
    """
    try:
        logger.info(f"Recommendation request: {request.companions}, {request.season}, budget={request.budget_per_day}k")

        # Step 1: RAG Retrieval
        destinations = rag_service.retrieve_destinations(
            companions=request.companions,
            interests=request.interests,
            season=request.season,
            budget_per_day=request.budget_per_day,
            place_type=request.place_type,
            min_safety=request.min_safety,
            top_k=5
        )

        if not destinations:
            raise HTTPException(
                status_code=404,
                detail="No destinations found matching your preferences. Try relaxing some filters."
            )

        # Step 2: Build RAG context
        rag_context = rag_service.build_rag_context(destinations)

        # Step 3: LLM Generation
        ai_response = await llm_service.generate_recommendations(
            rag_context=rag_context,
            user_prefs=request.model_dump(),
            destinations=destinations
        )

        # Step 4: Build structured response
        dest_objects = []
        for dest in destinations:
            dest_objects.append(Destination(
                name=dest["name"],
                state=dest["state"],
                place_type=dest["place_type"],
                rating=dest["rating"],
                avg_budget_per_day=dest["avg_budget_per_day"],
                safety_index=dest["safety_index"],
                best_season=dest["best_season"],
                highlights=dest["highlights"][:4],
                why_recommended=f"Matches your {request.companions.lower()} trip with {', '.join(request.interests[:2])} interests"
            ))

        total_cost = request.budget_per_day * request.duration_days

        return RecommendationResponse(
            destinations=dest_objects,
            ai_summary=ai_response.get("ai_summary", "Great destinations found for you!"),
            total_estimated_cost=total_cost,
            best_time_to_visit=destinations[0]["best_season"] if destinations else "October to March",
            packing_tips=[
                f"Pack for {request.season} weather",
                "Always carry a portable charger and power bank",
                "Keep digital copies of all IDs",
                "Download offline maps before you travel"
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")