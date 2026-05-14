"""
Chat Router — /api/v1/chat

Real chatbot implementation with:
- Full conversation history
- RAG context per message
- Follow-up question suggestions
"""
from fastapi import APIRouter, HTTPException
import logging

from models.schemas import ChatRequest, ChatResponse
from services.rag_service import rag_service
from services.llm_service import llm_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational travel assistant.

    Key difference from your old chatbot:
    - Old: sends 5 city names as context
    - New: searches knowledge base for relevant destinations,
           sends FULL destination data as context
    """
    try:
        # RAG: search knowledge base for relevant destinations based on user message
        relevant_destinations = rag_service.search_for_chat(
            query=request.message,
            top_k=3
        )

        # Build rich context
        rag_context = rag_service.build_rag_context(relevant_destinations)

        # Get LLM response with context + history
        response = await llm_service.chat(
            user_message=request.message,
            conversation_history=[msg.model_dump() for msg in request.conversation_history],
            rag_context=rag_context
        )

        # Extract destination names mentioned for frontend highlighting
        mentioned_destinations = [
            dest["name"] for dest in relevant_destinations
            if dest["name"].lower() in response["reply"].lower()
        ]

        return ChatResponse(
            reply=response["reply"],
            suggested_destinations=mentioned_destinations if mentioned_destinations else None,
            follow_up_questions=response.get("follow_up_questions", [])
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service error. Please try again.")