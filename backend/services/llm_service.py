"""
LLM Service — wraps Groq API

WHY SEPARATE FILE:
Fresher mistake: API calls directly inside route handlers.
Pro move: service layer. Agar Groq se OpenAI switch karna ho,
sirf ye file change hogi, routes unchanged.
"""
import os
import json
import logging
from typing import List, Dict, Optional
from groq import AsyncGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TravelGuru, an expert Indian travel advisor with 15 years of experience.
You have deep knowledge of every destination, budget tips, local culture, food, transport, and safety.

RULES:
1. ALWAYS base your answers on the destination data provided in context
2. Give specific, actionable advice (actual prices, names, timings)
3. Be warm and enthusiastic but honest about challenges
4. If user asks about a destination NOT in context, say you need more info
5. Always mention one budget tip and one hidden gem per recommendation
6. Respond in a friendly, conversational tone - like a knowledgeable friend, not a brochure
"""


class LLMService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail")
        self.client = AsyncGroq(api_key=api_key) if api_key else None
        self.model = "llama-3.3-70b-versatile"

    async def generate_recommendations(
        self,
        rag_context: str,
        user_prefs: dict,
        destinations: List[dict]
    ) -> dict:
        """Generate AI summary for recommendation results"""
        if not self.client:
            return self._fallback_response(destinations)

        prompt = f"""
Based on the user's preferences and the destination data below, provide personalized travel advice.

USER PREFERENCES:
- Budget: Rs{user_prefs['budget_per_day']}k per day
- Duration: {user_prefs['duration_days']} days
- Companions: {user_prefs['companions']}
- Interests: {', '.join(user_prefs['interests'])}
- Season: {user_prefs['season']}

RETRIEVED DESTINATION DATA:
{rag_context}

Provide:
1. A 2-3 sentence personalized summary of why these destinations match their needs
2. Top packing tip for their trip type
3. One money-saving tip specific to their budget

Keep it concise and actionable.
"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return {"ai_summary": response.choices[0].message.content}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._fallback_response(destinations)

    async def chat(
        self,
        user_message: str,
        conversation_history: List[Dict],
        rag_context: str
    ) -> dict:
        """Process chat message with RAG context"""
        if not self.client:
            return {
                "reply": "AI service unavailable. Please set GROQ_API_KEY.",
                "follow_up_questions": []
            }

        # Build messages array with full history (this is how real chatbots work)
        messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nRELEVANT DESTINATION INFO:\n{rag_context}"}]

        # Add conversation history (max last 10 exchanges to stay within token limits)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            reply = response.choices[0].message.content

            # Generate follow-up questions
            follow_ups = await self._generate_follow_ups(user_message, reply)

            return {"reply": reply, "follow_up_questions": follow_ups}
        except Exception as e:
            logger.error(f"Chat LLM call failed: {e}")
            return {
                "reply": f"Sorry, I encountered an error. Please try again.",
                "follow_up_questions": ["Tell me about Goa", "What's best for families?"]
            }

    async def _generate_follow_ups(self, user_msg: str, ai_reply: str) -> List[str]:
        """Generate contextual follow-up questions"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"Based on this travel conversation, suggest 2 short follow-up questions the user might ask. Return ONLY a JSON array of 2 strings, nothing else.\n\nUser asked: {user_msg}\nAI replied: {ai_reply[:200]}"
                }],
                temperature=0.5,
                max_tokens=100
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except Exception:
            return ["What's the best time to visit?", "What are the must-try foods?"]

    def _fallback_response(self, destinations: List[dict]) -> dict:
        """Used when API key not available"""
        names = [d["name"] for d in destinations[:3]]
        return {
            "ai_summary": f"Based on your preferences, we recommend: {', '.join(names)}. Set GROQ_API_KEY for detailed AI insights."
        }


llm_service = LLMService()