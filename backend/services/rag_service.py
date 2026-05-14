"""
RAG Service — Retrieval Augmented Generation

Tera purana app kya karta tha:
  context = f"Top destinations: {top_places}"  (sirf 5 city names!)

Ye naya system kya karta hai:
  1. User preferences → retrieve relevant destinations
  2. Real destination data → rich LLM context
  3. LLM generates intelligent, grounded response

Yahi RAG hai. Retrieval + Augmented context + Generation
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

TRAVEL_KNOWLEDGE_BASE = [
    {
        "id": "goa_001",
        "name": "Goa",
        "state": "Goa",
        "place_type": "Beach",
        "rating": 4.5,
        "avg_budget_per_day": 3,
        "safety_index": 4.2,
        "best_season": "November to February",
        "worst_season": "June to September (heavy monsoon)",
        "highlights": ["Baga Beach", "Dudhsagar Falls", "Old Goa Churches", "Spice Plantations", "Nightlife"],
        "good_for": ["Solo", "Friends", "Couple"],
        "interests": ["Relaxation", "Nature", "Food", "Culture"],
        "avg_trip_days": 5,
        "must_eat": ["Fish Curry Rice", "Bebinca", "Prawn Balchao"],
        "hidden_gems": ["Butterfly Beach", "Divar Island", "Fontainhas Latin Quarter"],
        "entry_fee_inr": 0,
        "nearest_airport": "Goa International Airport (GOI)",
        "local_transport": "Rent scooter Rs300/day or use Rapido/Ola",
        "avoid_if": "You hate crowds in peak season Dec-Jan",
        "text": "Goa is India's beach paradise on the western coast. Famous for pristine beaches, Portuguese heritage, vibrant nightlife, and fresh seafood. Best visited November to February. Ideal for friends groups, couples, and solo travelers. Budget travelers can manage in 2-3k per day.",
    },
    {
        "id": "manali_001",
        "name": "Manali",
        "state": "Himachal Pradesh",
        "place_type": "Hill Station",
        "rating": 4.6,
        "avg_budget_per_day": 2,
        "safety_index": 4.0,
        "best_season": "October to June",
        "worst_season": "July-August monsoon for road trips",
        "highlights": ["Rohtang Pass", "Solang Valley", "Hadimba Temple", "Old Manali", "River Rafting"],
        "good_for": ["Friends", "Couple", "Family", "Solo"],
        "interests": ["Adventure", "Nature", "Relaxation"],
        "avg_trip_days": 5,
        "must_eat": ["Siddu", "Trout Fish", "Chha Gosht"],
        "hidden_gems": ["Hampta Pass trek", "Chandratal Lake", "Prashar Lake"],
        "entry_fee_inr": 0,
        "nearest_airport": "Bhuntar Airport KUU 50km away",
        "local_transport": "Rent bike or scooty local buses to Rohtang",
        "avoid_if": "You have altitude sickness or respiratory issues",
        "text": "Manali is the adventure capital of Himachal Pradesh. Snow-covered peaks, adventure sports like paragliding and river rafting, and scenic mountain drives make it India's most visited hill station. Best for young travelers and adventure seekers. Very budget-friendly at 1500-2000 per day.",
    },
    {
        "id": "jaipur_001",
        "name": "Jaipur",
        "state": "Rajasthan",
        "place_type": "Historical",
        "rating": 4.4,
        "avg_budget_per_day": 2,
        "safety_index": 3.8,
        "best_season": "October to March",
        "worst_season": "May to July extreme heat 45 degrees",
        "highlights": ["Amber Fort", "Hawa Mahal", "City Palace", "Jantar Mantar", "Johari Bazaar"],
        "good_for": ["Family", "Couple", "Solo"],
        "interests": ["Culture", "Historical", "Food"],
        "avg_trip_days": 3,
        "must_eat": ["Dal Baati Churma", "Laal Maas", "Ghevar", "Pyaaz Kachori"],
        "hidden_gems": ["Nahargarh Fort sunset", "Panna Meena Kund stepwell", "Chokhi Dhani village"],
        "entry_fee_inr": 500,
        "nearest_airport": "Jaipur International Airport JAI",
        "local_transport": "Auto-rickshaw cycle-rickshaw Ola Uber",
        "avoid_if": "You dislike tourist crowds or extreme summer heat",
        "text": "Jaipur the Pink City is the crown jewel of Rajasthan tourism. UNESCO World Heritage Sites, majestic forts, vibrant bazaars and authentic Rajasthani cuisine make it a must-visit. Perfect for history and culture lovers, families, and photographers.",
    },
    {
        "id": "kerala_001",
        "name": "Kerala Alleppey",
        "state": "Kerala",
        "place_type": "Backwater",
        "rating": 4.7,
        "avg_budget_per_day": 4,
        "safety_index": 4.5,
        "best_season": "September to March",
        "worst_season": "June to August monsoon",
        "highlights": ["Houseboat stay", "Backwater cruise", "Kuttanad paddy fields", "Marari Beach", "Ayurvedic spas"],
        "good_for": ["Couple", "Family", "Solo"],
        "interests": ["Relaxation", "Nature", "Culture"],
        "avg_trip_days": 4,
        "must_eat": ["Karimeen pollichathu", "Kerala Sadya", "Appam with stew"],
        "hidden_gems": ["Pathiramanal Island", "Kumarakom Bird Sanctuary", "Ambalapuzha Temple"],
        "entry_fee_inr": 0,
        "nearest_airport": "Cochin International Airport COK 85km",
        "local_transport": "Houseboat country boats auto-rickshaws",
        "avoid_if": "You dislike humidity or slow-paced travel",
        "text": "Kerala backwaters are unlike anywhere in India. Alleppey is famous for houseboat stays drifting through serene canals surrounded by coconut palms. Perfect for honeymooners and couples. Ayurvedic treatments and fresh seafood make it a complete experience.",
    },
    {
        "id": "leh_001",
        "name": "Leh-Ladakh",
        "state": "Ladakh",
        "place_type": "Adventure",
        "rating": 4.8,
        "avg_budget_per_day": 4,
        "safety_index": 3.9,
        "best_season": "June to September only",
        "worst_season": "October to May roads closed -20 degrees",
        "highlights": ["Pangong Lake", "Nubra Valley", "Khardung La Pass", "Monasteries", "Magnetic Hill"],
        "good_for": ["Solo", "Friends", "Couple"],
        "interests": ["Adventure", "Nature", "Spiritual"],
        "avg_trip_days": 7,
        "must_eat": ["Thukpa", "Momos", "Skyu", "Butter tea"],
        "hidden_gems": ["Tso Moriri Lake", "Hanle dark sky reserve", "Zanskar Valley"],
        "entry_fee_inr": 0,
        "nearest_airport": "Kushok Bakula Rimpochee Airport IXL in Leh city",
        "local_transport": "Rent Royal Enfield Rs1200 per day shared taxis",
        "avoid_if": "You have heart or lung conditions or fear altitude sickness",
        "text": "Leh-Ladakh is India's crown jewel for adventure seekers. At 3500m altitude it offers breathtaking landscapes, Buddhist monasteries, the world's highest motorable roads, and surreal lakes. Requires physical fitness and altitude acclimatization. Best done June-September plan at least 7-10 days.",
    },
    {
        "id": "rishikesh_001",
        "name": "Rishikesh",
        "state": "Uttarakhand",
        "place_type": "Spiritual",
        "rating": 4.4,
        "avg_budget_per_day": 1,
        "safety_index": 4.3,
        "best_season": "September to June",
        "worst_season": "July-August Ganga floods dangerous",
        "highlights": ["Laxman Jhula", "River Rafting", "Yoga and Meditation", "Beatles Ashram", "Triveni Ghat"],
        "good_for": ["Solo", "Friends", "Couple", "Family"],
        "interests": ["Spiritual", "Adventure", "Nature"],
        "avg_trip_days": 3,
        "must_eat": ["Aloo puri", "Chole bhature", "Lassi famous"],
        "hidden_gems": ["Neer Garh Waterfall", "Rajaji National Park", "Kunjapuri Temple sunrise"],
        "entry_fee_inr": 0,
        "nearest_airport": "Jolly Grant Airport Dehradun 35km",
        "local_transport": "Walk most places Vikram shared autos cycle",
        "avoid_if": "You want beach or mountain snow experience",
        "text": "Rishikesh is the yoga capital of the world and a perfect blend of spirituality and adventure. River rafting in Ganga, world-class yoga centers, affordable cafes, and the iconic Laxman Jhula bridge. Most budget-friendly hill destination at 800-1000 per day.",
    },
    {
        "id": "udaipur_001",
        "name": "Udaipur",
        "state": "Rajasthan",
        "place_type": "Heritage",
        "rating": 4.6,
        "avg_budget_per_day": 3,
        "safety_index": 4.0,
        "best_season": "September to March",
        "worst_season": "April to July extreme heat",
        "highlights": ["Lake Pichola boat ride", "City Palace", "Jag Mandir", "Sajjangarh Monsoon Palace", "Saheliyon ki Bari"],
        "good_for": ["Couple", "Family", "Solo"],
        "interests": ["Culture", "Heritage", "Relaxation", "Food"],
        "avg_trip_days": 3,
        "must_eat": ["Dal Baati Churma", "Gatte ki Sabzi", "Mawa Kachori"],
        "hidden_gems": ["Bagore ki Haveli museum", "Ambrai Ghat at sunset", "Shilpgram Crafts Village"],
        "entry_fee_inr": 500,
        "nearest_airport": "Maharana Pratap Airport UDR",
        "local_transport": "Auto Ola Uber rented bicycle",
        "avoid_if": "Very tight budget lake-view hotels are pricier",
        "text": "Udaipur the City of Lakes is Rajasthan's most romantic destination. The shimmering Lake Pichola, majestic palaces, and traditional havelis create a fairy-tale setting. Most popular honeymoon destination in India. Famous for art crafts and classical music.",
    },
    {
        "id": "darjeeling_001",
        "name": "Darjeeling",
        "state": "West Bengal",
        "place_type": "Hill Station",
        "rating": 4.5,
        "avg_budget_per_day": 2,
        "safety_index": 4.1,
        "best_season": "March to May September to November",
        "worst_season": "June to August monsoon landslides",
        "highlights": ["Tiger Hill sunrise Kanchenjunga view", "Toy Train UNESCO", "Tea Gardens", "Batasia Loop", "Peace Pagoda"],
        "good_for": ["Couple", "Family", "Solo"],
        "interests": ["Nature", "Culture", "Relaxation"],
        "avg_trip_days": 4,
        "must_eat": ["Darjeeling Tea First Flush", "Momos", "Thukpa", "Sel Roti"],
        "hidden_gems": ["Mirik Lake", "Senchal Wildlife Sanctuary", "Lamahatta Eco Park"],
        "entry_fee_inr": 0,
        "nearest_airport": "Bagdogra Airport IXB 90km",
        "local_transport": "Toy Train shared jeeps walk in town",
        "avoid_if": "You are motion sick mountain roads are very winding",
        "text": "Darjeeling is the queen of hill stations, known for its tea gardens, stunning Kanchenjunga views, and the iconic Darjeeling Himalayan Railway UNESCO heritage. The sunrise from Tiger Hill is one of India's most spectacular sights. Perfect for nature lovers and couples.",
    },
]


class RAGService:
    """
    RAG = Retrieval Augmented Generation
    Step 1: Retrieve relevant destinations from knowledge base
    Step 2: Build rich context (Augment)
    Step 3: LLM uses this context to Generate grounded answers
    """

    def __init__(self):
        self.knowledge_base = TRAVEL_KNOWLEDGE_BASE
        self.initialized = False

    async def initialize(self):
        logger.info(f"Loading {len(self.knowledge_base)} destinations into knowledge base...")
        self.initialized = True
        logger.info("Knowledge base ready.")

    def retrieve_destinations(
        self,
        companions: str,
        interests: List[str],
        season: str,
        budget_per_day: int,
        place_type: Optional[str] = None,
        min_safety: float = 3.0,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Score and retrieve most relevant destinations.
        In production: replace with ChromaDB vector similarity search.
        """
        scored = []

        for dest in self.knowledge_base:
            score = 0.0

            if dest["safety_index"] < min_safety:
                continue

            if dest["avg_budget_per_day"] > budget_per_day * 2:
                continue

            if place_type and dest["place_type"] != place_type:
                continue

            if companions in dest.get("good_for", []):
                score += 2.0

            dest_interests = dest.get("interests", [])
            interest_overlap = len(set(interests) & set(dest_interests))
            score += interest_overlap * 1.5

            if season.lower() in dest.get("best_season", "").lower():
                score += 2.0

            budget_diff = abs(dest["avg_budget_per_day"] - budget_per_day)
            score += max(0, 2 - budget_diff * 0.5)

            score += dest["rating"]

            scored.append((score, dest))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [dest for _, dest in scored[:top_k]]

    def build_rag_context(self, destinations: List[Dict]) -> str:
        """Build rich context string to pass to LLM"""
        context_parts = []
        for dest in destinations:
            context_parts.append(
                f"DESTINATION: {dest['name']}, {dest['state']}\n"
                f"Type: {dest['place_type']} | Rating: {dest['rating']}/5 | Safety: {dest['safety_index']}/5\n"
                f"Budget: Rs{dest['avg_budget_per_day']}k-{dest['avg_budget_per_day']+2}k per day\n"
                f"Best Season: {dest['best_season']}\n"
                f"Highlights: {', '.join(dest['highlights'][:4])}\n"
                f"Good For: {', '.join(dest['good_for'])}\n"
                f"Must Eat: {', '.join(dest['must_eat'][:3])}\n"
                f"Hidden Gems: {', '.join(dest['hidden_gems'][:2])}\n"
                f"Transport: {dest['local_transport']}\n"
                f"Avoid If: {dest['avoid_if']}\n"
                f"Summary: {dest['text']}"
            )
        return "\n---\n".join(context_parts)

    def search_for_chat(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword search for chatbot context.
        Production: use embedding similarity.
        """
        query_lower = query.lower()
        scored = []
        for dest in self.knowledge_base:
            score = 0
            if dest["name"].lower() in query_lower:
                score += 5
            if dest["state"].lower() in query_lower:
                score += 3
            if dest["place_type"].lower() in query_lower:
                score += 2
            for interest in dest.get("interests", []):
                if interest.lower() in query_lower:
                    score += 1
            if score > 0:
                scored.append((score, dest))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]] if scored else self.knowledge_base[:top_k]


rag_service = RAGService()