"""
Unit tests for RAG Service
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.rag_service import rag_service

class TestRAGService:
    """Test RAG recommendation scoring"""
    
    def test_retrieve_destinations_friends_adventure(self):
        """Test: Friends + Adventure + Nature + Winter"""
        results = rag_service.retrieve_destinations(
            companions="Friends",
            interests=["Adventure", "Nature"],
            season="Winter",
            budget_per_day=3,
            min_safety=3.5,
            top_k=5
        )
        
        assert len(results) <= 5
        assert len(results) > 0
    
    def test_build_rag_context_format(self):
        """Test: RAG context format"""
        destinations = [rag_service.knowledge_base[0]]
        context = rag_service.build_rag_context(destinations)
        
        assert "DESTINATION:" in context
        assert "Budget:" in context