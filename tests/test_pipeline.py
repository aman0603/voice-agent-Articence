"""Tests for the Voice RAG pipeline components."""

import pytest
import time


class TestIntentDetector:
    """Tests for intent detection from partial transcripts."""
    
    def test_power_domain_detection(self):
        from src.asr.intent_detector import IntentDetector
        
        detector = IntentDetector()
        intent = detector.detect("What are the power supply redundancy modes?")
        assert intent is not None
        assert intent.domain == "power"
        assert "power" in intent.keywords
    
    def test_storage_domain_detection(self):
        from src.asr.intent_detector import IntentDetector
        
        detector = IntentDetector()
        intent = detector.detect("How do I configure RAID 5?")
        assert intent is not None
        assert intent.domain == "storage"
    
    def test_query_type_detection(self):
        from src.asr.intent_detector import IntentDetector
        
        detector = IntentDetector()
        
        # Troubleshooting
        intent = detector.detect("The power LED is showing amber error")
        assert intent is not None
        assert intent.query_type == "troubleshooting"
        
        # Configuration
        intent = detector.detect("How do I configure the network settings?")
        assert intent is not None
        assert intent.query_type == "configuration"
    
    def test_prefetch_query_generation(self):
        from src.asr.intent_detector import IntentDetector
        
        detector = IntentDetector()
        
        intent = detector.detect("power supply not working")
        assert intent is not None
        
        query = detector.get_prefetch_query(intent)
        assert "power" in query.lower()


class TestContextManager:
    """Tests for conversation context management."""
    
    def test_session_creation(self):
        from src.query.context_manager import ContextManager
        
        manager = ContextManager()
        session = manager.get_or_create_session("test-session")
        assert session.session_id == "test-session"
        assert len(session.entries) == 0
    
    def test_message_tracking(self):
        from src.query.context_manager import ContextManager
        
        manager = ContextManager()
        manager.add_user_message("test", "What are power redundancy modes?")
        manager.add_assistant_message("test", "There are three modes...", ["power"])
        
        session = manager.get_or_create_session("test")
        assert len(session.entries) == 2
        assert session.active_topic == "power"


class TestVoiceOptimizer:
    """Tests for voice optimization."""
    
    def test_sentence_length(self):
        from src.voice.optimizer import VoiceOptimizer
        
        optimizer = VoiceOptimizer(max_words_per_sentence=15)
        
        long_text = "Ensure the redundant power supply modules are configured in hot-swap mode and verify LED indicators according to Table 7-4 in the technical manual."
        
        optimized = optimizer.optimize(long_text)
        sentences = optimized.split('.')
        
        for sentence in sentences:
            if sentence.strip():
                word_count = len(sentence.split())
                assert word_count <= 20, f"Sentence too long: {sentence}"
    
    def test_reference_removal(self):
        from src.voice.optimizer import VoiceOptimizer
        
        optimizer = VoiceOptimizer()
        
        text = "Check the settings (see Table 7-4) and verify the connection."
        optimized = optimizer.optimize(text)
        
        assert "Table 7-4" not in optimized
        assert "(see" not in optimized
    
    def test_language_simplification(self):
        from src.voice.optimizer import VoiceOptimizer
        
        optimizer = VoiceOptimizer()
        
        text = "Ensure you utilize the correct module."
        optimized = optimizer.optimize(text)
        
        assert "make sure" in optimized.lower() or "use" in optimized.lower()


class TestPhoneticConverter:
    """Tests for phonetic conversion."""
    
    def test_raid_conversion(self):
        from src.voice.phonetic import PhoneticConverter
        
        converter = PhoneticConverter()
        
        text = "Configure RAID5 for the storage array."
        converted = converter.convert(text)
        
        assert "raid five" in converted.lower()
    
    def test_sfp_plus_conversion(self):
        from src.voice.phonetic import PhoneticConverter
        
        converter = PhoneticConverter()
        
        text = "Install the SFP+ module."
        converted = converter.convert(text)
        
        assert "S-F-P plus" in converted
    
    def test_ip_address_formatting(self):
        from src.voice.phonetic import PhoneticConverter
        
        converter = PhoneticConverter()
        
        text = "Set the IP to 192.168.1.1"
        formatted = converter.format_number_sequences(text)
        
        assert "192 dot 168 dot 1 dot 1" in formatted


class TestHybridSearch:
    """Tests for hybrid search (without index)."""
    
    def test_rrf_fusion(self):
        from src.retrieval.hybrid_merger import HybridMerger, SearchResult
        
        # Mock results
        dense_results = [
            ("doc1", "Power supply content", 0.9),
            ("doc2", "Network content", 0.8),
            ("doc3", "Storage content", 0.7),
        ]
        
        sparse_results = [
            ("doc2", "Network content", 5.0),
            ("doc1", "Power supply content", 3.0),
            ("doc4", "Memory content", 2.0),
        ]
        
        merger = HybridMerger()
        results = merger._rrf_fusion(dense_results, sparse_results, top_k=3)
        
        # doc1 and doc2 should be top since they appear in both
        top_ids = [r.doc_id for r in results[:2]]
        assert "doc1" in top_ids
        assert "doc2" in top_ids


class TestFillerGenerator:
    """Tests for filler generation."""
    
    def test_contextual_filler(self):
        from src.generation.filler_generator import FillerGenerator
        
        generator = FillerGenerator()
        
        filler = generator.get_filler(query_type="troubleshooting", domain="power")
        assert len(filler) > 0
        assert isinstance(filler, str)


class TestLatencyMetrics:
    """Tests for latency tracking."""
    
    def test_ttfb_calculation(self):
        from src.pipeline.metrics import LatencyMetrics
        
        metrics = LatencyMetrics(request_id="test")
        
        metrics.mark('asr_start')
        time.sleep(0.01)
        metrics.mark('tts_first_byte')
        
        assert metrics.ttfb is not None
        assert metrics.ttfb > 0
    
    def test_breakdown_calculation(self):
        from src.pipeline.metrics import LatencyMetrics
        
        metrics = LatencyMetrics(request_id="test")
        
        metrics.mark('asr_start')
        time.sleep(0.01)
        metrics.mark('asr_end')
        
        breakdown = metrics.get_breakdown()
        assert 'asr' in breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
