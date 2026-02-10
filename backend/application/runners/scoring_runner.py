# backend/application/runners/scoring_runner.py
from typing import Optional
from dataclasses import dataclass
from domain.entities import SessionStatus
from application.services.queue_service import QueueService
from application.services.scoring_service import FatigueScoringService
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScoringTickResult:
    """Rezultat jednog tick-a"""
    session_id: int
    player_name: str
    action: str
    fatigue_score: float
    risk_level: str
    confidence: float
    requires_review: bool
    is_exploring: bool
    processing_time_ms: float
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "player_name": self.player_name,
            "action": self.action,
            "fatigue_score": self.fatigue_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "is_exploring": self.is_exploring,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": time.time()
        }

class ScoringAgentRunner:
    """
    RUNNER koji implementira KOMPLETAN agentički ciklus:
    SENSE → THINK → ACT → LEARN
    """
    
    def __init__(self, queue_service: QueueService, 
                 scoring_service: FatigueScoringService):
        self.queue_service = queue_service
        self.scoring_service = scoring_service
        
        # Metrics za LEARN fazu
        self.processed_count = 0
        self.total_processing_time = 0
        self.exploration_count = 0
        self.low_confidence_count = 0
        self.review_needed_count = 0
        
        # Running averages
        self.avg_fatigue_score = 0
        self.avg_confidence = 0
        
    def step(self) -> Optional[ScoringTickResult]:
        """
        Izvrši JEDAN tick agentičkog ciklusa
        Vraća: ScoringTickResult ako ima posla, None ako nema
        """
        start_time = time.time()
        
        # ===== SENSE =====
        session = self.queue_service.dequeue_next()
        if not session:
            return None  # Nema posla
        
        # ===== THINK =====
        prediction = self.scoring_service.score_session(session)
        
        # ===== ACT =====
        self.queue_service.mark_as_processed(
            session_id=session.id,
            action=prediction.action.value,
            fatigue_score=prediction.fatigue_score,
            risk_level=prediction.risk_level.value,
            confidence=prediction.confidence
        )
        
        processing_time = (time.time() - start_time) * 1000  # u ms
        
        # ===== LEARN =====
        # OVO JE KLJUČNO! Profesor TRAŽI LEARN FAZU!
        self._learn_from_prediction(prediction, processing_time)
        
        result = ScoringTickResult(
            session_id=session.id,
            player_name=session.player_name,
            action=prediction.action.value,
            fatigue_score=prediction.fatigue_score,
            risk_level=prediction.risk_level.value,
            confidence=prediction.confidence,
            requires_review=prediction.requires_review,
            is_exploring=prediction.is_exploring,
            processing_time_ms=processing_time
        )
        
        logger.info(f"✅ Agent procesirao: {result.player_name} - {result.action} "
                   f"(fatigue: {result.fatigue_score:.1f}, conf: {result.confidence:.2f})")
        
        return result
    
    def _learn_from_prediction(self, prediction, processing_time: float):
        """
        LEARN FAZA - Ažuriraj metrike i uči iz predikcije
        OVO JE OBAVEZNO prema profesorovim uputama!
        """
        self.processed_count += 1
        self.total_processing_time += processing_time
        
        # Ažuriraj running averages
        alpha = 0.1  # Exponential moving average factor
        self.avg_fatigue_score = (alpha * prediction.fatigue_score + 
                                  (1 - alpha) * self.avg_fatigue_score)
        self.avg_confidence = (alpha * prediction.confidence + 
                              (1 - alpha) * self.avg_confidence)
        
        # Prati exploration
        if prediction.is_exploring:
            self.exploration_count += 1
            logger.info(f"🔬 Exploration #{self.exploration_count}: "
                       f"Tried {prediction.action.value} instead of ML prediction")
        
        # Prati low confidence cases
        if prediction.confidence < 0.7:
            self.low_confidence_count += 1
            logger.warning(f"⚠️ Low confidence prediction: {prediction.confidence:.2f}")
        
        # Prati review cases
        if prediction.requires_review:
            self.review_needed_count += 1
            logger.info(f"👁️ Case requires human review (total: {self.review_needed_count})")
        
        # Log learning insights every 10 sessions
        if self.processed_count % 10 == 0:
            self._log_learning_insights()
    
    def _log_learning_insights(self):
        """Log što je agent naučio"""
        exploration_rate = (self.exploration_count / self.processed_count * 100 
                           if self.processed_count > 0 else 0)
        review_rate = (self.review_needed_count / self.processed_count * 100 
                      if self.processed_count > 0 else 0)
        
        logger.info("="*70)
        logger.info("📚 LEARNING INSIGHTS:")
        logger.info(f"   Processed: {self.processed_count} sessions")
        logger.info(f"   Avg Fatigue: {self.avg_fatigue_score:.1f}")
        logger.info(f"   Avg Confidence: {self.avg_confidence:.2f}")
        logger.info(f"   Exploration Rate: {exploration_rate:.1f}%")
        logger.info(f"   Review Rate: {review_rate:.1f}%")
        logger.info("="*70)
    
    def get_status(self):
        """Vrati status runnera sa LEARN metrikama"""
        avg_time = (self.total_processing_time / self.processed_count 
                   if self.processed_count > 0 else 0)
        
        return {
            "processed_count": self.processed_count,
            "avg_processing_time_ms": avg_time,
            "total_processing_time_ms": self.total_processing_time,
            "is_active": True,
            # LEARN metrike
            "avg_fatigue_score": self.avg_fatigue_score,
            "avg_confidence": self.avg_confidence,
            "exploration_count": self.exploration_count,
            "low_confidence_count": self.low_confidence_count,
            "review_needed_count": self.review_needed_count
        }