# backend/application/services/scoring_service.py
import random
from typing import Tuple
from domain.entities import TrainingSession, PlayerAction, RiskLevel, FatiguePrediction

class FatigueScoringService:
    """Servis za scoring - implementira THINK fazu"""
    
    def __init__(self, classifier, exploration_rate: float = 0.05):
        self.classifier = classifier
        self.exploration_rate = exploration_rate
        
        # Pragovi za risk levels (mogu se učitati iz SystemSettings)
        self.low_threshold = 40.0
        self.medium_threshold = 60.0
        self.high_threshold = 80.0
    
    def score_session(self, session: TrainingSession) -> FatiguePrediction:
        """
        THINK fazu: izračunaj predikciju na osnovu sesije
        
        Returns: FatiguePrediction objekat sa svim detaljima
        """
        # Ekstraktuj features
        features = session.extract_features()
        
        # ML predikcija fatigue score-a (0-100)
        fatigue_score, confidence = self.classifier.predict(features)
        
        # Klasifikuj risk level na osnovu score-a
        risk_level = self._classify_risk(fatigue_score)
        
        # Odredi akciju na osnovu risk levela
        ml_action = self._risk_to_action(risk_level)
        
        # Exploration: nasumično probaj drugu akciju
        is_exploring = random.random() < self.exploration_rate
        if is_exploring:
            final_action = self._explore(ml_action)
            source = "exploration"
        else:
            final_action = ml_action
            source = "ml"
        
        # Da li zahtijeva review (nesigurni slučajevi)
        requires_review = self._requires_review(session, fatigue_score, confidence)
        
        return FatiguePrediction(
            session_id=session.id,
            action=final_action,
            fatigue_score=fatigue_score,
            risk_level=risk_level,
            confidence=confidence,
            requires_review=requires_review,
            is_exploring=is_exploring
        )
    
    def _classify_risk(self, fatigue_score: float) -> RiskLevel:
        """Klasifikuj risk level na osnovu fatigue score-a"""
        if fatigue_score < self.low_threshold:
            return RiskLevel.LOW
        elif fatigue_score < self.medium_threshold:
            return RiskLevel.MEDIUM
        elif fatigue_score < self.high_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _risk_to_action(self, risk_level: RiskLevel) -> PlayerAction:
        """Mapiranje risk levela na akciju"""
        mapping = {
            RiskLevel.LOW: PlayerAction.CLEARED,
            RiskLevel.MEDIUM: PlayerAction.MONITOR,
            RiskLevel.HIGH: PlayerAction.REDUCE_INTENSITY,
            RiskLevel.CRITICAL: PlayerAction.MUST_REST
        }
        return mapping[risk_level]
    
    def _explore(self, current_action: PlayerAction) -> PlayerAction:
        """Eksploracija: izaberi nasumičnu drugu akciju"""
        all_actions = list(PlayerAction)
        other_actions = [act for act in all_actions if act != current_action]
        return random.choice(other_actions) if other_actions else current_action
    
    def _requires_review(self, session: TrainingSession, fatigue_score: float,
                        confidence: float) -> bool:
        """Odluči da li slučaj zahtijeva human review"""
        
        # Nizak confidence
        if confidence < 0.65:
            return True
        
        # Ekstremne vrijednosti
        if session.sleep_hours <= 6.0 or session.sleep_hours > 10.0:
            return True
        
        if session.stress_level > 7:
            return True
        
        # Ekstremna distanca
        if session.distance_km > 10.0:
            return True
        
        # Na granici između risk levels
        threshold_margins = [self.low_threshold, self.medium_threshold, self.high_threshold]
        for threshold in threshold_margins:
            if abs(fatigue_score - threshold) < 3.0:  # Margin od 3
                return True
        
        return False
    
    def update_thresholds(self, low: float, medium: float, high: float):
        """Ažuriraj pragove za risk classification"""
        self.low_threshold = low
        self.medium_threshold = medium
        self.high_threshold = high