# backend/domain/entities.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class RiskLevel(str, Enum):
    """Domenski enum za nivoe rizika od umora"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SessionStatus(str, Enum):
    """Statusi sesije za agentički ciklus"""
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    REVIEW_NEEDED = "review_needed"

class Position(str, Enum):
    """Pozicije igrača"""
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"

class ActivityType(str, Enum):
    """Tip aktivnosti"""
    PRACTICE = "practice"
    GAME = "game"

class PlayerAction(str, Enum):
    """Akcije koje agent može preporučiti"""
    CLEARED = "cleared"
    MONITOR = "monitor"
    REDUCE_INTENSITY = "reduce_intensity"
    REST_RECOMMENDED = "rest_recommended"
    MUST_REST = "must_rest"

@dataclass
class TrainingSession:
    """Domenska entitet - Treninška sesija - UPDATED sa novim features"""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    player_name: str = ""
    position: Position = Position.MIDFIELDER
    activity_type: ActivityType = ActivityType.PRACTICE
    sleep_hours: float = 0.0
    stress_level: int = 0
    distance_km: float = 0.0
    sprint_count: int = 0
    # NOVI FEATURES iz dataseta
    soreness: Optional[int] = None  # 1-10 scale
    rpe: Optional[int] = None  # Rate of Perceived Exertion 1-10
    injury_illness: Optional[bool] = None  # Da/Ne
    # Rezultati predikcije
    predicted_action: Optional[PlayerAction] = None
    fatigue_score: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    confidence: Optional[float] = None
    status: SessionStatus = SessionStatus.QUEUED
    
    @classmethod
    def create_new(cls, player_name: str, position: str, activity_type: str,
                   sleep_hours: float, stress_level: int, distance_km: float,
                   sprint_count: int, soreness: int = None, rpe: int = None, 
                   injury_illness: bool = None) -> 'TrainingSession':
        """Factory metoda za kreiranje nove sesije - UPDATED"""
        return cls(
            timestamp=datetime.now(),
            player_name=player_name,
            position=Position(position.lower()),
            activity_type=ActivityType(activity_type.lower()),
            sleep_hours=sleep_hours,
            stress_level=stress_level,
            distance_km=distance_km,
            sprint_count=sprint_count,
            soreness=soreness,
            rpe=rpe,
            injury_illness=injury_illness
        )
    
    def extract_features(self) -> list:
        """Ekstraktuj features za ML model - UPDATED"""
        return [
            self.position.value,
            self.activity_type.value,
            self.sleep_hours,
            self.stress_level,
            self.distance_km,
            self.sprint_count,
            self.soreness if self.soreness is not None else 5,
            self.rpe if self.rpe is not None else 5,
            1 if self.injury_illness else 0
        ]

@dataclass
class FatiguePrediction:
    """Domenska entitet - Predikcija agenta"""
    session_id: int
    action: PlayerAction
    fatigue_score: float
    risk_level: RiskLevel
    confidence: float
    requires_review: bool = False
    is_exploring: bool = False

@dataclass
class SystemSettings:
    """Domenska entitet - Postavke sistema"""
    id: int = 1
    gold_threshold: int = 50
    enable_retraining: bool = True
    new_gold_since_last_train: int = 0
    exploration_rate: float = 0.05
    low_risk_threshold: float = 40.0
    medium_risk_threshold: float = 60.0
    high_risk_threshold: float = 80.0