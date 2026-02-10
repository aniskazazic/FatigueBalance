# backend/web/dtos.py
from pydantic import BaseModel, Field
from typing import Optional

class SessionRequest(BaseModel):
    """DTO za novu trening sesiju - UPDATED sa novim optional fields"""
    player_name: str = Field(..., description="Ime igrača")
    position: str = Field(..., description="goalkeeper/defender/midfielder/forward")
    activity_type: str = Field(..., description="practice ili game")
    sleep_hours: float = Field(..., ge=0, le=12, description="Sati spavanja")
    stress_level: int = Field(..., ge=1, le=10, description="Nivo stresa 1-10")
    distance_km: float = Field(..., ge=0, description="Pretrcana distanca u km")
    sprint_count: int = Field(..., ge=0, description="Broj sprinteva")
    # NOVI OPTIONAL FIELDS iz dataseta
    soreness: Optional[int] = Field(None, ge=1, le=10, description="Bol u mišićima 1-10")
    rpe: Optional[int] = Field(None, ge=1, le=10, description="Rate of Perceived Exertion 1-10")
    injury_illness: Optional[bool] = Field(None, description="Prethodna povreda/bolest")
    
    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Edin Džeko",
                "position": "forward",
                "activity_type": "game",
                "sleep_hours": 7.0,
                "stress_level": 6,
                "distance_km": 10.5,
                "sprint_count": 35,
                "soreness": 5,
                "rpe": 7,
                "injury_illness": False
            }
        }

class FeedbackRequest(BaseModel):
    """DTO za feedback od trenera"""
    session_id: int = Field(..., description="ID sesije")
    user_label: str = Field(..., description="Pravi fatigue score ili action")
    correct: bool = Field(..., description="Da li je predikcija bila tačna")
    comment: Optional[str] = Field(None, description="Dodatni komentar")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": 123,
                "user_label": "75.0",
                "correct": False,
                "comment": "Igrač je bio umorniji nego što je agent predvidio"
            }
        }

class QueueResponse(BaseModel):
    """Odgovor nakon stavljanja u queue"""
    status: str
    session_id: int
    message: str
    timestamp: str
    estimated_wait_time_ms: Optional[float] = None

class PredictionResultResponse(BaseModel):
    """Rezultat predikcije"""
    session_id: int
    status: str
    predicted_action: Optional[str] = None
    fatigue_score: Optional[float] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    processed_at: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class AgentStatusResponse(BaseModel):
    """Status agenta"""
    is_running: bool
    processed_count: int
    avg_processing_time_ms: float
    queue_size: int = 0