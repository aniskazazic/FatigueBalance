#infrastructure/models.py
from sqlalchemy import Column, Integer, Float, String
from infrastructure.database import Base

class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)

    position = Column(String)
    session_type = Column(String)
    distance = Column(Float)
    acceleration_count = Column(Integer)
    sleep_hours = Column(Float)
    stress = Column(Float)
    soreness = Column(Float)
    rpe = Column(Float)
    injury = Column(String)

    fatigue_score = Column(Float, nullable=True)
    fatigue_status = Column(String, nullable=True)
    recommended_action = Column(String, nullable=True)

    status = Column(String, default="QUEUED")
