# backend/application/services/queue_service.py - FIXED FOR SQL SERVER + NEW FIELDS
from typing import Optional
from domain.entities import TrainingSession, SessionStatus
from infrastructure.database import get_connection
import logging

logger = logging.getLogger(__name__)

class QueueService:
    """Servis za upravljanje redom (queue) trening sesija"""
    
    def enqueue(self, session: TrainingSession) -> TrainingSession:
        """Stavi sesiju u red za obradu - UPDATED sa novim fields"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO TrainingSessions 
                (Timestamp, PlayerName, Position, ActivityType, SleepHours, 
                 StressLevel, DistanceKm, SprintCount, Soreness, RPE, InjuryIllness, Status)
                OUTPUT INSERTED.Id
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.timestamp, session.player_name,
                session.position.value, session.activity_type.value,
                session.sleep_hours, session.stress_level,
                session.distance_km, session.sprint_count,
                session.soreness, session.rpe, session.injury_illness,
                SessionStatus.QUEUED.value
            ))
            
            session.id = cursor.fetchone()[0]
            conn.commit()
            
            logger.info(f"✅ Sesija #{session.id} ({session.player_name}) stavljena u queue")
            return session
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Greška pri enqueue: {e}")
            raise e
        finally:
            conn.close()
    
    def dequeue_next(self) -> Optional[TrainingSession]:
        """Uzmi sljedeću sesiju iz reda - FIXED SQL SERVER SYNTAX + NEW FIELDS"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            # FIXED: Prvo SELECT TOP 1 sa lock hint
            cursor.execute("""
                SELECT TOP 1 Id 
                FROM TrainingSessions WITH (UPDLOCK, READPAST)
                WHERE Status = 'queued'
                ORDER BY Timestamp ASC
            """)
            
            id_row = cursor.fetchone()
            if not id_row:
                conn.commit()
                return None
            
            session_id = id_row[0]
            
            # Onda UPDATE i OUTPUT - UPDATED sa novim fields
            cursor.execute("""
                UPDATE TrainingSessions
                SET Status = 'processing'
                OUTPUT INSERTED.Id, INSERTED.Timestamp, INSERTED.PlayerName,
                       INSERTED.Position, INSERTED.ActivityType, 
                       INSERTED.SleepHours, INSERTED.StressLevel,
                       INSERTED.DistanceKm, INSERTED.SprintCount,
                       INSERTED.Soreness, INSERTED.RPE, INSERTED.InjuryIllness
                WHERE Id = ?
            """, session_id)
            
            row = cursor.fetchone()
            conn.commit()
            
            if not row:
                return None
            
            from domain.entities import Position, ActivityType
            
            return TrainingSession(
                id=row[0],
                timestamp=row[1],
                player_name=row[2],
                position=Position(row[3]),
                activity_type=ActivityType(row[4]),
                sleep_hours=row[5],
                stress_level=row[6],
                distance_km=row[7],
                sprint_count=row[8],
                soreness=row[9],
                rpe=row[10],
                injury_illness=bool(row[11]) if row[11] is not None else None,
                status=SessionStatus.PROCESSING
            )
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Greška pri dequeue: {e}")
            return None
        finally:
            conn.close()
    
    def mark_as_processed(self, session_id: int, action: str, 
                         fatigue_score: float, risk_level: str, confidence: float):
        """Označi sesiju kao obrađenu"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE TrainingSessions 
                SET PredictedAction = ?, 
                    FatigueScore = ?,
                    RiskLevel = ?,
                    Confidence = ?,
                    Status = 'processed'
                WHERE Id = ?
            """, (action, fatigue_score, risk_level, confidence, session_id))
            
            conn.commit()
            logger.info(f"✅ Sesija #{session_id} processed: {action} (fatigue: {fatigue_score:.1f})")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Greška pri mark_as_processed: {e}")
            
        finally:
            conn.close()