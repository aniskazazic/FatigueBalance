# backend/application/runners/retrain_runner.py
from typing import Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from infrastructure.database import get_connection

logger = logging.getLogger(__name__)

@dataclass
class RetrainTickResult:
    """Rezultat retrain tick-a"""
    retrained: bool
    feedback_count: int
    model_updated: bool
    message: str
    last_retrain_date: Optional[datetime] = None

class RetrainAgentRunner:
    """
    Runner za periodični retrain modela
    SENSE → THINK → ACT → LEARN
    """
    
    def __init__(self, classifier, gold_threshold: int = 10):
        self.classifier = classifier
        self.gold_threshold = gold_threshold
        self.last_retrain_count = 0
        self.retrain_count = 0
        self._last_retrain_date = None
        
    def step(self) -> Optional[RetrainTickResult]:
        """
        SENSE → THINK → ACT → LEARN
        Jedan tick - provjeri da li treba retrain
        """
        
        # ===== SENSE =====
        # Koliko NOVOG feedbacka imamo od zadnjeg treninga?
        new_feedback_count = self._sense_new_feedback()
        
        if new_feedback_count == 0:
            return None  # Nema novog feedbacka, nema posla
        
        # ===== THINK =====
        # Da li je dovoljno feedbacka za retrain?
        should_retrain = new_feedback_count >= self.gold_threshold
        
        if not should_retrain:
            logger.info(f"📊 New feedback: {new_feedback_count}/{self.gold_threshold} "
                       f"(need {self.gold_threshold - new_feedback_count} more)")
            return None
        
        # ===== ACT =====
        # Izvuci sve feedback podatke i treniraj model
        logger.info(f"🔄 RETRAINING MODEL - {new_feedback_count} new feedback items")
        retrain_date, success = self._retrain_with_feedback()
        
        if not success:
            logger.error("❌ Retrain failed!")
            return RetrainTickResult(
                retrained=False,
                feedback_count=new_feedback_count,
                model_updated=False,
                message="Retrain failed",
                last_retrain_date=retrain_date if retrain_date else None
            )
        
        # ===== LEARN =====
        # Reset brojač, ažuriraj SystemSettings, loguj metrics
        self.retrain_count += 1
        self.last_retrain_count = new_feedback_count
        self._last_retrain_date = retrain_date
        self._learn_from_retrain(new_feedback_count, retrain_date)
        
        logger.info(f"✅ MODEL RETRAINED! (retrain #{self.retrain_count}, "
                   f"{new_feedback_count} feedback items, date: {retrain_date})")
        
        return RetrainTickResult(
            retrained=True,
            feedback_count=new_feedback_count,
            model_updated=True,
            message=f"Model retrained with {new_feedback_count} feedback items",
            last_retrain_date=retrain_date
        )
    
    def _sense_new_feedback(self) -> int:
        """SENSE: Koliko NOVIH feedback stavki imamo?"""
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Brojimo feedback koji nije korišten za trening
            cursor.execute("""
                SELECT COUNT(*) 
                FROM Feedback 
                WHERE Correct = 0 AND Processed = 0
            """)
            
            count = cursor.fetchone()[0]
            return count
            
        except Exception as e:
            # Ako kolona Processed ne postoji, vrati samo Correct = 0
            try:
                cursor.execute("SELECT COUNT(*) FROM Feedback WHERE Correct = 0")
                count = cursor.fetchone()[0]
                return count
            except:
                logger.error(f"❌ Error sensing feedback: {e}")
                return 0
        finally:
            if conn:
                conn.close()
    
    def _retrain_with_feedback(self) -> tuple[Optional[datetime], bool]:
        """
        ACT: Treniraj model sa svim incorrect feedback-om
        Vraća: (retrain_datetime, success)
        """
        conn = None
        retrain_date = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Zabilježi vrijeme retreniranja
            retrain_date = datetime.now()
            
            # Izvuci sve incorrect feedback sa session detaljima
            cursor.execute("""
                SELECT 
                    f.Id as FeedbackId,
                    f.UserLabel,
                    ts.Position, ts.ActivityType, ts.SleepHours, 
                    ts.StressLevel, ts.DistanceKm, ts.SprintCount,
                    ts.Soreness, ts.RPE, ts.InjuryIllness
                FROM Feedback f
                JOIN TrainingSessions ts ON f.SessionId = ts.Id
                WHERE f.Correct = 0 AND f.Processed = 0
            """)
            
            rows = cursor.fetchall()
            
            if len(rows) == 0:
                logger.warning("⚠️ No incorrect feedback to train on")
                return retrain_date, False
            
            logger.info(f"📚 Training on {len(rows)} feedback examples")
            
            # Treniraj model sa svakim feedback-om
            trained_count = 0
            for row in rows:
                try:
                    feedback_id = row[0]
                    user_label = float(row[1])
                    
                    features = [
                        row[2],  # Position
                        row[3],  # ActivityType
                        float(row[4]),  # SleepHours
                        float(row[5]),  # StressLevel
                        float(row[6]),  # DistanceKm
                        float(row[7]),  # SprintCount
                        float(row[8]) if row[8] is not None else 5.0,  # Soreness
                        float(row[9]) if row[9] is not None else 5.0,  # RPE
                        float(row[10]) if row[10] is not None else 0.0  # InjuryIllness
                    ]
                    
                    # KLJUČNO: Pozovi train_single za incremental learning
                    success = self.classifier.train_single(features, user_label)
                    
                    if success:
                        trained_count += 1
                        
                        # Označi feedback kao processed
                        cursor.execute(
                            "UPDATE Feedback SET Processed = 1 WHERE Id = ?",
                            (feedback_id,)
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Failed to train on feedback ID {row[0]}: {e}")
                    continue
            
            # 🌟 AŽURIRAJ LASTRETAINDATE U SystemSettings 🌟
            cursor.execute("""
                UPDATE SystemSettings 
                SET LastRetrainDate = ?, NewGoldSinceLastTrain = 0
                WHERE Id = 1
            """, (retrain_date,))
            
            conn.commit()
            
            logger.info(f"✅ Successfully trained on {trained_count}/{len(rows)} examples")
            logger.info(f"✅ SystemSettings ažuriran (LastRetrainDate = {retrain_date})")
            
            return retrain_date, trained_count > 0
            
        except Exception as e:
            logger.error(f"❌ Retrain error: {e}")
            return retrain_date, False
        finally:
            if conn:
                conn.close()
    
    def _learn_from_retrain(self, feedback_count: int, retrain_date: datetime):
        """LEARN: Ažuriraj metrike nakon retrain-a"""
        # Log learning insights
        logger.info("="*70)
        logger.info("🎓 RETRAIN LEARNING:")
        logger.info(f"   Retrain count: {self.retrain_count}")
        logger.info(f"   Feedback processed: {feedback_count}")
        logger.info(f"   Retrain date/time: {retrain_date}")
        logger.info(f"   Last retrain size: {self.last_retrain_count}")
        logger.info(f"   Gold threshold: {self.gold_threshold}")
        
        # Dodatne metrike o učenju
        model_info = self.classifier.get_learning_stats()
        logger.info(f"   Total feedback learned: {model_info.get('feedback_learned', 0)}")
        
        # Izvuci poslednji retrain date iz baze za potvrdu
        self._log_last_retrain_from_db()
        
        logger.info("="*70)
    
    def _log_last_retrain_from_db(self):
        """Pomoćna funkcija da proveri poslednji retrain date iz baze"""
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT LastRetrainDate FROM SystemSettings WHERE Id = 1")
            result = cursor.fetchone()
            if result and result[0]:
                db_date = result[0]
                logger.info(f"   DB LastRetrainDate: {db_date}")
            else:
                logger.info("   DB LastRetrainDate: NULL (first retrain)")
        except Exception as e:
            logger.error(f"❌ Error reading LastRetrainDate: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_last_retrain_date(self) -> Optional[datetime]:
        """Vraća poslednji datum retreniranja iz memorije"""
        return self._last_retrain_date
    
    def get_db_last_retrain_date(self) -> Optional[datetime]:
        """Vraća poslednji datum retreniranja iz baze podataka"""
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT LastRetrainDate FROM SystemSettings WHERE Id = 1")
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"❌ Error reading LastRetrainDate from DB: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_status(self):
        """Status retrain runnera"""
        db_retrain_date = self.get_db_last_retrain_date()
        
        return {
            "retrain_count": self.retrain_count,
            "gold_threshold": self.gold_threshold,
            "last_retrain_size": self.last_retrain_count,
            "memory_last_retrain_date": self._last_retrain_date,
            "db_last_retrain_date": db_retrain_date,
            "is_active": True,
            "feedback_awaiting": self._sense_new_feedback(),
            "time_since_last_retrain": self._get_time_since_last_retrain()
        }
    
    def _get_time_since_last_retrain(self) -> Optional[str]:
        """Izračunaj koliko je vremena prošlo od poslednjeg retreniranja"""
        retrain_date = self.get_db_last_retrain_date()
        if not retrain_date:
            return "Never retrained"
        
        now = datetime.now()
        delta = now - retrain_date
        
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h ago"
        elif hours > 0:
            return f"{hours}h {minutes}m ago"
        else:
            return f"{minutes}m ago"