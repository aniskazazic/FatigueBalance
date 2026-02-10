# backend/web/main.py

"""
WEB LAYER - FastAPI endpoints
ODGOVORNOSTI SAMO:
- Definisanje API endpoints-a
- Request/Response validacija
- DI konfiguracija (eksplicitno!)

ŠTA OVAJ LAYER NE RADI:
- NE kreira DB (to radi infrastructure)
- NE kreira ML model (to radi infrastructure)  
- NE kreira servise (to radi bootstrap)
- NE pokreće agente direktno (to radi lifespan)
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from .dtos import SessionRequest, FeedbackRequest
from domain.entities import TrainingSession
from infrastructure.database import save_feedback, get_session_status, get_connection

logger = logging.getLogger(__name__)

class QueueResponse(BaseModel):
    status: str
    session_id: int
    message: str
    timestamp: str
    estimated_wait_time_ms: Optional[float] = None

class PredictionResultResponse(BaseModel):
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
    is_running: bool
    processed_count: int
    avg_processing_time_ms: float
    queue_size: int = 0
    # LEARN metrike
    avg_fatigue_score: float = 0
    avg_confidence: float = 0
    exploration_count: int = 0
    review_needed_count: int = 0
    # Retrain metrike
    retrain_count: int = 0
    gold_threshold: int = 0

def create_fastapi_app(system_container):
    """
    Factory za FastAPI aplikaciju - PRAVA DI KONFIGURACIJA!
    Web layer samo konfiguriše DI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Lifecycle management - SAMO pokreće i zaustavlja agente.
        Sve ostalo je već inicijalizovano u bootstrap-u.
        """
        logger.info("🤖 WEB LAYER: Pokretanje background agenata...")
        
        try:
            # Agent manager je već kreiran u bootstrap-u!
            agent_manager = system_container.get_agent_manager()
            await agent_manager.start_agents()
            
            logger.info("✅ Agenti pokrenuti")
            
            yield
            
            logger.info("👋 WEB LAYER: Zaustavljanje agenata...")
            await agent_manager.stop_agents()
            logger.info("✅ Agenti zaustavljeni")
            
        except Exception as e:
            logger.error(f"❌ Greška u lifecycle-u: {e}")
            raise
    
    # Kreiraj FastAPI app sa lifespan-om
    app = FastAPI(
        title="FatigueAgent API - Clean Architecture FINAL",
        version="4.0",
        description="Minimal coupling - Factory pattern + DI Container",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ===== DEPENDENCY INJECTION KONFIGURACIJA =====
    # OVO JE KLJUČNO! Eksplicitno definisane zavisnosti. 
    def get_container():
        """Dependency za system container"""
        return system_container
    
    def get_agent_manager(container = Depends(get_container)):
        """Dependency za agent manager"""
        return container.get_agent_manager()
    
    def get_queue_service(agent_manager = Depends(get_agent_manager)):
        """Dependency za queue service"""
        return agent_manager.get_queue_service()
    
    # ===== API ENDPOINTS =====
    @app.post("/predict", response_model=QueueResponse)
    async def predict(
        session: SessionRequest,
        queue_service = Depends(get_queue_service)
    ):
        """Stavi sesiju u queue - KORISTI DI!"""
        if not queue_service:
            raise HTTPException(status_code=503, detail="Queue service not available")
        
        try:
            # Kreiranje domain entiteta
            training_session = TrainingSession.create_new(
                player_name=session.player_name,
                position=session.position,
                activity_type=session.activity_type,
                sleep_hours=session.sleep_hours,
                stress_level=session.stress_level,
                distance_km=session.distance_km,
                sprint_count=session.sprint_count,
                soreness=session.soreness,
                rpe=session.rpe,
                injury_illness=session.injury_illness
            )
            
            # Stavi u queue - queue_service je injektovan!
            saved_session = queue_service.enqueue(training_session)
            
            logger.info(f"📥 Sesija #{saved_session.id} ({saved_session.player_name}) stavljena u queue")
            
            return QueueResponse(
                status="queued",
                session_id=saved_session.id,
                message=f"Session for {saved_session.player_name} queued for processing",
                timestamp=datetime.now().isoformat(),
                estimated_wait_time_ms=100.0
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Greška u /predict: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/predictions/{session_id}", response_model=PredictionResultResponse)
    async def get_prediction_result(session_id: int):
        """Dohvati rezultat predikcije"""
        try:
            session_status = get_session_status(session_id)
            
            if not session_status:
                raise HTTPException(status_code=404, detail="Session not found")
            
            if session_status['status'] in ['queued', 'processing']:
                return PredictionResultResponse(
                    session_id=session_id,
                    status=session_status['status']
                )
            
            if session_status['status'] == 'processed':
                return PredictionResultResponse(
                    session_id=session_id,
                    status='processed',
                    predicted_action=session_status['predicted_action'],
                    fatigue_score=session_status['fatigue_score'],
                    risk_level=session_status['risk_level'],
                    confidence=session_status['confidence'],
                    processed_at=session_status['timestamp'].isoformat() if session_status['timestamp'] else None
                )
            
            return PredictionResultResponse(
                session_id=session_id,
                status=session_status['status'],
                error=f"Unexpected status: {session_status['status']}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Greška: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def feedback(fb: FeedbackRequest):
        """Primi feedback - retrain agent će procesirati"""
        try:
            success = save_feedback(
                session_id=fb.session_id,
                user_label=fb.user_label,
                correct=fb.correct,
                comment=fb.comment
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to save feedback")
            
            logger.info(f"💬 Feedback saved za sesiju #{fb.session_id} "
                       f"(correct: {fb.correct}, label: {fb.user_label})")
            
            return {
                "ok": True, 
                "message": "Feedback saved. Retrain agent will process it in next cycle."
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Greška: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/agent/status", response_model=AgentStatusResponse)
    async def get_agent_status(agent_manager = Depends(get_agent_manager)):
        """Dohvati status agenata sa LEARN metrikama"""
        if not agent_manager:
            return AgentStatusResponse(
                is_running=False,
                processed_count=0,
                avg_processing_time_ms=0,
                queue_size=0
            )
        
        try:
            # Dohvati queue size iz baze
            queue_size = 0
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM TrainingSessions WHERE Status = 'queued'")
                queue_size = cursor.fetchone()[0]
                conn.close()
            except:
                queue_size = 0
            
            # Dohvati status preko agent manager-a (INJEKTOVAN!)
            status = agent_manager.get_status()
            
            scoring_status = status.get("scoring_agent", {})
            retrain_status = status.get("retrain_agent", {})
            
            return AgentStatusResponse(
                is_running=status.get("agents_running", False),
                processed_count=scoring_status.get("processed_count", 0),
                avg_processing_time_ms=scoring_status.get("avg_processing_time_ms", 0),
                queue_size=queue_size,
                # LEARN metrike
                avg_fatigue_score=scoring_status.get("avg_fatigue_score", 0),
                avg_confidence=scoring_status.get("avg_confidence", 0),
                exploration_count=scoring_status.get("exploration_count", 0),
                review_needed_count=scoring_status.get("review_needed_count", 0),
                # Retrain metrike
                retrain_count=retrain_status.get("retrain_count", 0),
                gold_threshold=retrain_status.get("gold_threshold", 0)
            )
            
        except Exception as e:
            logger.error(f"❌ Greška: {e}")
            return AgentStatusResponse(
                is_running=False,
                processed_count=0,
                avg_processing_time_ms=0,
                queue_size=0
            )
    
    return app