# backend/application/agent_manager.py
"""
Agent Manager - upravlja lifecycle-om agenata.
Odgovornosti:
- Pokretanje i zaustavljanje scoring agenta
- Pokretanje i zaustavljanje retrain agenta
- Praćenje statusa oba agenta
"""
import asyncio
import logging
from typing import Optional
from .services.queue_service import QueueService
from .services.scoring_service import FatigueScoringService
from .runners.scoring_runner import ScoringAgentRunner
from .runners.retrain_runner import RetrainAgentRunner
from infrastructure.ml.classifier import FatigueClassifier
from infrastructure.ml.risk_classifier import RiskClassifier

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Centralizovano upravljanje agentima.
    Razdvaja odgovornost upravljanja agentima od web sloja.
    """
    
    def __init__(self, classifier: FatigueClassifier):
        """
        Args:
            classifier: ML model za predikcije
        """
        self.classifier = classifier
        
        # Servisi
        self.queue_service: Optional[QueueService] = None
        self.scoring_service: Optional[FatigueScoringService] = None
        
        # Runneri
        self.scoring_runner: Optional[ScoringAgentRunner] = None
        self.retrain_runner: Optional[RetrainAgentRunner] = None
        
        # Background tasks
        self._scoring_task: Optional[asyncio.Task] = None
        self._retrain_task: Optional[asyncio.Task] = None
        self._agents_running = False
    
    def initialize_services(self, exploration_rate: float = 0.05, 
                           gold_threshold: int = 10):
        """
        Inicijalizuj servise i runnere.
        
        Args:
            exploration_rate: Stopa eksploracije za scoring (0.0-1.0)
            gold_threshold: Broj feedback-a potrebnih za retrain
        """
        logger.info("⚙️ Kreiranje servisa i runnera...")
        
        # Kreiraj servise
        self.queue_service = QueueService()
        self.risk_classifier = RiskClassifier()
        self.scoring_service = FatigueScoringService(
            self.classifier,
            exploration_rate=exploration_rate,
            risk_classifier=self.risk_classifier
        )
        
        # Kreiraj runnere
        self.scoring_runner = ScoringAgentRunner(
            self.queue_service,
            self.scoring_service
        )
        self.retrain_runner = RetrainAgentRunner(
            self.classifier,
            risk_classifier=self.risk_classifier,
            gold_threshold=gold_threshold
        )
        
        logger.info("✅ Servisi i runneri spremni")
    
    async def start_agents(self):
        """Pokreni oba background agenta"""
        if self._agents_running:
            logger.warning("⚠️ Agenti su već pokrenuti")
            return
        
        if not self.scoring_runner or not self.retrain_runner:
            raise RuntimeError("Servisi nisu inicijalizovani! Pozovi initialize_services() prvo.")
        
        logger.info("🤖 Pokretanje background agenata...")
        self._agents_running = True
        
        # Pokreni scoring loop
        self._scoring_task = asyncio.create_task(self._run_scoring_loop())
        
        # Pokreni retrain loop
        self._retrain_task = asyncio.create_task(self._run_retrain_loop())
        
        logger.info("✅ Oba agenta pokrenuta (Scoring + Retrain)")
    
    async def stop_agents(self):
        """Zaustavi oba agenta"""
        if not self._agents_running:
            logger.warning("⚠️ Agenti nisu pokrenuti")
            return
        
        logger.info("🛑 Zaustavljanje agenata...")
        self._agents_running = False
        
        # Zaustavi scoring agent
        if self._scoring_task:
            self._scoring_task.cancel()
            try:
                await self._scoring_task
            except asyncio.CancelledError:
                logger.info("🤖 Scoring agent zaustavljen")
        
        # Zaustavi retrain agent
        if self._retrain_task:
            self._retrain_task.cancel()
            try:
                await self._retrain_task
            except asyncio.CancelledError:
                logger.info("🎓 Retrain agent zaustavljen")
        
        logger.info("✅ Svi agenti zaustavljeni")
    
    async def _run_scoring_loop(self):
        """Background loop za scoring agenta"""
        logger.info("🤖 Scoring agent loop pokrenut")
        
        try:
            while self._agents_running and self.scoring_runner:
                try:
                    result = self.scoring_runner.step()
                    
                    if result:
                        # Ima posla, kratka pauza
                        await asyncio.sleep(0.05)
                    else:
                        # Nema posla, duža pauza
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.error(f"🤖 Greška u scoring loopu: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("🤖 Scoring agent loop prekinut")
        except Exception as e:
            logger.error(f"🤖 Kritična greška: {e}")
        finally:
            logger.info("🤖 Scoring agent loop završen")
    
    async def _run_retrain_loop(self):
        """Background loop za retrain agenta"""
        logger.info("🎓 Retrain agent loop pokrenut")
        
        try:
            while self._agents_running and self.retrain_runner:
                try:
                    result = self.retrain_runner.step()
                    
                    if result:
                        logger.info(f"✅ Retrain completed: {result.message}")
                        # Čekaj 30s nakon retrain-a
                        await asyncio.sleep(30)
                    else:
                        # Provjeri svaki 10s
                        await asyncio.sleep(10)
                        
                except Exception as e:
                    logger.error(f"🎓 Greška u retrain loopu: {e}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info("🎓 Retrain agent loop prekinut")
        except Exception as e:
            logger.error(f"🎓 Kritična greška: {e}")
        finally:
            logger.info("🎓 Retrain agent loop završen")
    
    def is_running(self) -> bool:
        """Provjeri da li su agenti aktivni"""
        return self._agents_running
    
    def get_status(self) -> dict:
        """Dohvati status agenata i metrike"""
        scoring_status = {}
        retrain_status = {}
        
        if self.scoring_runner:
            scoring_status = self.scoring_runner.get_status()
        
        if self.retrain_runner:
            retrain_status = self.retrain_runner.get_status()
        
        return {
            "agents_running": self._agents_running,
            "scoring_agent": scoring_status,
            "retrain_agent": retrain_status
        }


    def get_queue_service(self):
        """Vrati queue service (za web layer dependency)"""
        return self.queue_service
