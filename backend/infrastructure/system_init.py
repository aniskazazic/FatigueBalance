#INFRASTRUCTURE/SYSTEM_INIT.PY
"""
SYSTEM INIT - Bootstrap za inicijalizaciju sistema
Odgovornosti:
- Inicijalizacija baze podataka
- Kreiranje ML modela  
- Kreiranje svih servisa i runnera
- Kreiranje DI Container-a
"""
import logging
from typing import Optional
from .database import init_database
from .ml.classifier import FatigueClassifier

logger = logging.getLogger(__name__)

class SystemContainer:
    """
    DEPENDENCY INJECTION CONTAINER
    Drži sve komponente sistema.
    Web layer samo koristi ovaj kontejner.
    """
    
    def __init__(self, classifier: FatigueClassifier):
        self.classifier = classifier
        self._agent_manager = None
    
    def set_agent_manager(self, agent_manager):
        """Postavi agent manager (post-construct pattern)"""
        self._agent_manager = agent_manager
    
    def get_agent_manager(self):
        """Vrati agent manager"""
        return self._agent_manager
    
    def is_ready(self) -> bool:
        """Provjeri da li je sistem spreman"""
        return self._agent_manager is not None

class InfrastructureSystemInit:
    """
    GLAVNI BOOTSTRAP - Inicijalizuje CIJELI sistem.
    Ovo je core profesorovih zahtjeva!
    """
    
    def __init__(self):
        self._db_initialized = False
        self._classifier: Optional[FatigueClassifier] = None
        self._container: Optional[SystemContainer] = None
    
    def initialize_system(self, 
                         model_file: str = "fatigue_model.joblib",
                         exploration_rate: float = 0.05,
                         gold_threshold: int = 10) -> Optional[SystemContainer]:
        """
        GLAVNA METODA: Inicijalizuje CIJELI sistem.
        
        Returns: SystemContainer sa svim komponentama
        """
        logger.info("="*70)
        logger.info("🏗️  INFRASTRUCTURE: Pokrećem inicijalizaciju sistema...")
        logger.info("="*70)
        
        # 1. Inicijalizuj bazu podataka
        if not self._initialize_database():
            logger.error("❌ Sistem se ne može pokrenuti bez baze!")
            return None
        
        # 2. Inicijalizuj ML model
        if not self._initialize_ml_model(model_file):
            logger.error("❌ Sistem se ne može pokrenuti bez ML modela!")
            return None
        
        # 3. Kreiraj DI Container
        container = SystemContainer(self._classifier)
        
        # 4. Kreiraj sve servise, runnere i Agent Manager
        if not self._initialize_services_and_agents(container, exploration_rate, gold_threshold):
            logger.error("❌ Sistem se ne može pokrenuti bez servisa!")
            return None
        
        self._container = container
        
        logger.info("="*70)
        logger.info("✅ INFRASTRUCTURE: Sistem potpuno inicijalizovan!")
        logger.info("="*70)
        
        return container
    
    def _initialize_database(self) -> bool:
        """Inicijalizuj bazu podataka"""
        logger.info("📦 Inicijalizacija baze podataka...")
        
        try:
            success = init_database()
            
            if success:
                self._db_initialized = True
                logger.info("✅ Baza podataka spremna")
            else:
                logger.error("❌ Neuspješna inicijalizacija baze!")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Greška pri inicijalizaciji baze: {e}")
            return False
    
    def _initialize_ml_model(self, model_file: str) -> bool:
        """Inicijalizuj ML model"""
        logger.info("🤖 Učitavanje ML modela...")
        
        try:
            self._classifier = FatigueClassifier(model_file=model_file)
            logger.info("✅ ML model spreman")
            
            # Log info o modelu
            model_info = self._classifier.get_model_info()
            logger.info(f"   Model tip: {model_info['model_type']}")
            logger.info(f"   Features: {model_info['n_features']}")
            logger.info(f"   File: {model_info['model_file']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Greška pri učitavanju ML modela: {e}")
            return False
    
    def _initialize_services_and_agents(self, container: SystemContainer,
                                      exploration_rate: float, 
                                      gold_threshold: int) -> bool:
        """
        Kreiraj sve servise, runnere i Agent Manager.
        OVO JE KLJUČNO: Sve se kreira OVDE, ne u web layeru!
        """
        logger.info("⚙️ Kreiranje servisa, runnera i Agent Manager-a...")
        
        try:
            # Importi ovdje da izbjegnemo circular dependencies
            # OVO JE ISPRAVAN DIZAJN!
            from application.services.queue_service import QueueService
            from application.services.scoring_service import FatigueScoringService
            from application.runners.scoring_runner import ScoringAgentRunner
            from application.runners.retrain_runner import RetrainAgentRunner
            from application.agent_manager import AgentManager
            
            # 1. Kreiraj servise
            queue_service = QueueService()
            scoring_service = FatigueScoringService(
                container.classifier,
                exploration_rate=exploration_rate
            )
            
            logger.info("   ✓ Queue Service kreiran")
            logger.info("   ✓ Scoring Service kreiran")
            
            # 2. Kreiraj runnere
            scoring_runner = ScoringAgentRunner(queue_service, scoring_service)
            retrain_runner = RetrainAgentRunner(
                container.classifier,
                gold_threshold=gold_threshold
            )
            
            logger.info("   ✓ Scoring Runner kreiran")
            logger.info("   ✓ Retrain Runner kreiran")
            
            # 3. Kreiraj Agent Manager i inicijalizuj ga
            agent_manager = AgentManager(container.classifier)
            agent_manager.initialize_services(
                exploration_rate=exploration_rate,
                gold_threshold=gold_threshold
            )
            
            # 4. Postavi agent manager u container
            container.set_agent_manager(agent_manager)
            
            logger.info("   ✓ Agent Manager kreiran i inicijalizovan")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            logger.error("Provjerite da li su svi moduli instalirani!")
            return False
        except Exception as e:
            logger.error(f"❌ Greška pri kreiranju servisa: {e}")
            return False
    
    def get_system_container(self) -> Optional[SystemContainer]:
        """Vrati system container (ako je inicijalizovan)"""
        return self._container
    
    def is_ready(self) -> bool:
        """Provjeri da li je sistem spreman"""
        return (self._db_initialized and 
                self._classifier is not None and 
                self._container is not None)
    
    def get_status(self) -> dict:
        """Vrati status infrastrukture"""
        return {
            "database_initialized": self._db_initialized,
            "ml_model_loaded": self._classifier is not None,
            "system_container_ready": self._container is not None,
            "ready": self.is_ready()
        }