# backend/bootstrap.py

"""
BOOTSTRAP MODUL - Factory za kreiranje aplikacije
Odgovornosti:
- Kreiranje i konfiguracija svih komponenti sistema
- Dependency Injection setup
- Inicijalizacija baze i ML modela


"""
import logging
from typing import Optional
from infrastructure.system_init import InfrastructureSystemInit

logger = logging.getLogger(__name__)

def create_app():
    """
    Factory funkcija za kreiranje FastAPI aplikacije.
    OVO JE KLJUČNO: Web layer samo poziva ovu funkciju.
    """
    from fastapi import FastAPI
    from web.main import create_fastapi_app
    
    logger.info("="*70)
    logger.info("🚀 BOOTSTRAP: Pokrećem inicijalizaciju sistema...")
    logger.info("="*70)
    
    try:
        # 1. Inicijalizuj CIJELI sistem - ovo je SRŽ profesorovih zahtjeva!
        bootstrap = InfrastructureSystemInit()
        
        # Sve komponente se kreiraju OVDE, ne u web layeru!
        container = bootstrap.initialize_system(
            model_file="fatigue_model.joblib",
            exploration_rate=0.05,
            gold_threshold=10
        )
        
        if not container:
            raise RuntimeError("❌ Sistem se nije uspio inicijalizovati!")
        
        logger.info("✅ BOOTSTRAP: Sistem uspješno inicijalizovan!")
        logger.info("   ✓ Baza podataka")
        logger.info("   ✓ ML Classifier")
        logger.info("   ✓ DI Container")
        logger.info("   ✓ Servisi (Queue, Scoring)")
        logger.info("   ✓ Runneri (Scoring, Retrain)")
        logger.info("   ✓ Agent Manager")
        logger.info("="*70)
        
        # 2. Kreiraj FastAPI aplikaciju sa container-om
        # OVO JE ČISTA KONFIGURACIJA - što profesor traži!
        app = create_fastapi_app(container)
        
        return app
        
    except Exception as e:
        logger.error(f"❌ FATALNA GREŠKA u bootstrap-u: {e}")
        raise

def get_system_container():
    """
    Factory za dobijanje system container-a.
    Koristi se za testove ili CLI naredbe.
    """
    bootstrap = InfrastructureSystemInit()
    return bootstrap.initialize_system()

if __name__ == "__main__":
    print("🔧 Ovo je bootstrap modul. Koristi se kao:")
    print("   from backend.bootstrap import create_app")
    print("   app = create_app()")