# backend/infrastructure/database.py - FULL FIXED VERSION
import pyodbc
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_SERVER = "localhost"
DB_NAME = "FatigueAgentProba"

def create_database_if_not_exists():
    """
    Kreiraj bazu 'FatigueAgent' ako ne postoji.
    FIXED: autocommit=True za CREATE DATABASE
    """
    master_conn = None
    try:
        master_conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )
        
        logger.info(f"🔗 Povezivanje na SQL Server: {DB_SERVER}")
        # KLJUČNA PROMJENA: autocommit=True
        master_conn = pyodbc.connect(master_conn_str, autocommit=True)
        cursor = master_conn.cursor()
        
        logger.info(f"🔍 Provjeravam da li baza '{DB_NAME}' postoji...")
        cursor.execute(f"SELECT name FROM sys.databases WHERE name = '{DB_NAME}'")
        
        if not cursor.fetchone():
            logger.info(f"📦 Kreiranje baze '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"✅ Baza '{DB_NAME}' uspješno kreirana")
        else:
            logger.info(f"ℹ️ Baza '{DB_NAME}' već postoji")
        
        return True
        
    except pyodbc.InterfaceError as e:
        logger.error(f"❌ Nije moguće spojiti se na SQL Server: {e}")
        logger.error("Provjerite da li je SQL Server pokrenut")
        return False
    except Exception as e:
        logger.error(f"❌ Greška pri kreiranju baze: {e}")
        return False
    finally:
        if master_conn:
            master_conn.close()

def get_connection():
    """Vrati konekciju za FatigueAgent bazu"""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def init_database():
    """
    Kompletna inicijalizacija baze:
    1. Kreira bazu ako ne postoji
    2. Kreira sve tabele ako ne postoje
    3. Popuni SystemSettings ako je prazno
    """
    if not create_database_if_not_exists():
        logger.error("❌ Nije moguće nastaviti bez baze podataka")
        return False
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        logger.info("📊 Inicijalizacija tabela...")
        
        # Tabela TrainingSessions - SA NOVIM KOLONAMA!
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'TrainingSessions')
            BEGIN
                CREATE TABLE TrainingSessions (
                    Id INT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME NOT NULL DEFAULT GETDATE(),
                    PlayerName NVARCHAR(100) NOT NULL,
                    Position NVARCHAR(20) NOT NULL,
                    ActivityType NVARCHAR(20) NOT NULL,
                    SleepHours FLOAT NOT NULL,
                    StressLevel INT NOT NULL,
                    DistanceKm FLOAT NOT NULL,
                    SprintCount INT NOT NULL,
                    Soreness INT NULL,
                    RPE INT NULL,
                    InjuryIllness BIT NULL,
                    PredictedAction NVARCHAR(50) NULL,
                    FatigueScore FLOAT NULL,
                    RiskLevel NVARCHAR(20) NULL,
                    Status NVARCHAR(20) DEFAULT 'queued',
                    Confidence FLOAT NULL
                )
                PRINT 'Tabela TrainingSessions kreirana'
            END
            ELSE
            BEGIN
                -- Dodaj nove kolone ako ne postoje
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                              WHERE TABLE_NAME = 'TrainingSessions' AND COLUMN_NAME = 'Soreness')
                    ALTER TABLE TrainingSessions ADD Soreness INT NULL
                
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                              WHERE TABLE_NAME = 'TrainingSessions' AND COLUMN_NAME = 'RPE')
                    ALTER TABLE TrainingSessions ADD RPE INT NULL
                
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                              WHERE TABLE_NAME = 'TrainingSessions' AND COLUMN_NAME = 'InjuryIllness')
                    ALTER TABLE TrainingSessions ADD InjuryIllness BIT NULL
                
                PRINT 'Tabela TrainingSessions već postoji (ažurirane nove kolone ako su potrebne)'
            END
        """)
        
        # Tabela Feedback
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'Feedback')
            BEGIN
                CREATE TABLE Feedback (
                    Id INT IDENTITY(1,1) PRIMARY KEY,
                    SessionId INT NOT NULL,
                    UserLabel NVARCHAR(50) NOT NULL,
                    Correct BIT NOT NULL,
                    Comment NVARCHAR(255) NULL,
                    CreatedAt DATETIME DEFAULT GETDATE(),
                    FOREIGN KEY (SessionId) REFERENCES TrainingSessions(Id)
                )
                PRINT 'Tabela Feedback kreirana'
            END
            ELSE
                PRINT 'Tabela Feedback već postoji'
        """)
        
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                  WHERE TABLE_NAME = 'Feedback' AND COLUMN_NAME = 'Processed')
            BEGIN
                ALTER TABLE Feedback ADD Processed BIT DEFAULT 0
            PRINT 'Dodata Processed kolona u Feedback'
            END
        """)

        # Tabela SystemSettings
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'SystemSettings')
            BEGIN
                CREATE TABLE SystemSettings (
                    Id INT PRIMARY KEY DEFAULT 1,
                    GoldThreshold INT DEFAULT 50,
                    EnableRetraining BIT DEFAULT 1,
                    NewGoldSinceLastTrain INT DEFAULT 0,
                    ExplorationRate FLOAT DEFAULT 0.05,
                    LowRiskThreshold FLOAT DEFAULT 40.0,
                    MediumRiskThreshold FLOAT DEFAULT 60.0,
                    HighRiskThreshold FLOAT DEFAULT 80.0,
                    CONSTRAINT CK_SingleRow CHECK (Id = 1)
                )
                PRINT 'Tabela SystemSettings kreirana'
            END
            ELSE
                PRINT 'Tabela SystemSettings već postoji'
        """)
        
        # Popuni SystemSettings
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM SystemSettings WHERE Id = 1)
            BEGIN
                INSERT INTO SystemSettings DEFAULT VALUES
                PRINT 'SystemSettings popunjen podrazumijevanim vrijednostima'
            END
            ELSE
                PRINT 'SystemSettings već ima podatke'
        """)

        # Takođe dodaj kolonu LastRetrainDate u SystemSettings ako ne postoji
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                  WHERE TABLE_NAME = 'SystemSettings' AND COLUMN_NAME = 'LastRetrainDate')
            BEGIN
                ALTER TABLE SystemSettings ADD LastRetrainDate DATETIME NULL
                PRINT 'Dodata LastRetrainDate kolona u SystemSettings'
            END
        """)
        
        conn.commit()
        logger.info("🎉 Baza potpuno inicijalizirana!")
        return True
        
    except pyodbc.ProgrammingError as e:
        logger.error(f"❌ SQL greška: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ Neočekivana greška: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def save_session(player_name: str, position: str, activity_type: str,
                 sleep_hours: float, stress_level: int, distance_km: float,
                 sprint_count: int, predicted_action: str = None,
                 fatigue_score: float = None, risk_level: str = None,
                 confidence: float = None) -> Optional[int]:
    """Sačuvaj sesiju u bazu i vrati ID"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO TrainingSessions 
            (Timestamp, PlayerName, Position, ActivityType, SleepHours, 
             StressLevel, DistanceKm, SprintCount, PredictedAction, 
             FatigueScore, RiskLevel, Confidence, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processed')
        """, (datetime.now(), player_name, position, activity_type,
              sleep_hours, stress_level, distance_km, sprint_count,
              predicted_action, fatigue_score, risk_level, confidence))
        
        conn.commit()
        
        cursor.execute("SELECT @@IDENTITY")
        row = cursor.fetchone()
        session_id = row[0] if row else None
        
        logger.debug(f"📝 Sesija #{session_id} sačuvana")
        return session_id
        
    except Exception as e:
        logger.error(f"❌ Greška pri čuvanju sesije: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def save_feedback(session_id: int, user_label: str, 
                  correct: bool, comment: str = None) -> bool:
    """Sačuvaj feedback u bazu"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO Feedback (SessionId, UserLabel, Correct, Comment)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_label, int(correct), comment))
        
        conn.commit()
        logger.debug(f"💬 Feedback sačuvan za sesiju #{session_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Greška pri čuvanju feedbacka: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_session_status(session_id: int) -> Optional[Dict[str, Any]]:
    """Dohvati status i rezultat sesije"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT Id, Timestamp, PredictedAction, FatigueScore, 
                   RiskLevel, Confidence, Status
            FROM TrainingSessions 
            WHERE Id = ?
        """, session_id)
        
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'timestamp': row[1],
                'predicted_action': row[2],
                'fatigue_score': row[3],
                'risk_level': row[4],
                'confidence': row[5],
                'status': row[6]
            }
        return None
        
    except Exception as e:
        logger.error(f"❌ Greška pri dohvatanju statusa: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_session_details(session_id: int) -> Optional[Dict[str, Any]]:
    """Dohvati sve detalje sesije"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT Id, Timestamp, PlayerName, Position, ActivityType,
                   SleepHours, StressLevel, DistanceKm, SprintCount,
                   PredictedAction, FatigueScore, RiskLevel, Confidence, Status
            FROM TrainingSessions 
            WHERE Id = ?
        """, session_id)
        
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'timestamp': row[1],
                'player_name': row[2],
                'position': row[3],
                'activity_type': row[4],
                'sleep_hours': row[5],
                'stress_level': row[6],
                'distance_km': row[7],
                'sprint_count': row[8],
                'predicted_action': row[9],
                'fatigue_score': row[10],
                'risk_level': row[11],
                'confidence': row[12],
                'status': row[13]
            }
        return None
        
    except Exception as e:
        logger.error(f"❌ Greška pri dohvatanju detalja: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_database_info() -> Dict[str, Any]:
    """Vrati informacije o bazi"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM TrainingSessions")
        sessions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM TrainingSessions WHERE Status = 'queued'")
        queued_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Feedback")
        feedback_count = cursor.fetchone()[0]
        
        return {
            "database": DB_NAME,
            "server": DB_SERVER,
            "sessions": sessions_count,
            "queued": queued_count,
            "feedback": feedback_count
        }
        
    except Exception as e:
        logger.error(f"❌ Greška pri dohvatanju informacija: {e}")
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

# Auto-inicijalizacija kada se modul učitava
if __name__ == "__main__":
    print("🔄 Pokrećem inicijalizaciju baze...")
    success = init_database()
    if success:
        print("✅ Inicijalizacija završena uspješno!")
    else:
        print("❌ Inicijalizacija nije uspjela")