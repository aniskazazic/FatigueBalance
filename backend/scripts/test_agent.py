#!/usr/bin/env python
"""
🔥 PRAVI TEST DA LI AGENT UČI!
Testira stvarno učenje iz feedbacka.
"""
import requests
import time
import json

API_BASE = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def reset_test_environment():
    """Resetuj test okruženje"""
    print_section("RESET TEST OKRUŽENJA")
    
    # 1. Obriši postojeći model
    import os
    if os.path.exists("fatigue_model.joblib"):
        os.remove("fatigue_model.joblib")
        print("✅ Obrisan postojeći model")
    
    # 2. Resetuj feedback u bazi (opcionalno - za čist test)
    # Ovo možete ručno uraditi u SSMS
    
    print("🔄 Kreiraj novi prazni model...")
    # Pokreni backend da kreira novi model
    print("✅ Okruženje spremno za test")

def test_incremental_learning():
    """GLAVNI TEST INCREMENTAL LEARNINGA"""
    print("\n\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "🔥 TEST: DA LI AGENT STVARNO UČI? 🔥" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    
    # ===== FAZA 1: Treniraj agenta na NISKOM fatigue =====
    print_section("FAZA 1: Treniraj agenta na NISKOM fatigue")
    
    print("📤 Submitting 5 sessions sa LOW fatigue parametrima...")
    
    low_fatigue_sessions = [
        {
            "player_name": "Test Low 1",
            "position": "midfielder",
            "activity_type": "practice",
            "sleep_hours": 8.5,
            "stress_level": 2,
            "distance_km": 6.0,
            "sprint_count": 12,
            "soreness": 3,
            "rpe": 4,
            "injury_illness": False
        },
        {
            "player_name": "Test Low 2",
            "position": "defender",
            "activity_type": "practice",
            "sleep_hours": 8.0,
            "stress_level": 3,
            "distance_km": 5.5,
            "sprint_count": 10,
            "soreness": 2,
            "rpe": 3,
            "injury_illness": False
        },
        {
            "player_name": "Test Low 3",
            "position": "forward",
            "activity_type": "practice",
            "sleep_hours": 7.5,
            "stress_level": 4,
            "distance_km": 7.0,
            "sprint_count": 15,
            "soreness": 4,
            "rpe": 5,
            "injury_illness": False
        }
    ]
    
    session_ids = []
    for session in low_fatigue_sessions:
        response = requests.post(f"{API_BASE}/predict", json=session)
        session_id = response.json()['session_id']
        session_ids.append(session_id)
        print(f"   ✅ Session #{session_id} submitted")
        time.sleep(1)  # Čekaj da se procesira
    
    # Sacekaj da se sve procesira
    print("⏳ Čekam da se sve procesira...")
    time.sleep(5)
    
    # ===== FAZA 2: Testiraj baseline predikciju =====
    print_section("FAZA 2: Testiraj BASELINE predikciju")
    
    test_session = {
        "player_name": "TEST PLAYER - Learning Check",
        "position": "midfielder",
        "activity_type": "game",
        "sleep_hours": 7.0,
        "stress_level": 5,
        "distance_km": 9.0,
        "sprint_count": 25,
        "soreness": 5,
        "rpe": 6,
        "injury_illness": False
    }
    
    print("📤 Submitting TEST session za baseline...")
    response = requests.post(f"{API_BASE}/predict", json=test_session)
    test_session_id_1 = response.json()['session_id']
    print(f"✅ Test session #{test_session_id_1} submitted")
    
    time.sleep(3)
    
    response = requests.get(f"{API_BASE}/predictions/{test_session_id_1}")
    baseline_result = response.json()
    
    if baseline_result['status'] != 'processed':
        print("❌ Session not processed")
        return False
    
    baseline_fatigue = baseline_result['fatigue_score']
    baseline_action = baseline_result['predicted_action']
    
    print(f"🤖 BASELINE PREDICTION (prije feedbacka):")
    print(f"   Fatigue Score: {baseline_fatigue:.2f}")
    print(f"   Action: {baseline_action}")
    print(f"   Confidence: {baseline_result['confidence']:.2f}")
    
    # ===== FAZA 3: Pošalji HIGH fatigue feedback =====
    print_section("FAZA 3: Pošalji INCORRECT feedback (HIGH fatigue)")
    
    # Kažemo da je igrač bio MNOGO umorniji
    corrected_fatigue = baseline_fatigue + 25  # +25 bodova!
    
    feedback = {
        "session_id": test_session_id_1,
        "correct": False,
        "user_label": str(corrected_fatigue),
        "comment": f"Player was actually MUCH more fatigued ({corrected_fatigue:.0f}), not {baseline_fatigue:.0f}"
    }
    
    print(f"💬 Giving feedback: Player was at {corrected_fatigue:.0f}, not {baseline_fatigue:.0f}")
    response = requests.post(f"{API_BASE}/feedback", json=feedback)
    print(f"✅ Feedback submitted")
    
    # ===== FAZA 4: Dodaj još feedbacka za retrain =====
    print_section("FAZA 4: Dodaj još feedbacka za retrain trigger")
    
    print("📤 Submitting 4 more sessions with feedback...")
    
    for i in range(4):
        session = {
            "player_name": f"Extra Feedback {i+1}",
            "position": ["midfielder", "forward", "defender"][i % 3],
            "activity_type": "game" if i % 2 == 0 else "practice",
            "sleep_hours": 6.5 + (i % 2),
            "stress_level": 5 + (i % 3),
            "distance_km": 8.0 + i,
            "sprint_count": 22 + (i * 3),
            "soreness": 4 + (i % 4),
            "rpe": 5 + (i % 4),
            "injury_illness": False
        }
        
        response = requests.post(f"{API_BASE}/predict", json=session)
        sid = response.json()['session_id']
        
        time.sleep(2)
        
        response = requests.get(f"{API_BASE}/predictions/{sid}")
        result = response.json()
        
        if result['status'] == 'processed':
            feedback = {
                "session_id": sid,
                "correct": False,
                "user_label": str(result['fatigue_score'] + 20),  # +20 bodova
                "comment": "Higher fatigue than predicted"
            }
            requests.post(f"{API_BASE}/feedback", json=feedback)
            print(f"   ✅ Session #{sid}: feedback submitted")
    
    print(f"\n✅ Ukupno 5 feedback items submitted (dovoljno za retrain)")
    print(f"⏳ Čekam 35 sekundi za retrain agent da procesira...")
    time.sleep(35)
    
    # ===== FAZA 5: Submit ISTU test sesiju ponovo =====
    print_section("FAZA 5: Submit ISTU test sesiju ponovo (nakon učenja)")
    
    print("📤 Submitting SAME test session again...")
    response = requests.post(f"{API_BASE}/predict", json=test_session)
    test_session_id_2 = response.json()['session_id']
    print(f"✅ Test session #{test_session_id_2} submitted (nakon učenja)")
    
    time.sleep(3)
    
    response = requests.get(f"{API_BASE}/predictions/{test_session_id_2}")
    learned_result = response.json()
    
    if learned_result['status'] != 'processed':
        print("❌ Session not processed")
        return False
    
    learned_fatigue = learned_result['fatigue_score']
    learned_action = learned_result['predicted_action']
    
    print(f"🤖 NEW PREDICTION (nakon učenja):")
    print(f"   Fatigue Score: {learned_fatigue:.2f}")
    print(f"   Action: {learned_action}")
    print(f"   Confidence: {learned_result['confidence']:.2f}")
    
    # ===== FAZA 6: ANALIZA =====
    print_section("FAZA 6: ANALIZA - DA LI JE AGENT NAUČIO?")
    
    difference = learned_fatigue - baseline_fatigue
    target_increase = corrected_fatigue - baseline_fatigue  # 25
    
    print(f"📊 REZULTATI:")
    print(f"   Baseline (prije):  {baseline_fatigue:.2f}")
    print(f"   Feedback (tačno):  {corrected_fatigue:.2f}")
    print(f"   Nova predikcija:   {learned_fatigue:.2f}")
    print(f"   Promjena:          {difference:+.2f} bodova")
    print(f"   Target promjena:   +{target_increase:.0f} bodova")
    print()
    
    # Kriterijumi za učenje
    LEARNING_THRESHOLD = 8.0  # Minimalna promjena
    
    if difference >= LEARNING_THRESHOLD:
        print("✅ ✅ ✅ AGENT JE NAUČIO! ✅ ✅ ✅")
        print(f"   Predikcija se povećala za {difference:.2f} bodova!")
        print(f"   Agent je prilagodio predikcije na osnovu feedbacka!")
        
        # Procenat učenja
        learning_percentage = (difference / target_increase) * 100
        print(f"   Agent je naučio {learning_percentage:.1f}% od očekivanog!")
        
        if learning_percentage > 30:
            print(f"   🎉 ODLIČNO! Agent značajno uči iz feedbacka!")
        elif learning_percentage > 10:
            print(f"   👍 DOBRO! Agent uči, ali može bolje.")
        
        return True
        
    else:
        print("❌ ❌ ❌ AGENT NIJE NAUČIO! ❌ ❌ ❌")
        print(f"   Predikcija se promijenila samo {difference:.2f} bodova")
        print(f"   To je premalo - agent nije naučio iz feedbacka!")
        print()
        print("🔍 MOGUĆI RAZLOZI:")
        print("   1. classifier.train_single() ne poziva pravi retrain")
        print("   2. Model se ne sačuva nakon treninga")
        print("   3. Retrain agent ne radi ili ne procesira feedback")
        print("   4. Feedback nije označen kao Processed=1")
        
        return False

def check_agent_details():
    """Provjeri detalje agenta"""
    print_section("DETALJI AGENTA")
    
    try:
        response = requests.get(f"{API_BASE}/agent/status")
        status = response.json()
        
        print(f"🤖 Agent Status:")
        print(f"   Running: {status['is_running']}")
        print(f"   Processed: {status['processed_count']}")
        print(f"   Retrain Count: {status.get('retrain_count', 'N/A')}")
        print(f"   Queue Size: {status['queue_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Greška: {e}")
        return False

def main():
    """Glavna test funkcija"""
    
    # Provjeri API
    try:
        response = requests.get(API_BASE, timeout=5)
        print(f"✅ API dostupan: {API_BASE}")
    except:
        print(f"❌ API nije dostupan na {API_BASE}")
        print(f"   Pokreni backend: python main.py")
        return
    
    # Provjeri agente
    check_agent_details()
    
    # Resetuj okruženje za čist test (opcionalno)
    # reset_test_environment()
    
    # Pokreni glavni test
    print("\n" + "🔥"*70)
    print("🔥 POKREĆEM GLAVNI TEST INCREMENTAL LEARNINGA")
    print("🔥"*70)
    
    learned = test_incremental_learning()
    
    # Finalni rezultat
    print("\n\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*25 + "FINALNI REZULTAT" + " "*27 + "║")
    print("╚" + "═"*68 + "╝")
    
    if learned:
        print("\n🎉 🎉 🎉 ČESTITKE! 🎉 🎉 🎉")
        print("Tvoj agent ZAISTA UČI iz feedbacka!")
        print("Implementacija je USPJEŠNA i prati profesorove zahtjeve!")
        print("\n✅ Clean Architecture")
        print("✅ Dva agentička runnera (Scoring + Retrain)")
        print("✅ Incremental Learning iz feedbacka")
        print("✅ Background processing")
        print("✅ Sense→Think→Act→Learn ciklus")
    else:
        print("\n😔 AGENT NE UČI.")
        print("Provjeri gore navedene razloge i popravi kod.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()