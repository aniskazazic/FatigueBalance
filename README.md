# Fatigue Agent - Intelligent Player Fatigue Management System

## Overview

Fatigue Agent is an autonomous software agent that helps coaches and sports scientists make data-driven decisions about player training loads and injury prevention by analyzing training session data (sleep hours, stress level, distance covered, sprint count). Using machine learning, it provides actionable recommendations through an asynchronous, queue-based processing system.

The system monitors player fatigue levels and suggests appropriate training intensities, rest days, or recovery protocols based on continuous learning from feedback and historical data.

## How It Works

### Core Architecture

The system follows Clean Architecture with four distinct layers:

1. **Domain Layer** - Core business entities (TrainingSession, Prediction, ActionType)
2. **Application Layer** - Business logic and agent runners implementing the Sense→Think→Act cycle
3. **Infrastructure Layer** - Technical implementations (SQL Server database, ML classifier model)
4. **Web Layer** - Thin API layer (FastAPI endpoints)

### The Agent Cycle

Fatigue Agent operates autonomously in the background using a continuous loop:

1. **SENSE** - Monitors the queue for new training sessions from coaches/athletes
2. **THINK** - Processes data through ML models and business rules to predict fatigue and recommend actions
3. **ACT** - Stores predictions, evaluates risk levels, and updates session status

This cycle runs independently of API requests, ensuring real-time processing without blocking user interactions.

### Learning Capability

The system improves over time through user feedback. When coaches disagree with a fatigue prediction and provide the correct assessment, the agent uses this feedback to retrain and improve future predictions. The system automatically triggers retraining when sufficient feedback data is accumulated.

## Key Features

* **Autonomous Operation** - Runs continuously in background threads without manual intervention
* **Asynchronous Processing** - Training sessions are queued and processed independently from API calls
* **Machine Learning Integration** - Uses scikit-learn for regression-based fatigue prediction
* **Feedback Learning** - Continuously improves from coach corrections and field observations
* **Risk Assessment** - Automatically categorizes fatigue into Low/Medium/High risk levels
* **RESTful API** - Simple endpoints for integration
* **Web Dashboard** - Frontend interface for submitting sessions and monitoring predictions

## Installation & Setup

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize database (SQL Server):
   ```bash
   python scripts/init_db.py
   ```

5. (Optional) Train model from CSV data:
   ```bash
   python scripts/train_from_csv.py
   ```

6. Start the backend server:
   ```bash
   python main.py
   ```
   Server will run on `http://localhost:8000`


## Usage

### Submitting Training Data

POST `/predict`
```json
{
  "player_name": "John Smith",
  "position": "midfielder",
  "activity_type": "game",
  "sleep_hours": 7.5,
  "stress_level": 5,
  "distance_km": 9.2,
  "sprint_count": 24,
  "soreness": 4,
  "rpe": 6,
  "injury_illness": false
}
```

Response:
```json
{
  "session_id": 42,
  "status": "queued"
}
```

### Checking Predictions

GET `/predictions/{session_id}`

Returns:
```json
{
  "session_id": 42,
  "status": "processed",
  "predicted_action": "reduce_load",
  "fatigue_score": 72.5,
  "risk_level": "high",
  "confidence": 0.89
}
```

### Providing Feedback

POST `/feedback`
```json
{
  "session_id": 42,
  "correct": false,
  "user_label": "complete_rest",
  "comment": "Player showed severe fatigue symptoms"
}
```

## Testing

Run the comprehensive test suite to verify agent learning:

```bash
python scripts/test_agent.py
```

This test script:
- Submits sample training sessions
- Gets baseline predictions
- Provides feedback with corrections
- Triggers model retraining
- Verifies that agent predictions improved based on feedback

## Project Structure

```
.
├── backend/
│   ├── scripts/
│   │   ├── train_from_csv.py      # Train model from CSV
│   │   ├── init_db.py             # Initialize SQL Server database
│   │   └── test_agent.py          # Test agent learning capability
│   ├── application/
│   │   ├── agent_manager.py       # Core agent orchestration
│   │   ├── runners/               # Scoring and retraining runners
│   │   └── services/              # Business logic services
│   ├── domain/
│   │   └── entities.py            # Domain models
│   ├── infrastructure/
│   │   ├── database.py            # Database interactions
│   │   ├── models.py              # SQLAlchemy models
│   │   └── ml/
│   │       └── classifier.py      # ML model wrapper
│   ├── web/
│   │   ├── main.py                # FastAPI application
│   │   └── dtos.py                # Request/response models
│   ├── bootstrap.py               # Dependency injection setup
│   ├── main.py                    # Entry point
│   ├── fatigue_model.joblib       # Trained ML model
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── index.html                 # Main HTML
│   ├── script.js                  # React application
│   └── style.css                  # Styling
.
```

## Why It's an Agent (Not Just an API)

Unlike traditional ML APIs that process requests synchronously, Fatigue Agent:

* **Operates Continuously** - Background workers process requests without user initiation
* **Maintains State** - Uses a database queue to track sessions and feedback
* **Implements Cognitive Cycle** - Follows complete Sense→Think→Act→Learn loop
* **Learns Autonomously** - Improves predictions without explicit retraining commands
* **Runs Independently** - Keeps processing even when users aren't actively using the system
* **Handles Background Tasks** - Retraining happens automatically without blocking API responses

## Practical Benefits for Coaches

1. **Immediate Response** - Training sessions are queued and processed within seconds
2. **Continuous Monitoring** - Agent works 24/7, even outside training hours
3. **Improved Accuracy** - Learning from feedback creates increasingly accurate fatigue predictions
4. **Non-blocking Interface** - Coaches can submit data and check results later without waiting
5. **Actionable Insights** - Clear recommendations: "full training," "reduce load," "complete rest," or "no action needed"
6. **Risk Awareness** - Automatic detection of high-risk fatigue states to prevent injuries

## Technical Details

### Machine Learning Model
- **Algorithm** - Scikit-learn RandomForestRegressor
- **Target Variable** - Fatigue Score (0-100)
- **Features** - Sleep hours, stress level, distance, sprint count, soreness, RPE
- **Training Data** - Workout routine dataset with 1000+ labeled examples
- **Metrics** - MAE, RMSE, R² calculated after each training session

### Database
- **System** - SQL Server (ODBC connection)
- **Main Tables** - TrainingSessions, Feedback, SystemSettings
- **Features** - Automatic timestamps, foreign key relationships, status tracking

### Agent Configuration
- **Scoring Runner** - Processes predictions every 10 seconds
- **Retrain Runner** - Triggers retraining when 50+ feedback items accumulated
- **Risk Thresholds** - Low (<40), Medium (40-60), High (>60)
- **Exploration Rate** - 5% chance of random exploration vs exploitation

## Configuration

Edit `SystemSettings` table in SQL Server to adjust:
- `GoldThreshold` - Feedback items required before retraining (default: 50)
- `LowRiskThreshold` - Fatigue score for low risk (default: 40)
- `MediumRiskThreshold` - Fatigue score for medium risk (default: 60)
- `ExplorationRate` - Probability of exploration (default: 0.05)

## Future Improvements

- Real-time athlete wearable integration
- Multi-model ensemble for fatigue prediction
- Advanced visualizations and trend analysis
- Predictive injury risk modeling
- Mobile app for on-field feedback submission

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Developed as an intelligent agent system for sports science applications.

---

**Note**: This system is designed to assist coaches and sports scientists in decision-making. Final decisions regarding player training loads should always consider professional medical advice and coaching expertise.
