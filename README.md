# FatigueBalance - Intelligent Player Fatigue Assessment Agent

## Overview

FatigueBalance is an intelligent agent for assessing football player fatigue levels after training sessions and matches, implemented as a complete AI system with perception, continuous learning mechanisms, and real-time monitoring interface. The system uses a Neural Network (MLPRegressor) for fatigue regression (fatigue score 0-100), combined with rule-based classification for risk level assessment and feedback-driven incremental learning for continuous performance improvement.

The system autonomously monitors player fatigue levels and provides actionable recommendations for training intensity adjustments, rest days, and recovery protocols based on continuous learning from coach feedback and historical training data.

## How It Works

### Core Architecture

The system follows Clean Architecture with four distinct layers:

1. **Domain Layer** - Core business entities (TrainingSession, Prediction, ActionType)
2. **Application Layer** - Business logic and agent runners implementing the Sense→Think→Act→Learn cycle
3. **Infrastructure Layer** - Technical implementations (SQL Server database, Neural Network ML classifier)
4. **Web Layer** - Thin API layer (FastAPI endpoints)

### The Agent Cycle

FatigueBalance operates autonomously in the background using a continuous cognitive loop:

1. **SENSE** - Monitors the queue for new training sessions from coaches/athletes
2. **THINK** - Processes data through Neural Network models and rule-based business rules to predict fatigue and recommend actions
3. **ACT** - Stores predictions, evaluates risk levels using classification rules, and updates session status
4. **LEARN** - Incorporates coach feedback to incrementally retrain the Neural Network

This cycle runs independently of API requests, ensuring real-time processing without blocking user interactions.

### Learning Capability

The system improves over time through user feedback (incremental learning). When coaches disagree with a fatigue prediction and provide the correct assessment, the agent uses this feedback to retrain the Neural Network model and improve future predictions. The system automatically triggers retraining when sufficient feedback data is accumulated, implementing a complete feedback-driven learning loop.

## Key Features

* **Autonomous Operation** - Runs continuously in background threads without manual intervention
* **Asynchronous Processing** - Training sessions are queued and processed independently from API calls
* **Neural Network Integration** - Uses scikit-learn MLPRegressor (Multi-layer Perceptron) for intelligent fatigue prediction
* **Rule-Based Classification** - Intelligent classification of fatigue into Low/Medium/High risk levels based on learned thresholds
* **Feedback-Driven Learning** - Continuously improves from coach corrections and field observations using incremental learning
* **Continuous Learning** - Model retrains automatically when sufficient feedback accumulates
* **RESTful API** - Simple endpoints for integration with coaching systems
* **Web Dashboard** - Frontend interface for submitting sessions and monitoring predictions in real-time

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

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Open `index.html` in your web browser or serve with a local server:
   ```bash
   # If you have Python installed
   python -m http.server 8080
   ```
   Dashboard will be available on `http://localhost:8080`

## Testing

Run the comprehensive test suite to verify agent learning:

```bash
python scripts/test_agent.py
```

This test script:
- Submits sample training sessions
- Gets baseline predictions from the Neural Network
- Provides feedback with corrections
- Triggers incremental model retraining
- Verifies that agent predictions improved based on feedback
- Validates the complete Sense→Think→Act→Learn cycle

## Why It's an Agent (Not Just an ML Model)

Unlike traditional ML APIs that process requests synchronously, FatigueBalance:

* **Operates Continuously** - Background workers (runners) process requests without user initiation
* **Maintains State** - Uses a database queue to track sessions, feedback, and learning progress
* **Implements Cognitive Cycle** - Follows complete Sense→Think→Act→Learn loop with perception and action
* **Learns Autonomously** - Improves predictions automatically through feedback without explicit retraining commands
* **Runs Independently** - Keeps processing even when users aren't actively using the system
* **Handles Background Tasks** - Incremental learning and model retraining happens automatically in background threads
* **Intelligent Decision Making** - Combines Neural Network predictions with rule-based classification for robust decisions

## Technical Details

### Neural Network Model
- **Algorithm** - Scikit-learn MLPRegressor (Multi-layer Perceptron)
- **Architecture** - Hidden layers with ReLU activation for non-linear pattern recognition
- **Target Variable** - Fatigue Score (0-100 continuous regression)
- **Input Features** - Sleep hours, stress level, distance, sprint count, soreness, RPE, injury/illness status
- **Training Data** - Workout routine dataset with 1000+ labeled examples
- **Learning** - Supports incremental learning (warm_start=True) for continuous model improvement
- **Metrics** - MAE, RMSE, R² calculated after each training session

### Rule-Based Risk Classification
- **Low Risk** - Fatigue score < 40 (Player ready for full training)
- **Medium Risk** - Fatigue score 40-60 (Player should reduce load)
- **High Risk** - Fatigue score > 60 (Complete rest or minimal activity recommended)

### Database
- **System** - SQL Server 
- **Main Tables** - TrainingSessions, Feedback, SystemSettings
- **Features** - Automatic timestamps, foreign key relationships, status tracking for queue management

### Agent Components
- **Scoring Runner** - Processes predictions every 10 seconds through the Neural Network
- **Retrain Runner** - Triggers incremental retraining when 10+ feedback items accumulated
- **Queue Service** - Manages asynchronous session processing
- **Learning Service** - Handles feedback incorporation and model updates

## Configuration

Edit `SystemSettings` table in SQL Server to adjust:
- `GoldThreshold` - Feedback items required before retraining (default: 50)
- `LowRiskThreshold` - Fatigue score threshold for low risk (default: 40)
- `MediumRiskThreshold` - Fatigue score threshold for medium risk (default: 60)
- `ExplorationRate` - Probability of exploration vs exploitation (default: 0.05)

## Incremental Learning Process

1. Coach submits training session data
2. Neural Network predicts fatigue score
3. Rule-based classifier assigns risk level
4. Coach reviews prediction and provides feedback if incorrect
5. Feedback is stored in database
6. When threshold reached, retrain runner triggers
7. Neural Network is retrained with warm_start to incorporate new knowledge
8. Updated model becomes active for future predictions
9. Agent improves accuracy over time

## Future Improvements

- Real-time athlete wearable integration
- Advanced Neural Network architectures (LSTM for temporal patterns)
- Ensemble methods combining multiple neural networks
- Advanced visualizations and trend analysis
- Predictive injury risk modeling
- Mobile app for on-field feedback submission
- Transfer learning from other sports domains

## Performance

The Neural Network learns from feedback in real-time, continuously improving prediction accuracy. The system has been tested to validate:
- Accurate fatigue prediction from training metrics
- Proper classification of risk levels
- Effective learning from coach feedback
- Autonomous operation without manual intervention

---

**Note**: FatigueBalance is designed to assist coaches and sports scientists in decision-making. Final decisions regarding player training loads should always consider professional medical advice, coaching expertise, and player well-being.

