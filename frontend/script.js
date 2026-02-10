// script.js - FatigueAgent Frontend Logic - V2.0 sa LEARN metrikama

const API_BASE = 'http://localhost:8000';

// DOM Elements
const form = document.getElementById('sessionForm');
const submitBtn = document.getElementById('submitBtn');
const toggleOptionalBtn = document.getElementById('toggleOptional');
const optionalFields = document.getElementById('optionalFields');

// States
const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const resultsState = document.getElementById('resultsState');
const errorState = document.getElementById('errorState');

// Sliders
const sleepHours = document.getElementById('sleepHours');
const stressLevel = document.getElementById('stressLevel');
const soreness = document.getElementById('soreness');
const rpe = document.getElementById('rpe');

let currentSessionId = null;
let currentPredictedAction = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 FatigueAgent Frontend v2.0 initialized');
    console.log('📡 API Base URL:', API_BASE);
    initSliders();
    loadAgentStatus();
    setInterval(loadAgentStatus, 5000);
});

// ============================================================================
// SLIDER HANDLERS
// ============================================================================

function initSliders() {
    sleepHours.addEventListener('input', (e) => {
        document.getElementById('sleepValue').textContent = `${e.target.value} hours`;
    });

    stressLevel.addEventListener('input', (e) => {
        document.getElementById('stressValue').textContent = e.target.value;
    });

    soreness.addEventListener('input', (e) => {
        document.getElementById('sorenessValue').textContent = e.target.value;
    });

    rpe.addEventListener('input', (e) => {
        document.getElementById('rpeValue').textContent = e.target.value;
    });
}

function syncSliderLabels() {
    // Sleep (number input)
    document.getElementById('sleepValue').textContent =
        `${sleepHours.value} hours`;

    // Sliders
    document.getElementById('stressValue').textContent =
        stressLevel.value;

    document.getElementById('sorenessValue').textContent =
        soreness.value;

    document.getElementById('rpeValue').textContent =
        rpe.value;
}


// ============================================================================
// TOGGLE OPTIONAL FIELDS
// ============================================================================

toggleOptionalBtn.addEventListener('click', () => {
    const isHidden = optionalFields.style.display === 'none';
    optionalFields.style.display = isHidden ? 'block' : 'none';
    toggleOptionalBtn.querySelector('span').textContent = isHidden 
        ? '➖ Hide Advanced Fields' 
        : '➕ Advanced Fields (Optional)';
});

// ============================================================================
// FORM SUBMISSION
// ============================================================================

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    console.log('📝 Form submitted');
    
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline';
    
    const formData = getFormData();
    console.log('📦 Payload:', JSON.stringify(formData, null, 2));
    
    try {
        console.log('🌐 Sending request to:', `${API_BASE}/predict`);
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        console.log('📡 Response status:', response.status);
        const responseText = await response.text();
        console.log('📄 Response body:', responseText);
        
        if (!response.ok) {
            let errorMsg = `HTTP ${response.status}`;
            try {
                const errorData = JSON.parse(responseText);
                errorMsg = errorData.detail || errorMsg;
                console.error('❌ Error details:', errorData);
            } catch (e) {
                console.error('❌ Raw error:', responseText);
            }
            throw new Error(errorMsg);
        }
        
        const data = JSON.parse(responseText);
        console.log('✅ Success! Session ID:', data.session_id);
        
        showLoadingState(data.session_id);
        await pollForResults(data.session_id);
        
    } catch (error) {
        console.error('💥 Error:', error);
        showErrorState(error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// ============================================================================
// GET FORM DATA
// ============================================================================

function getFormData() {
    const data = {
        player_name: document.getElementById('playerName').value.trim(),
        position: document.getElementById('position').value,
        activity_type: document.querySelector('input[name="activityType"]:checked').value,
        sleep_hours: parseFloat(document.getElementById('sleepHours').value),
        stress_level: parseInt(document.getElementById('stressLevel').value),
        distance_km: parseFloat(document.getElementById('distance').value),
        sprint_count: parseInt(document.getElementById('sprintCount').value)
    };
    
    if (optionalFields.style.display !== 'none') {
        const sorenessVal = parseInt(document.getElementById('soreness').value);
        const rpeVal = parseInt(document.getElementById('rpe').value);
        const injuryVal = document.getElementById('injuryIllness').checked;
        
        if (!isNaN(sorenessVal)) data.soreness = sorenessVal;
        if (!isNaN(rpeVal)) data.rpe = rpeVal;
        data.injury_illness = injuryVal;
        
        console.log('📋 Optional fields included:', { soreness: sorenessVal, rpe: rpeVal, injury: injuryVal });
    }
    
    return data;
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

function showLoadingState(sessionId) {
    initialState.style.display = 'none';
    resultsState.style.display = 'none';
    errorState.style.display = 'none';
    loadingState.style.display = 'flex';
    
    document.getElementById('loadingSessionId').textContent = `#${sessionId}`;
}

function showResultsState(results) {
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    initialState.style.display = 'none';
    resultsState.style.display = 'block';
    
    updateResults(results);
}

function showErrorState(message) {
    loadingState.style.display = 'none';
    resultsState.style.display = 'none';
    initialState.style.display = 'none';
    errorState.style.display = 'flex';
    
    document.getElementById('errorMessage').textContent = message;
}
/*
function resetToInitial() {
    resultsState.style.display = 'none';
    errorState.style.display = 'none';
    loadingState.style.display = 'none';
    initialState.style.display = 'flex';
    
    form.reset();
    optionalFields.style.display = 'none';
    toggleOptionalBtn.querySelector('span').textContent = '➕ Advanced Fields (Optional)';
}*/

function resetToInitial() {
    resultsState.style.display = 'none';
    errorState.style.display = 'none';
    loadingState.style.display = 'none';
    initialState.style.display = 'flex';

    form.reset();

    optionalFields.style.display = 'none';
    toggleOptionalBtn.querySelector('span').textContent =
        '➕ Advanced Fields (Optional)';

    // 🔑 OVO FALI
    syncSliderLabels();
}


// ============================================================================
// POLL FOR RESULTS
// ============================================================================

async function pollForResults(sessionId, maxAttempts = 30) {
    console.log(`🔄 Polling for results (session #${sessionId})...`);
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        try {
            const response = await fetch(`${API_BASE}/predictions/${sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`📊 Poll attempt ${attempts + 1}: status = ${data.status}`);
            
            if (data.status === 'processed') {
                console.log('✅ Results ready!', data);
                showResultsState(data);
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
            
        } catch (error) {
            console.error('❌ Polling error:', error);
            throw error;
        }
    }
    
    throw new Error('Timeout: Results not ready after 30 seconds');
}

// ============================================================================
// UPDATE RESULTS UI
// ============================================================================

function updateResults(data) {
    const { session_id, fatigue_score, risk_level, predicted_action, confidence, processed_at } = data;
    
    console.log('🎨 Updating UI with results:', data);
    
    document.getElementById('fatigueScore').textContent = Math.round(fatigue_score);
    updateGauge(fatigue_score);
    
    const riskBadge = document.getElementById('riskBadge');
    const riskLevelText = document.getElementById('riskLevel');
    riskBadge.className = 'risk-badge ' + risk_level;
    riskLevelText.textContent = risk_level.toUpperCase();
    
    const riskIcons = {
        low: '✅',
        medium: '⚠️',
        high: '🔴',
        critical: '🚨'
    };
    riskBadge.querySelector('.risk-icon').textContent = riskIcons[risk_level] || '⚠️';
    
    const actionMap = {
        cleared: { icon: '✅', text: 'Cleared to Play' },
        monitor: { icon: '👀', text: 'Monitor Closely' },
        reduce_intensity: { icon: '⚡', text: 'Reduce Training Intensity' },
        rest_recommended: { icon: '🛌', text: 'Rest Recommended' },
        must_rest: { icon: '🚫', text: 'Must Rest - High Risk' }
    };
    
    const action = actionMap[predicted_action] || { icon: '💡', text: predicted_action };
    document.getElementById('actionIcon').textContent = action.icon;
    document.getElementById('actionText').textContent = action.text;
    
    const confidencePercent = Math.round(confidence * 100);
    document.getElementById('confidenceFill').style.width = `${confidencePercent}%`;
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    
    document.getElementById('sessionId').textContent = `#${session_id}`;
    document.getElementById('processedAt').textContent = formatTime(processed_at);

    currentSessionId = session_id;
    currentPredictedAction = predicted_action;
    
    document.getElementById('feedbackForm').reset();
    document.getElementById('feedbackSuccess').style.display = 'none';
    document.getElementById('correctActionSection').style.display = 'none';
}

// ============================================================================
// UPDATE GAUGE
// ============================================================================

function updateGauge(score) {
    const progress = document.getElementById('gaugeProgress');
    const needle = document.getElementById('gaugeNeedle');
    
    const maxOffset = 251.2;
    const offset = maxOffset - (score / 100) * maxOffset;
    progress.style.strokeDashoffset = offset;
    
    const angle = -90 + (score / 100) * 180;
    needle.style.transform = `rotate(${angle}deg)`;
}

// ============================================================================
// LOAD AGENT STATUS - SA LEARN METRIKAMA
// ============================================================================

async function loadAgentStatus() {
    try {
        const response = await fetch(`${API_BASE}/agent/status`);
        const data = await response.json();
        
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (data.is_running) {
            statusDot.classList.remove('inactive');
            statusText.textContent = 'Agent is Active';
        } else {
            statusDot.classList.add('inactive');
            statusText.textContent = 'Agents Offline';
        }
        
        // Osnovne metrike
        document.getElementById('processedCount').textContent = data.processed_count || 0;
        document.getElementById('avgTime').textContent = `${Math.round(data.avg_processing_time_ms || 0)}ms`;
        document.getElementById('queueSize').textContent = data.queue_size || 0;
        
        // LEARN metrike - loguj u konzolu
        console.log('📚 LEARN Metrics:', {
            avg_fatigue: data.avg_fatigue_score?.toFixed(1) || 0,
            avg_confidence: (data.avg_confidence * 100)?.toFixed(1) || 0,
            exploration_count: data.exploration_count || 0,
            review_needed: data.review_needed_count || 0,
            retrain_count: data.retrain_count || 0
        });
        
    } catch (error) {
        console.error('❌ Failed to load agent status:', error);
        document.querySelector('.status-text').textContent = 'Connection Error';
    }
}

// ============================================================================
// FEEDBACK HANDLING
// ============================================================================

document.addEventListener('change', (e) => {
    if (e.target.name === 'feedbackCorrect') {
        const correctActionSection = document.getElementById('correctActionSection');
        const isIncorrect = e.target.value === 'false';
        correctActionSection.style.display = isIncorrect ? 'block' : 'none';
        
        const correctActionInput = document.getElementById('correctAction');
        const estimatedFatigueInput = document.getElementById('estimatedFatigue');
        correctActionInput.required = isIncorrect;
        estimatedFatigueInput.required = isIncorrect;
    }
});

const feedbackForm = document.getElementById('feedbackForm');
feedbackForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!currentSessionId) {
        alert('No session to provide feedback for');
        return;
    }
    
    const submitBtn = document.getElementById('submitFeedbackBtn');
    const feedbackSuccess = document.getElementById('feedbackSuccess');
    
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline';
    
    try {
        const isCorrect = document.querySelector('input[name="feedbackCorrect"]:checked').value === 'true';
        
        let userLabel = currentPredictedAction;
        let estimatedScore = null;
        
        if (!isCorrect) {
            const correctAction = document.getElementById('correctAction').value;
            const estimatedFatigue = document.getElementById('estimatedFatigue').value;
            
            if (!correctAction || !estimatedFatigue) {
                alert('Please select both correct action and estimated fatigue level');
                return;
            }
            
            userLabel = correctAction;
            
            const fatigueMap = {
                'low': 20,
                'medium': 50,
                'high': 70,
                'critical': 90
            };
            estimatedScore = fatigueMap[estimatedFatigue];
        }
        
        const outcomes = [];
        if (document.getElementById('outcomeInjured').checked) outcomes.push('injured');
        if (document.getElementById('outcomeExhausted').checked) outcomes.push('exhausted');
        if (document.getElementById('outcomePerformedWell').checked) outcomes.push('performed_well');
        
        const comment = document.getElementById('feedbackComment').value;
        
        let fullComment = comment || '';
        if (outcomes.length > 0) {
            fullComment += (fullComment ? ' | ' : '') + 'Outcome: ' + outcomes.join(', ');
        }
        
        const feedbackData = {
            session_id: currentSessionId,
            correct: isCorrect,
            user_label: estimatedScore !== null ? estimatedScore.toString() : userLabel,
            comment: fullComment || null
        };
        
        console.log('📤 Sending feedback:', feedbackData);
        
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Feedback error response:', errorText);
            throw new Error('Failed to submit feedback');
        }
        
        const data = await response.json();
        console.log('✅ Feedback saved:', data);
        
        feedbackSuccess.style.display = 'block';
        feedbackSuccess.textContent = '✅ Feedback saved! Retrain agent will process it.';
        
        setTimeout(() => {
            feedbackForm.reset();
            feedbackSuccess.style.display = 'none';
            document.getElementById('correctActionSection').style.display = 'none';
        }, 3000);
        
    } catch (error) {
        console.error('❌ Feedback error:', error);
        alert('Failed to submit feedback: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// ============================================================================
// RESET BUTTONS
// ============================================================================

document.getElementById('resetBtn').addEventListener('click', resetToInitial);
document.getElementById('retryBtn').addEventListener('click', resetToInitial);

// ============================================================================
// UTILS
// ============================================================================

function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}