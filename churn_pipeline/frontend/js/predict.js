// ─────────────────────────────────────────
//  PREDICT — Form + Risk Meter + Results
// ─────────────────────────────────────────

let riskMeterChart = null;

// ─────────────────────────────────────────
//  RISK METER GAUGE
// ─────────────────────────────────────────
function initRiskMeter() {
    const ctx = document.getElementById('riskMeterCanvas');
    if (!ctx) return;

    if (riskMeterChart) riskMeterChart.destroy();

    riskMeterChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data           : [0, 100],
                backgroundColor: ['rgba(59,130,246,0.3)', 'rgba(255,255,255,0.03)'],
                borderWidth    : 0,
                circumference  : 180,
                rotation       : 270
            }]
        },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            cutout              : '75%',
            plugins             : { legend: { display: false }, tooltip: { enabled: false } },
            animation           : { duration: 800, easing: 'easeInOutQuart' }
        }
    });
}

function updateRiskMeter(score, tier) {
    if (!riskMeterChart) return;

    const colors = {
        Critical : ['rgba(239,68,68,0.9)',  'rgba(239,68,68,0.15)'],
        High     : ['rgba(249,115,22,0.9)', 'rgba(249,115,22,0.15)'],
        Medium   : ['rgba(245,158,11,0.9)', 'rgba(245,158,11,0.15)'],
        Low      : ['rgba(16,185,129,0.9)', 'rgba(16,185,129,0.15)']
    };

    const textColors = {
        Critical: '#ef4444',
        High    : '#f97316',
        Medium  : '#f59e0b',
        Low     : '#10b981'
    };

    const col = colors[tier] || colors.Low;

    riskMeterChart.data.datasets[0].data            = [score, 100 - score];
    riskMeterChart.data.datasets[0].backgroundColor = [col[0], col[1]];
    riskMeterChart.update();

    // Update score text
    const scoreEl = document.getElementById('risk-score-number');
    const labelEl = document.getElementById('risk-score-label');
    if (scoreEl) {
        scoreEl.textContent = score.toFixed(1);
        scoreEl.style.color = textColors[tier] || '#f1f5f9';
    }
    if (labelEl) labelEl.textContent = 'Risk Score';
}

// ─────────────────────────────────────────
//  SUBMIT PREDICTION
// ─────────────────────────────────────────
async function submitPrediction() {
    const btn = document.getElementById('predict-btn');
    btn.disabled     = true;
    btn.textContent  = '⏳ Analyzing...';

    try {
        const payload = collectFormData();
        if (!payload) {
            btn.disabled    = false;
            btn.textContent = '🔍 Analyze Churn Risk';
            return;
        }

        const res  = await fetch('/predict', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await res.json();
        displayResult(data);

    } catch (err) {
        alert('❌ Error: ' + err.message);
        console.error(err);
    } finally {
        btn.disabled    = false;
        btn.textContent = '🔍 Analyze Churn Risk';
    }
}

// ─────────────────────────────────────────
//  COLLECT FORM DATA
// ─────────────────────────────────────────
function collectFormData() {
    const fields = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ];

    const payload = {};

    for (const field of fields) {
        const el = document.getElementById('f_' + field);
        if (!el) { console.warn('Missing field:', field); continue; }

        let val = el.value.trim();
        if (val === '') {
            alert(`Please fill in: ${field}`);
            return null;
        }

        // Convert numeric fields
        if (['SeniorCitizen', 'tenure'].includes(field)) {
            payload[field] = parseInt(val);
        } else if (['MonthlyCharges', 'TotalCharges'].includes(field)) {
            payload[field] = parseFloat(val);
        } else {
            payload[field] = val;
        }
    }

    return payload;
}

// ─────────────────────────────────────────
//  DISPLAY RESULT
// ─────────────────────────────────────────
function displayResult(data) {
    // Hide placeholder, show result
    document.getElementById('result-placeholder').style.display = 'none';
    const resultContent = document.getElementById('result-content');
    resultContent.style.display = 'block';
    resultContent.classList.add('fade-in');

    const score = data.churn_risk_score;
    const tier  = data.risk_tier;

    // Update meter
    updateRiskMeter(score, tier);

    // Update tier badge
    const badgeEl = document.getElementById('result-tier-badge');
    if (badgeEl) {
        badgeEl.textContent = tier;
        badgeEl.className   = `risk-tier-badge tier-${tier}`;
    }

    // Update details
    setValue('result-score',    score.toFixed(1) + '%');
    setValue('result-tier',     tier);
    setValue('result-anomaly',  data.is_anomaly ? '⚠️ Anomaly' : '✅ Normal');
    setValue('result-model',    data.model_version || 'N/A');
    setValue('result-time',     new Date(data.timestamp).toLocaleTimeString());

    // Risk interpretation
    const interpEl = document.getElementById('result-interpretation');
    if (interpEl) {
        const messages = {
            Critical: '🔴 Very high churn probability. Immediate intervention recommended — offer retention deals, escalate to account manager.',
            High    : '🟠 High churn risk detected. Consider proactive outreach with personalized offers.',
            Medium  : '🟡 Moderate churn risk. Monitor this customer and consider engagement campaigns.',
            Low     : '🟢 Low churn risk. Customer appears stable and satisfied.'
        };
        interpEl.textContent = messages[tier] || '';
    }

    // Color the anomaly result
    const anomalyEl = document.getElementById('result-anomaly');
    if (anomalyEl) {
        anomalyEl.style.color = data.is_anomaly ? '#ef4444' : '#10b981';
    }
}

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ─────────────────────────────────────────
//  RESET FORM
// ─────────────────────────────────────────
function resetForm() {
    document.getElementById('predict-form').reset();
    document.getElementById('result-placeholder').style.display = 'flex';
    document.getElementById('result-content').style.display     = 'none';
    initRiskMeter();
}

// ─────────────────────────────────────────
//  FILL SAMPLE DATA
// ─────────────────────────────────────────
function fillSampleData(type) {
    const samples = {
        high_risk: {
            gender: 'Female', SeniorCitizen: 1, Partner: 'No',
            Dependents: 'No', tenure: 1, PhoneService: 'Yes',
            MultipleLines: 'No', InternetService: 'Fiber optic',
            OnlineSecurity: 'No', OnlineBackup: 'No',
            DeviceProtection: 'No', TechSupport: 'No',
            StreamingTV: 'Yes', StreamingMovies: 'Yes',
            Contract: 'Month-to-month', PaperlessBilling: 'Yes',
            PaymentMethod: 'Electronic check',
            MonthlyCharges: 95.75, TotalCharges: 95.75
        },
        low_risk: {
            gender: 'Male', SeniorCitizen: 0, Partner: 'Yes',
            Dependents: 'Yes', tenure: 60, PhoneService: 'Yes',
            MultipleLines: 'Yes', InternetService: 'DSL',
            OnlineSecurity: 'Yes', OnlineBackup: 'Yes',
            DeviceProtection: 'Yes', TechSupport: 'Yes',
            StreamingTV: 'No', StreamingMovies: 'No',
            Contract: 'Two year', PaperlessBilling: 'No',
            PaymentMethod: 'Bank transfer (automatic)',
            MonthlyCharges: 65.50, TotalCharges: 3930.0
        }
    };

    const sample = samples[type];
    if (!sample) return;

    for (const [key, value] of Object.entries(sample)) {
        const el = document.getElementById('f_' + key);
        if (el) el.value = value;
    }
}