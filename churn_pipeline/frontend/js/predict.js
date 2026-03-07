// ─────────────────────────────────────────
//  PREDICT — Form + Risk Meter + Results
// ─────────────────────────────────────────

let riskMeterChart = null;

function initRiskMeter() {
    const ctx = document.getElementById('riskMeterCanvas');
    if (!ctx) return;
    if (riskMeterChart) riskMeterChart.destroy();
    riskMeterChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: ['rgba(59,130,246,0.3)', 'rgba(255,255,255,0.03)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            animation: { duration: 800, easing: 'easeInOutQuart' }
        }
    });
}

function updateRiskMeter(score, tier) {
    if (!riskMeterChart) return;
    const colors = {
        Critical: ['rgba(239,68,68,0.9)', 'rgba(239,68,68,0.15)'],
        High: ['rgba(249,115,22,0.9)', 'rgba(249,115,22,0.15)'],
        Medium: ['rgba(245,158,11,0.9)', 'rgba(245,158,11,0.15)'],
        Low: ['rgba(16,185,129,0.9)', 'rgba(16,185,129,0.15)']
    };
    const textColors = {
        Critical: '#ef4444', High: '#f97316', Medium: '#f59e0b', Low: '#10b981'
    };
    const col = colors[tier] || colors.Low;
    riskMeterChart.data.datasets[0].data = [score, 100 - score];
    riskMeterChart.data.datasets[0].backgroundColor = [col[0], col[1]];
    riskMeterChart.update();
    const scoreEl = document.getElementById('risk-score-number');
    const labelEl = document.getElementById('risk-score-label');
    if (scoreEl) { scoreEl.textContent = score.toFixed(1); scoreEl.style.color = textColors[tier] || '#f1f5f9'; }
    if (labelEl) labelEl.textContent = 'Risk Score';
}

// ─────────────────────────────────────────
//  SUBMIT PREDICTION
// ─────────────────────────────────────────
async function submitPrediction() {
    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing...';
    try {
        const payload = collectFormData();
        if (!payload) { btn.disabled = false; btn.textContent = '🔍 Analyze Churn Risk'; return; }

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Prediction failed'); }

        const data = await res.json();
        console.log('✅ Predict response:', data); // helpful for debugging
        displayResult(data);

    } catch (err) {
        alert('❌ Error: ' + err.message);
        console.error(err);
    } finally {
        btn.disabled = false;
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
        const val = el.value.trim();
        if (val === '') { alert(`Please fill in: ${field}`); return null; }
        if (['SeniorCitizen', 'tenure'].includes(field)) payload[field] = parseInt(val);
        else if (['MonthlyCharges', 'TotalCharges'].includes(field)) payload[field] = parseFloat(val);
        else payload[field] = val;
    }
    return payload;
}

// ─────────────────────────────────────────
//  DISPLAY RESULT
// ─────────────────────────────────────────
function displayResult(data) {
    document.getElementById('result-placeholder').style.display = 'none';
    const resultContent = document.getElementById('result-content');
    resultContent.style.display = 'block';
    resultContent.classList.add('fade-in');

    const score = data.churn_risk_score;
    const tier = data.risk_tier;

    updateRiskMeter(score, tier);

    const badgeEl = document.getElementById('result-tier-badge');
    if (badgeEl) { badgeEl.textContent = tier; badgeEl.className = `tier-badge tier-${tier}`; }

    setValue('result-score', score.toFixed(1) + '%');
    setValue('result-tier', tier);
    setValue('result-anomaly', data.is_anomaly ? '⚠️ Anomaly' : '✅ Normal');
    setValue('result-model', data.model_version || 'N/A');
    setValue('result-time', new Date(data.timestamp).toLocaleTimeString());

    const msgs = {
        Critical: '🔴 Very high churn probability. Immediate intervention recommended — offer retention deals.',
        High: '🟠 High churn risk detected. Consider proactive outreach with personalized offers.',
        Medium: '🟡 Moderate churn risk. Monitor this customer and consider engagement campaigns.',
        Low: '🟢 Low churn risk. Customer appears stable and satisfied.'
    };
    setValue('result-interpretation', msgs[tier] || '');

    const anomalyEl = document.getElementById('result-anomaly');
    if (anomalyEl) anomalyEl.style.color = data.is_anomaly ? '#ef4444' : '#10b981';

    // ── SHAP Panel ──
    const shapPanel = document.getElementById('shap-panel');
    const shapStatus = document.getElementById('result-shap-status');

    if (data.explanation && data.explanation.length > 0) {
        // Render inline panel
        renderShapPanel(data.explanation, data.explanation_summary || '');

        if (shapStatus) {
            shapStatus.textContent = '✅ Available';
            shapStatus.style.color = '#10b981';
        }
        // Cache for modal
        window._lastShapData = {
            explanation: data.explanation,
            summary: data.explanation_summary || ''
        };
    } else {
        // Hide panel, show unavailable
        if (shapPanel) shapPanel.classList.remove('visible');
        if (shapStatus) {
            shapStatus.textContent = '⚠️ Unavailable';
            shapStatus.style.color = '#f59e0b';
        }
        window._lastShapData = null;
        console.warn('⚠️ No SHAP explanation in response:', data);
    }
}

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ─────────────────────────────────────────
//  SHAP PANEL RENDERER
// ─────────────────────────────────────────
function renderShapPanel(explanation, summary) {
    const panel = document.getElementById('shap-panel');
    if (!panel) return;

    // Summary text
    const summaryEl = document.getElementById('shap-summary-text');
    if (summaryEl && summary) {
        const parts = summary.replace('Risk driven by: ', '').split(', ');
        summaryEl.innerHTML = '<strong>Key risk drivers:</strong> ' +
            parts.map(p => {
                const isUp = p.includes('↑');
                return `<span style="color:${isUp ? 'var(--red)' : 'var(--green)'}">${p}</span>`;
            }).join(', ');
    }

    // Top-5 bars
    const container = document.getElementById('shap-bars-container');
    if (container) {
        const top5 = explanation.slice(0, 5);
        const maxAbs = Math.max(...top5.map(e => Math.abs(e.shap_value)), 0.0001);
        container.innerHTML = top5.map(e => {
            const isRisk = e.direction === 'increases_risk';
            const cls = isRisk ? 'risk' : 'safe';
            const arrow = isRisk ? '▲' : '▼';
            const pct = Math.min((Math.abs(e.shap_value) / maxAbs) * 100, 100).toFixed(0);
            return `<div class="shap-bar-row">
                <div class="shap-bar-header">
                    <span class="shap-feature-name">${e.feature}</span>
                    <span class="shap-value-chip ${cls}">${arrow} ${Math.abs(e.shap_value).toFixed(4)}</span>
                </div>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill ${cls}" style="width:${pct}%"></div>
                </div>
            </div>`;
        }).join('');
    }

    panel.classList.add('visible');
}

// ─────────────────────────────────────────
//  SHAP MODAL
// ─────────────────────────────────────────
function openShapModal() {
    const overlay = document.getElementById('shap-modal-overlay');
    const content = document.getElementById('shap-modal-content');
    if (!overlay || !content) return;

    const shapData = window._lastShapData;
    if (!shapData || !shapData.explanation || shapData.explanation.length === 0) {
        content.innerHTML = '<div style="text-align:center;padding:32px;color:var(--text-muted)">No SHAP data available. Run a prediction first.</div>';
        overlay.classList.add('open');
        return;
    }

    const formPayload = collectFormData();
    if (!formPayload) {
        // Fall back to cached data without re-fetching
        renderShapModalContent(shapData.explanation, shapData.summary);
        overlay.classList.add('open');
        return;
    }

    content.innerHTML = '<div style="text-align:center;padding:32px"><div class="spinner" style="margin:0 auto"></div><div style="margin-top:12px;color:var(--text-muted);font-size:12px">Fetching full breakdown...</div></div>';
    overlay.classList.add('open');

    fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formPayload)
    })
        .then(r => {
            if (!r.ok) throw new Error('explain endpoint failed');
            return r.json();
        })
        .then(data => {
            renderShapModalContent(
                data.explanation && data.explanation.length > 0 ? data.explanation : shapData.explanation,
                data.explanation_summary || shapData.summary
            );
        })
        .catch(() => {
            // Graceful fallback to cached top-5
            renderShapModalContent(shapData.explanation, shapData.summary);
        });
}

function renderShapModalContent(explanation, summary) {
    const content = document.getElementById('shap-modal-content');
    if (!content) return;

    const maxAbs = Math.max(...explanation.map(e => Math.abs(e.shap_value)), 0.0001);
    let html = '';

    html += `<div class="shap-modal-summary"><strong>Summary:</strong> ${summary}</div>`;
    html += '<div class="shap-modal-bars">';

    explanation.forEach((e, i) => {
        const isRisk = e.direction === 'increases_risk';
        const cls = isRisk ? 'risk' : 'safe';
        const arrow = isRisk ? '▲ Increases Risk' : '▼ Decreases Risk';
        const pct = Math.min((Math.abs(e.shap_value) / maxAbs) * 100, 100).toFixed(0);
        html += `<div class="shap-modal-row">
            <div class="shap-modal-row-header">
                <span class="shap-modal-feature">${e.feature}</span>
                <span class="shap-modal-rank">#${e.rank || (i + 1)}</span>
            </div>
            <div class="shap-modal-track">
                <div class="shap-modal-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <div class="shap-modal-meta">
                <span>${arrow}</span>
                <span>SHAP: ${e.shap_value > 0 ? '+' : ''}${e.shap_value.toFixed(4)}</span>
            </div>
        </div>`;
    });

    html += '</div>';
    html += `<div class="shap-legend">
        <div class="shap-legend-item"><div class="shap-legend-dot" style="background:linear-gradient(135deg,#ef4444,#f97316)"></div> Increases Risk</div>
        <div class="shap-legend-item"><div class="shap-legend-dot" style="background:linear-gradient(135deg,#10b981,#06b6d4)"></div> Decreases Risk</div>
    </div>`;

    content.innerHTML = html;
}

function closeShapModal() {
    const overlay = document.getElementById('shap-modal-overlay');
    if (overlay) overlay.classList.remove('open');
}

function closeShapModalOutside(event) {
    if (event.target === event.currentTarget) closeShapModal();
}

// ─────────────────────────────────────────
//  RESET FORM
// ─────────────────────────────────────────
function resetForm() {
    document.getElementById('predict-form').reset();
    document.getElementById('result-placeholder').style.display = 'flex';
    document.getElementById('result-content').style.display = 'none';
    const sp = document.getElementById('shap-panel');
    if (sp) sp.classList.remove('visible');
    window._lastShapData = null;
    initRiskMeter();
}

// ─────────────────────────────────────────
//  FILL SAMPLE DATA
// ─────────────────────────────────────────
function fillSampleData(type) {
    const samples = {
        high_risk: {
            gender: 'Female', SeniorCitizen: 1, Partner: 'No', Dependents: 'No',
            tenure: 1, PhoneService: 'Yes', MultipleLines: 'No',
            InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'No',
            DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'Yes',
            StreamingMovies: 'Yes', Contract: 'Month-to-month', PaperlessBilling: 'Yes',
            PaymentMethod: 'Electronic check', MonthlyCharges: 95.75, TotalCharges: 95.75
        },
        low_risk: {
            gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'Yes',
            tenure: 60, PhoneService: 'Yes', MultipleLines: 'Yes',
            InternetService: 'DSL', OnlineSecurity: 'Yes', OnlineBackup: 'Yes',
            DeviceProtection: 'Yes', TechSupport: 'Yes', StreamingTV: 'No',
            StreamingMovies: 'No', Contract: 'Two year', PaperlessBilling: 'No',
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