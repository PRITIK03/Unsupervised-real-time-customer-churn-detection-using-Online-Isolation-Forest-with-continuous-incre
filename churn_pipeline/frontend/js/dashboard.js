// ─────────────────────────────────────────
//  DASHBOARD — Charts + Live Stats
// ─────────────────────────────────────────

let riskDonutChart  = null;
let trendLineChart  = null;
let ksDistanceChart = null;

// ─────────────────────────────────────────
//  LOAD DASHBOARD
// ─────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch('/dashboard-stats');
        if (!res.ok) throw new Error('API error: ' + res.status);
        const data = await res.json();

        console.log('Dashboard data:', data);

        updateStatCards(data);
        renderRiskDonut(data.risk_distribution);
        renderTrendLine(data.trend_data);
        renderKSChart(data.trend_data);
        updateModelBadge(data.model_version);

    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

// ─────────────────────────────────────────
//  STAT CARDS
// ─────────────────────────────────────────
function updateStatCards(data) {
    // Total Customers
    updateStatCard('stat-total-customers', 'Total Customers', data.total_customers, '✅');

    // Processed Records
    updateStatCard('stat-processed', 'Processed Records', data.processed_customers, '⚠️');

    // Anomaly Rate
    const latest = data.trend_data && data.trend_data.length > 0
        ? data.trend_data[0] : null;

    if (latest) {
        updateStatCard('stat-anomaly-rate', 'Anomaly Rate', `${parseFloat(latest.anomaly_rate).toFixed(1)}%`, '⚠️', `▲ ${latest.anomaly_rate_change}% vs last`);
        updateStatCard('stat-mean-score', 'Mean Risk Score', `${(parseFloat(latest.mean_anomaly_score) * 100).toFixed(1)}`, '🎯', `▲ ${latest.mean_score_change}% vs last`);
    }

    // Critical Risk
    const dist = data.risk_distribution;
    if (dist) {
        updateStatCard('stat-critical', 'Critical Risk', dist.Critical || 0, '🔴');
    }

    // Sidebar model version
    updateModelBadge(data.model_version);
}

function updateStatCard(elementId, label, value, icon, subtext = '') {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${icon} ${value.toLocaleString()}</div>
            <div class="stat-label">${label}</div>
            ${subtext ? `<div class="stat-subtext">${subtext}</div>` : ''}
        </div>
    `;
}

// ─────────────────────────────────────────
//  COUNT ANIMATION
// ─────────────────────────────────────────
function animateCount(elementId, target) {
    const el = document.getElementById(elementId);
    if (!el) return;

    let current = 0;
    const step  = Math.max(1, Math.ceil(target / 60));
    const timer = setInterval(() => {
        current += step;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        el.textContent = current.toLocaleString();
    }, 16);
}

// ─────────────────────────────────────────
//  RISK DONUT CHART
// ─────────────────────────────────────────
function renderRiskDonut(dist) {
    const ctx = document.getElementById('riskDonutChart');
    if (!ctx) return;

    if (riskDonutChart) riskDonutChart.destroy();

    const labels = ['Critical', 'High', 'Medium', 'Low'];
    const values = [
        dist.Critical || 0,
        dist.High     || 0,
        dist.Medium   || 0,
        dist.Low      || 0
    ];

    const total = values.reduce((a, b) => a + b, 0);

    riskDonutChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data           : values,
                backgroundColor: [
                    'rgba(239,68,68,0.85)',
                    'rgba(249,115,22,0.85)',
                    'rgba(245,158,11,0.85)',
                    'rgba(16,185,129,0.85)'
                ],
                hoverBackgroundColor: [
                    'rgba(239,68,68,1)',
                    'rgba(249,115,22,1)',
                    'rgba(245,158,11,1)',
                    'rgba(16,185,129,1)'
                ],
                borderWidth    : 2,
                hoverOffset    : 10
            }]
        },
        options: {
            responsive         : true,
            maintainAspectRatio: false,
            cutout             : '70%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels  : {
                        color          : '#94a3b8',
                        padding        : 16,
                        font           : { size: 12, family: 'Inter' },
                        usePointStyle  : true,
                        pointStyleWidth: 8
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2235',
                    titleColor     : '#f1f5f9',
                    bodyColor      : '#94a3b8',
                    borderColor    : '#2d3748',
                    borderWidth    : 1,
                    callbacks      : {
                        label: ctx => {
                            const percentage = ((ctx.raw / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${ctx.raw} customers (${percentage}%)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration     : 1500,
                easing       : 'easeInOutQuart'
            }
        }
    });
}

// ─────────────────────────────────────────
//  TREND LINE CHART
// ─────────────────────────────────────────
function renderTrendLine(trendData) {
    const ctx = document.getElementById('trendLineChart');
    if (!ctx) return;

    if (trendLineChart) trendLineChart.destroy();

    if (!trendData || trendData.length === 0) return;

    const reversed = [...trendData].reverse();
    const labels   = reversed.map(d => d.date.substring(0, 16));
    const scores   = reversed.map(d =>
        (parseFloat(d.mean_anomaly_score) * 100).toFixed(2));
    const rates    = reversed.map(d =>
        parseFloat(d.anomaly_rate).toFixed(2));

    trendLineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label              : 'Mean Risk Score',
                    data               : scores,
                    borderColor        : '#3b82f6',
                    backgroundColor    : 'rgba(59,130,246,0.1)',
                    tension            : 0.4,
                    fill               : true,
                    pointBackgroundColor: '#3b82f6',
                    pointRadius        : 5,
                    pointHoverRadius   : 8
                },
                {
                    label              : 'Anomaly Rate %',
                    data               : rates,
                    borderColor        : '#f97316',
                    backgroundColor    : 'rgba(249,115,22,0.1)',
                    tension            : 0.4,
                    fill               : true,
                    pointBackgroundColor: '#f97316',
                    pointRadius        : 5,
                    pointHoverRadius   : 8
                }
            ]
        },
        options: {
            responsive         : true,
            maintainAspectRatio: false,
            interaction        : { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color        : '#94a3b8',
                        font         : { size: 12, family: 'Inter' },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2235',
                    titleColor     : '#f1f5f9',
                    bodyColor      : '#94a3b8',
                    borderColor    : '#2d3748',
                    borderWidth    : 1
                }
            },
            scales: {
                x: {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.03)' }
                },
                y: {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.05)' }
                }
            },
            animation: { duration: 1000, easing: 'easeInOutQuart' }
        }
    });
}

// ─────────────────────────────────────────
//  KS DISTANCE CHART
// ─────────────────────────────────────────
function renderKSChart(trendData) {
    const ctx = document.getElementById('ksDistanceChart');
    if (!ctx) return;

    if (ksDistanceChart) ksDistanceChart.destroy();

    if (!trendData || trendData.length === 0) return;

    const reversed = [...trendData].reverse();
    const labels   = reversed.map(d => d.date.substring(0, 16));
    const ks       = reversed.map(d => parseFloat(d.ks_distance).toFixed(4));

    ksDistanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label          : 'KS Distance (Drift)',
                data           : ks,
                backgroundColor: ks.map(v =>
                    v > 0.30 ? 'rgba(239,68,68,0.7)'  :
                    v > 0.15 ? 'rgba(245,158,11,0.7)' :
                               'rgba(16,185,129,0.7)'
                ),
                borderColor    : ks.map(v =>
                    v > 0.30 ? '#ef4444' :
                    v > 0.15 ? '#f59e0b' :
                               '#10b981'
                ),
                borderWidth    : 1,
                borderRadius   : 6
            }]
        },
        options: {
            responsive         : true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font : { size: 12, family: 'Inter' }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2235',
                    titleColor     : '#f1f5f9',
                    bodyColor      : '#94a3b8',
                    borderColor    : '#2d3748',
                    borderWidth    : 1,
                    callbacks      : {
                        label: ctx => ` KS: ${ctx.raw} ${
                            ctx.raw > 0.30 ? '🔴 CRITICAL' :
                            ctx.raw > 0.15 ? '🟡 WARNING'  :
                                             '🟢 GOOD'}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.03)' }
                },
                y: {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.05)' },
                    min  : 0,
                    max  : 0.5
                }
            },
            animation: { duration: 1000 }
        }
    });
}

// ─────────────────────────────────────────
//  MODEL BADGE UPDATE
// ─────────────────────────────────────────
function updateModelBadge(version) {
    const el = document.getElementById('sidebar-model-version');
    if (el && version) el.textContent = version;
}

// ─────────────────────────────────────────
//  MICRO CHARTS (STOCK MARKET STYLE)
// ─────────────────────────────────────────
function renderMicroChart(elementId, data, color) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                data,
                borderColor: color,
                backgroundColor: `${color}33`, // Add transparency for gradient
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 6 // Highlight points on hover
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: '#1a2235',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: '#2d3748',
                    borderWidth: 1
                }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            hover: {
                mode: 'nearest',
                intersect: true
            }
        }
    });
}

// ─────────────────────────────────────────
//  SMOOTH UI TRANSITIONS
// ─────────────────────────────────────────
function applySmoothTransitions() {
    const elements = document.querySelectorAll('.stat-card, canvas');
    elements.forEach(el => {
        el.style.transition = 'all 0.3s ease-in-out';
        el.addEventListener('mouseover', () => {
            el.style.transform = 'scale(1.05)';
            el.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.2)';
        });
        el.addEventListener('mouseout', () => {
            el.style.transform = 'scale(1)';
            el.style.boxShadow = 'none';
        });
    });
}

// Call the function to apply transitions after the dashboard loads
document.addEventListener('DOMContentLoaded', () => {
    applySmoothTransitions();
});