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
    animateCount('stat-total-customers', data.total_customers);

    // Processed
    animateCount('stat-processed', data.processed_customers);

    // Anomaly Rate + Mean Score
    const latest = data.trend_data && data.trend_data.length > 0
        ? data.trend_data[0] : null;

    if (latest) {
        const anomalyEl = document.getElementById('stat-anomaly-rate');
        const scoreEl   = document.getElementById('stat-mean-score');
        if (anomalyEl) anomalyEl.textContent =
            parseFloat(latest.anomaly_rate).toFixed(1) + '%';
        if (scoreEl) scoreEl.textContent =
            (parseFloat(latest.mean_anomaly_score) * 100).toFixed(1);
    }

    // Status badge
    const statusEl = document.getElementById('stat-status');
    if (statusEl) {
        const status         = data.latest_status || 'GOOD';
        statusEl.textContent = status;
        statusEl.className   = 'badge badge-' + status.toLowerCase();
    }

    // Risk counts
    const dist = data.risk_distribution;
    if (dist) {
        animateCount('stat-critical', dist.Critical || 0);
        animateCount('stat-high',     dist.High     || 0);
    }

    // Sidebar model version
    updateModelBadge(data.model_version);
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
                borderColor    : [
                    '#ef4444', '#f97316', '#f59e0b', '#10b981'
                ],
                borderWidth    : 2,
                hoverOffset    : 8
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
                        label: ctx => ` ${ctx.label}: ${ctx.raw} customers`
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration     : 1000,
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