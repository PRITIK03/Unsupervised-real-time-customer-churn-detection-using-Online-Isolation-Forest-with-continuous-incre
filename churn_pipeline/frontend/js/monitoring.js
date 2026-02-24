// ─────────────────────────────────────────
//  MONITORING — Metrics History + Charts
// ─────────────────────────────────────────

let anomalyTrendChart = null;
let statusHistoryChart = null;
let scoreDistChart = null;

async function loadMonitoring() {
    try {
        showMonitoringLoader();
        const res  = await fetch('/monitoring?days=30');
        const data = await res.json();

        if (data.metrics.length === 0) {
            showNoDataMessage();
            return;
        }

        renderMetricsTable(data.metrics);
        renderAnomalyTrend(data.metrics);
        renderStatusHistory(data.metrics);
        renderScoreDist(data.metrics);
        updateMonitoringSummary(data.metrics);

    } catch (err) {
        console.error('Monitoring load error:', err);
    }
}

// ─────────────────────────────────────────
//  LOADER
// ─────────────────────────────────────────
function showMonitoringLoader() {
    const el = document.getElementById('monitoring-table-body');
    if (el) el.innerHTML = '<tr><td colspan="8" class="text-center"><div class="spinner"></div></td></tr>';
}

function showNoDataMessage() {
    const el = document.getElementById('monitoring-table-body');
    if (el) el.innerHTML = '<tr><td colspan="8" class="text-center" style="color:var(--text-muted);padding:40px">No metrics logged yet. Run the pipeline first.</td></tr>';
}

// ─────────────────────────────────────────
//  SUMMARY CARDS
// ─────────────────────────────────────────
function updateMonitoringSummary(metrics) {
    const latest = metrics[0];
    if (!latest) return;

    setValue('mon-total-runs',    metrics.length);
    setValue('mon-latest-status', latest.status);
    setValue('mon-latest-score',  (latest.mean_anomaly_score * 100).toFixed(1) + '%');
    setValue('mon-latest-ks',     parseFloat(latest.ks_distance).toFixed(4));
    setValue('mon-anomaly-rate',  parseFloat(latest.anomaly_rate_pct).toFixed(1) + '%');

    // Good runs count
    const goodRuns = metrics.filter(m => m.status === 'GOOD').length;
    setValue('mon-good-runs', goodRuns + ' / ' + metrics.length);

    // Status badge color
    const statusEl = document.getElementById('mon-latest-status');
    if (statusEl) {
        statusEl.className = 'badge badge-' + latest.status.toLowerCase();
    }

    // Trend direction
    if (metrics.length >= 2) {
        const curr = parseFloat(metrics[0].mean_anomaly_score);
        const prev = parseFloat(metrics[1].mean_anomaly_score);
        const diff = ((curr - prev) * 100).toFixed(2);
        const trendEl = document.getElementById('mon-trend-direction');
        if (trendEl) {
            trendEl.textContent = diff > 0
                ? `↑ +${diff}% vs last run`
                : `↓ ${diff}% vs last run`;
            trendEl.style.color = diff > 0 ? '#ef4444' : '#10b981';
        }
    }
}

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ─────────────────────────────────────────
//  METRICS TABLE
// ─────────────────────────────────────────
function renderMetricsTable(metrics) {
    const tbody = document.getElementById('monitoring-table-body');
    if (!tbody) return;

    tbody.innerHTML = metrics.map(m => `
        <tr class="fade-in">
            <td style="color:var(--accent-blue);font-weight:600;font-size:12px">
                ${m.model_version}
            </td>
            <td>${(parseFloat(m.mean_anomaly_score) * 100).toFixed(2)}%</td>
            <td>${parseFloat(m.std_anomaly_score).toFixed(4)}</td>
            <td>${parseFloat(m.anomaly_rate_pct).toFixed(2)}%</td>
            <td>
                <span style="color:${getKSColor(m.ks_distance)}">
                    ${parseFloat(m.ks_distance).toFixed(4)}
                </span>
            </td>
            <td>${parseFloat(m.trend_score).toFixed(2)}</td>
            <td>${m.total_records_scored.toLocaleString()}</td>
            <td>
                <span class="badge badge-${m.status.toLowerCase()}">
                    ${m.status}
                </span>
            </td>
            <td style="color:var(--text-muted);font-size:12px">
                ${m.logged_at ? m.logged_at.substring(0, 16) : 'N/A'}
            </td>
        </tr>
    `).join('');
}

function getKSColor(ks) {
    const v = parseFloat(ks);
    if (v > 0.30) return '#ef4444';
    if (v > 0.15) return '#f59e0b';
    return '#10b981';
}

// ─────────────────────────────────────────
//  ANOMALY TREND CHART
// ─────────────────────────────────────────
function renderAnomalyTrend(metrics) {
    const ctx = document.getElementById('anomalyTrendChart');
    if (!ctx) return;

    if (anomalyTrendChart) anomalyTrendChart.destroy();

    const reversed = [...metrics].reverse();
    const labels   = reversed.map(m => m.logged_at
        ? m.logged_at.substring(0, 16) : '');
    const scores   = reversed.map(m =>
        (parseFloat(m.mean_anomaly_score) * 100).toFixed(2));
    const rates    = reversed.map(m =>
        parseFloat(m.anomaly_rate_pct).toFixed(2));

    anomalyTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label           : 'Mean Risk Score (%)',
                    data            : scores,
                    borderColor     : '#3b82f6',
                    backgroundColor : 'rgba(59,130,246,0.08)',
                    tension         : 0.4,
                    fill            : true,
                    pointRadius     : 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#3b82f6'
                },
                {
                    label           : 'Anomaly Rate (%)',
                    data            : rates,
                    borderColor     : '#f97316',
                    backgroundColor : 'rgba(249,115,22,0.08)',
                    tension         : 0.4,
                    fill            : true,
                    pointRadius     : 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#f97316'
                }
            ]
        },
        options: chartOptions('Model Performance Over Time')
    });
}

// ─────────────────────────────────────────
//  STATUS HISTORY CHART
// ─────────────────────────────────────────
function renderStatusHistory(metrics) {
    const ctx = document.getElementById('statusHistoryChart');
    if (!ctx) return;

    if (statusHistoryChart) statusHistoryChart.destroy();

    const reversed   = [...metrics].reverse();
    const labels     = reversed.map(m =>
        m.logged_at ? m.logged_at.substring(0, 10) : '');
    const statusNums = reversed.map(m =>
        m.status === 'GOOD' ? 3 : m.status === 'WARNING' ? 2 : 1);
    const bgColors   = reversed.map(m =>
        m.status === 'GOOD'     ? 'rgba(16,185,129,0.7)' :
        m.status === 'WARNING'  ? 'rgba(245,158,11,0.7)' :
                                  'rgba(239,68,68,0.7)');

    statusHistoryChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label          : 'Model Status',
                data           : statusNums,
                backgroundColor: bgColors,
                borderRadius   : 6,
                borderWidth    : 0
            }]
        },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                tooltip: {
                    backgroundColor : '#1a2235',
                    titleColor      : '#f1f5f9',
                    bodyColor       : '#94a3b8',
                    borderColor     : '#2d3748',
                    borderWidth     : 1,
                    callbacks: {
                        label: ctx => {
                            const map = { 3: 'GOOD', 2: 'WARNING', 1: 'DEGRADED' };
                            return ' Status: ' + (map[ctx.raw] || ctx.raw);
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.03)' }
                },
                y: {
                    ticks: {
                        color   : '#475569',
                        font    : { size: 11 },
                        callback: v => ({ 1: 'DEGRADED', 2: 'WARNING', 3: 'GOOD' }[v] || '')
                    },
                    grid : { color: 'rgba(255,255,255,0.05)' },
                    min  : 0, max: 4
                }
            },
            animation: { duration: 1000 }
        }
    });
}

// ─────────────────────────────────────────
//  SCORE DISTRIBUTION CHART
// ─────────────────────────────────────────
function renderScoreDist(metrics) {
    const ctx = document.getElementById('scoreDistChart');
    if (!ctx) return;

    if (scoreDistChart) scoreDistChart.destroy();

    const reversed = [...metrics].reverse();
    const labels   = reversed.map(m =>
        m.logged_at ? m.logged_at.substring(0, 10) : '');
    const ks       = reversed.map(m =>
        parseFloat(m.ks_distance).toFixed(4));
    const trend    = reversed.map(m =>
        parseFloat(m.trend_score).toFixed(2));

    scoreDistChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label           : 'KS Distance (Drift)',
                    data            : ks,
                    borderColor     : '#8b5cf6',
                    backgroundColor : 'rgba(139,92,246,0.08)',
                    tension         : 0.4,
                    fill            : true,
                    pointRadius     : 5,
                    pointBackgroundColor: '#8b5cf6',
                    yAxisID         : 'y'
                },
                {
                    label           : 'Trend Score',
                    data            : trend,
                    borderColor     : '#06b6d4',
                    backgroundColor : 'rgba(6,182,212,0.08)',
                    tension         : 0.4,
                    fill            : true,
                    pointRadius     : 5,
                    pointBackgroundColor: '#06b6d4',
                    yAxisID         : 'y1'
                }
            ]
        },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            interaction         : { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font : { size: 12, family: 'Inter' },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor : '#1a2235',
                    titleColor      : '#f1f5f9',
                    bodyColor       : '#94a3b8',
                    borderColor     : '#2d3748',
                    borderWidth     : 1
                }
            },
            scales: {
                x  : {
                    ticks: { color: '#475569', font: { size: 11 } },
                    grid : { color: 'rgba(255,255,255,0.03)' }
                },
                y  : {
                    type    : 'linear',
                    position: 'left',
                    ticks   : { color: '#8b5cf6', font: { size: 11 } },
                    grid    : { color: 'rgba(255,255,255,0.05)' }
                },
                y1 : {
                    type    : 'linear',
                    position: 'right',
                    ticks   : { color: '#06b6d4', font: { size: 11 } },
                    grid    : { drawOnChartArea: false }
                }
            },
            animation: { duration: 1000 }
        }
    });
}

// ─────────────────────────────────────────
//  SHARED CHART OPTIONS
// ─────────────────────────────────────────
function chartOptions(title) {
    return {
        responsive          : true,
        maintainAspectRatio : false,
        interaction         : { mode: 'index', intersect: false },
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font : { size: 12, family: 'Inter' },
                    usePointStyle: true
                }
            },
            tooltip: {
                backgroundColor : '#1a2235',
                titleColor      : '#f1f5f9',
                bodyColor       : '#94a3b8',
                borderColor     : '#2d3748',
                borderWidth     : 1
            }
        },
        scales: {
            x: {
                ticks: { color: '#475569', font: { size: 10 }, maxRotation: 45 },
                grid : { color: 'rgba(255,255,255,0.03)' }
            },
            y: {
                ticks: { color: '#475569', font: { size: 11 } },
                grid : { color: 'rgba(255,255,255,0.05)' }
            }
        },
        animation: { duration: 1000, easing: 'easeInOutQuart' }
    };
}