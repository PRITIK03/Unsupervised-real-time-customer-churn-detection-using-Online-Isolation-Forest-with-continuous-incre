// ─────────────────────────────────────────
//  REGISTRY — Model Versions Table
// ─────────────────────────────────────────

async function loadRegistry() {
    try {
        showRegistryLoader();
        const res  = await fetch('/registry');
        const data = await res.json();

        if (data.models.length === 0) {
            showNoRegistryData();
            return;
        }

        renderRegistryTable(data.models);
        updateRegistrySummary(data.models);

    } catch (err) {
        console.error('Registry load error:', err);
    }
}

// ─────────────────────────────────────────
//  LOADER
// ─────────────────────────────────────────
function showRegistryLoader() {
    const el = document.getElementById('registry-table-body');
    if (el) el.innerHTML = `
        <tr>
            <td colspan="7" class="text-center">
                <div class="spinner"></div>
            </td>
        </tr>`;
}

function showNoRegistryData() {
    const el = document.getElementById('registry-table-body');
    if (el) el.innerHTML = `
        <tr>
            <td colspan="7" class="text-center"
                style="color:var(--text-muted);padding:40px">
                No models registered yet.
                Run the training pipeline first.
            </td>
        </tr>`;
}

// ─────────────────────────────────────────
//  SUMMARY CARDS
// ─────────────────────────────────────────
function updateRegistrySummary(models) {
    const total  = models.length;
    const active = models.find(m => m.is_active === 1);
    const latest = models[0];

    setVal('reg-total-versions', total);
    setVal('reg-active-version', active ? active.model_version : 'None');
    setVal('reg-latest-date',
        latest && latest.training_date
            ? latest.training_date.substring(0, 16)
            : 'N/A');
    setVal('reg-total-records',
        active
            ? parseInt(active.trained_on_records).toLocaleString()
            : 'N/A');

    // Color active badge
    const activeEl = document.getElementById('reg-active-version');
    if (activeEl) {
        activeEl.style.color     = 'var(--accent-green)';
        activeEl.style.fontWeight = '600';
        activeEl.style.fontSize  = '12px';
        activeEl.style.wordBreak = 'break-all';
    }
}

function setVal(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ─────────────────────────────────────────
//  REGISTRY TABLE
// ─────────────────────────────────────────
function renderRegistryTable(models) {
    const tbody = document.getElementById('registry-table-body');
    if (!tbody) return;

    tbody.innerHTML = models.map((m, index) => `
        <tr class="fade-in" style="animation-delay:${index * 0.05}s">

            <td>
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="
                        width:8px;height:8px;border-radius:50%;
                        background:${m.is_active ? '#10b981' : '#475569'};
                        display:inline-block;
                        ${m.is_active ? 'box-shadow:0 0 8px #10b981' : ''}
                    "></span>
                    <span style="
                        color:var(--accent-blue);
                        font-weight:600;
                        font-size:12px;
                        font-family:monospace
                    ">
                        ${m.model_version}
                    </span>
                </div>
            </td>

            <td>
                ${m.is_active
                    ? '<span class="badge badge-active">● Active</span>'
                    : '<span class="badge" style="background:rgba(71,85,105,0.2);color:#475569;border:1px solid #1e293b">Inactive</span>'
                }
            </td>

            <td style="color:var(--text-secondary)">
                ${m.training_date
                    ? m.training_date.substring(0, 16)
                    : 'N/A'}
            </td>

            <td style="color:var(--text-secondary)">
                ${m.trained_on_records
                    ? parseInt(m.trained_on_records).toLocaleString()
                    : 'N/A'}
            </td>

            <td>
                <span style="
                    font-size:11px;
                    color:var(--text-muted);
                    font-family:monospace;
                    word-break:break-all;
                    max-width:200px;
                    display:block
                ">
                    ${m.model_path
                        ? '...' + m.model_path.slice(-40)
                        : 'N/A'}
                </span>
            </td>

            <td style="color:var(--text-muted);font-size:12px">
                ${m.notes || '—'}
            </td>

            <td>
                ${m.is_active
                    ? `<button
                            onclick="reloadModel()"
                            style="
                                background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(6,182,212,0.15));
                                border:1px solid rgba(16,185,129,0.3);
                                color:#10b981;
                                padding:5px 12px;
                                border-radius:6px;
                                cursor:pointer;
                                font-size:12px;
                                font-family:Inter,sans-serif;
                                transition:all 0.2s
                            "
                            onmouseover="this.style.transform='translateY(-1px)'"
                            onmouseout="this.style.transform='none'"
                        >
                            🔄 Reload
                        </button>`
                    : `<button
                            onclick="activateModel('${m.model_version}', '${m.model_path}', ${m.trained_on_records})"
                            style="
                                background:linear-gradient(135deg,rgba(59,130,246,0.15),rgba(139,92,246,0.15));
                                border:1px solid rgba(59,130,246,0.3);
                                color:#3b82f6;
                                padding:5px 12px;
                                border-radius:6px;
                                cursor:pointer;
                                font-size:12px;
                                font-family:Inter,sans-serif;
                                transition:all 0.2s
                            "
                            onmouseover="this.style.transform='translateY(-1px)'"
                            onmouseout="this.style.transform='none'"
                        >
                            ⚡ Activate
                        </button>`
                }
            </td>
        </tr>
    `).join('');
}

// ─────────────────────────────────────────
//  RELOAD ACTIVE MODEL
// ─────────────────────────────────────────
async function reloadModel() {
    try {
        const res  = await fetch('/reload-model', { method: 'POST' });
        const data = await res.json();
        showToast('✅ ' + data.message, 'success');
        loadRegistry();
    } catch (err) {
        showToast('❌ Failed to reload model', 'error');
    }
}

// ─────────────────────────────────────────
//  ACTIVATE A MODEL VERSION
// ─────────────────────────────────────────
async function activateModel(version, path, records) {
    if (!confirm(`Activate model: ${version}?`)) return;

    try {
        // We call db directly via a custom endpoint
        const res = await fetch('/activate-model', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({
                model_version      : version,
                model_path         : path,
                trained_on_records : records
            })
        });

        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        showToast('✅ Model activated: ' + version, 'success');
        loadRegistry();

    } catch (err) {
        showToast('❌ Failed to activate model', 'error');
    }
}

// ─────────────────────────────────────────
//  TOAST NOTIFICATION
// ─────────────────────────────────────────
function showToast(message, type = 'success') {
    // Remove existing toast
    const existing = document.getElementById('toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id    = 'toast';

    const colors = {
        success: { bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.4)', color: '#10b981' },
        error  : { bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.4)',  color: '#ef4444' },
        warning: { bg: 'rgba(245,158,11,0.15)',  border: 'rgba(245,158,11,0.4)', color: '#f59e0b' }
    };

    const c = colors[type] || colors.success;

    toast.style.cssText = `
        position        : fixed;
        bottom          : 32px;
        right           : 32px;
        background      : ${c.bg};
        border          : 1px solid ${c.border};
        color           : ${c.color};
        padding         : 14px 24px;
        border-radius   : 10px;
        font-size       : 14px;
        font-weight     : 600;
        font-family     : Inter, sans-serif;
        z-index         : 9999;
        backdrop-filter : blur(10px);
        animation       : fadeInUp 0.3s ease;
        box-shadow      : 0 8px 32px rgba(0,0,0,0.3);
    `;

    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}