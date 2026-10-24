/**
 * RescuGuard AI — Emergency Response Prioritization Logic
 * INSA Cyber Talent Summer Camp 2026
 * -----------------------------------------------------------
 * Clean, lightweight, enterprise JavaScript for real-time incident
 * severity inference, dynamic probability distribution rendering,
 * explainable risk factor attribution, and automated dispatch ticketing.
 */

// Global State
let lastPredictionResult = null;
let lastInputPayload = null;

// Benchmark Scenarios Grounded on the US Accidents Kaggle Dataset
const BENCHMARK_PRESETS = {
    i95_fog: {
        sector: "I-95 Northern Corridor (Freezing & Fog)",
        lat: 41.8240,
        lng: -71.4128,
        temp_f: 28.0,
        humidity: 94,
        visibility: 0.8,
        bump: false,
        crossing: false,
        traffic_signal: false,
        weather: "Fog",
        solar: "Night"
    },
    i80_rain: {
        sector: "I-80 Midwest Highway (Wet & Low Visibility)",
        lat: 41.5868,
        lng: -87.6847,
        temp_f: 45.0,
        humidity: 92,
        visibility: 2.5,
        bump: false,
        crossing: true,
        traffic_signal: false,
        weather: "Heavy Rain",
        solar: "Day"
    },
    la_metro: {
        sector: "LA Urban Arterial (Daylight Rush Hour)",
        lat: 34.0522,
        lng: -118.2437,
        temp_f: 74.0,
        humidity: 42,
        visibility: 10.0,
        bump: false,
        crossing: true,
        traffic_signal: true,
        weather: "Clear",
        solar: "Day"
    },
    texas_heat: {
        sector: "Texas I-10 Freight Sector (Extreme Heat)",
        lat: 29.7604,
        lng: -95.3698,
        temp_f: 99.0,
        humidity: 65,
        visibility: 10.0,
        bump: false,
        crossing: false,
        traffic_signal: false,
        weather: "Clear",
        solar: "Day"
    },
    signal_crossing: {
        sector: "Suburban Protected Intersection (Daylight)",
        lat: 39.8651,
        lng: -84.0587,
        temp_f: 65.0,
        humidity: 50,
        visibility: 10.0,
        bump: false,
        crossing: true,
        traffic_signal: true,
        weather: "Clear",
        solar: "Day"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    initPresetButtons();
    initDynamicFieldHints();
    initFormSubmission();
    initTicketModal();
    checkSystemHealth();
});

/**
 * Scenario Presets Controller
 */
function initPresetButtons() {
    const buttons = document.querySelectorAll(".preset-btn");

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const presetKey = btn.getAttribute("data-preset");
            applyPreset(presetKey);
        });
    });
}

function applyPreset(key) {
    const data = BENCHMARK_PRESETS[key];
    if (!data) return;

    document.getElementById("start_lat").value = data.lat;
    document.getElementById("start_lng").value = data.lng;
    document.getElementById("temp_f").value = data.temp_f;
    document.getElementById("humidity").value = data.humidity;
    document.getElementById("visibility").value = data.visibility;

    document.getElementById("bump").checked = data.bump;
    document.getElementById("crossing").checked = data.crossing;
    document.getElementById("traffic_signal").checked = data.traffic_signal;

    document.getElementById("weather_condition").value = data.weather;
    document.getElementById("sunrise_sunset").value = data.solar;

    const sectorLabel = document.getElementById("activeSectorLabel");
    if (sectorLabel) {
        sectorLabel.textContent = `Scenario: ${data.sector}`;
    }

    updateDynamicFieldHints();
}

/**
 * Live temperature & humidity hint updates
 */
function initDynamicFieldHints() {
    const tempInput = document.getElementById("temp_f");
    const humidityInput = document.getElementById("humidity");
    const visibilityInput = document.getElementById("visibility");

    if (tempInput) tempInput.addEventListener("input", updateDynamicFieldHints);
    if (humidityInput) humidityInput.addEventListener("input", updateDynamicFieldHints);
    if (visibilityInput) visibilityInput.addEventListener("input", updateDynamicFieldHints);

    updateDynamicFieldHints();
}

function updateDynamicFieldHints() {
    const tempVal = parseFloat(document.getElementById("temp_f")?.value);
    const humVal = parseFloat(document.getElementById("humidity")?.value);
    const visVal = parseFloat(document.getElementById("visibility")?.value);

    // Temp Celsius
    const tempHint = document.getElementById("tempCelsiusHint");
    if (tempHint && !isNaN(tempVal)) {
        const c = ((tempVal - 32) * 5 / 9).toFixed(1);
        tempHint.textContent = `${c} °C`;
        if (tempVal <= 32) {
            tempHint.style.color = "#f43f5e";
            tempHint.textContent = `${c} °C (Freezing)`;
        } else if (tempVal >= 90) {
            tempHint.style.color = "#f97316";
            tempHint.textContent = `${c} °C (High Heat)`;
        } else {
            tempHint.style.color = "#38bdf8";
        }
    }

    // Humidity status
    const humHint = document.getElementById("humidityStatusHint");
    if (humHint && !isNaN(humVal)) {
        if (humVal >= 85) {
            humHint.textContent = "High (Slip Risk)";
            humHint.style.color = "#f97316";
        } else if (humVal <= 30) {
            humHint.textContent = "Low (Dry)";
            humHint.style.color = "#94a3b8";
        } else {
            humHint.textContent = "Nominal";
            humHint.style.color = "#10b981";
        }
    }

    // Visibility status
    const visHint = document.getElementById("visibilityStatusHint");
    if (visHint && !isNaN(visVal)) {
        if (visVal <= 2.0) {
            visHint.textContent = "Severe Fog / Mist";
            visHint.style.color = "#ef4444";
        } else if (visVal <= 5.0) {
            visHint.textContent = "Moderate Hazard";
            visHint.style.color = "#eab308";
        } else {
            visHint.textContent = "Clear Visibility";
            visHint.style.color = "#10b981";
        }
    }
}

/**
 * Form Submission & Inference API Call
 */
function initFormSubmission() {
    const form = document.getElementById("accidentForm");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await executeSeverityAssessment();
    });

    // Keyboard shortcut: Ctrl + Enter
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            executeSeverityAssessment();
        }
    });
}

async function executeSeverityAssessment() {
    const submitBtn = document.getElementById("submitBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");

    // Gather and validate input payload
    const payload = {
        "Start_Lat": parseFloat(document.getElementById("start_lat").value),
        "Start_Lng": parseFloat(document.getElementById("start_lng").value),
        "Temperature(F)": parseFloat(document.getElementById("temp_f").value),
        "Humidity(%)": parseFloat(document.getElementById("humidity").value),
        "Visibility(mi)": parseFloat(document.getElementById("visibility").value),
        "Bump": document.getElementById("bump").checked ? 1 : 0,
        "Crossing": document.getElementById("crossing").checked ? 1 : 0,
        "Traffic_Signal": document.getElementById("traffic_signal").checked ? 1 : 0,
        "Weather_Condition": document.getElementById("weather_condition").value,
        "Sunrise_Sunset": document.getElementById("sunrise_sunset").value
    };

    lastInputPayload = payload;

    // UI Loading state
    if (submitBtn) submitBtn.disabled = true;
    if (btnSpinner) btnSpinner.classList.remove("hidden");
    if (btnText) btnText.textContent = "Analyzing Incident Telemetry...";

    try {
        const response = await fetch("/api/v1/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned error status ${response.status}`);
        }

        const result = await response.json();
        lastPredictionResult = result;
        renderPredictionResult(result);
    } catch (err) {
        console.error("Severity inference error:", err);
        alert(`Assessment Error: ${err.message}`);
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (btnSpinner) btnSpinner.classList.add("hidden");
        if (btnText) btnText.textContent = "Evaluate Incident Severity & Dispatch";
    }
}

/**
 * Render Inference Results
 */
function renderPredictionResult(data) {
    const placeholder = document.getElementById("placeholder");
    const resultContent = document.getElementById("resultContent");

    if (placeholder) placeholder.classList.add("hidden");
    if (resultContent) resultContent.classList.remove("hidden");

    // Priority Banner
    const priorityBanner = document.getElementById("priorityBanner");
    const priorityBadge = document.getElementById("priorityBadge");
    const severityTitle = document.getElementById("severityTitle");
    const protocolText = document.getElementById("protocolText");
    const latencyPill = document.getElementById("inferenceLatencyPill");

    const sev = data.predicted_severity || 2;
    if (priorityBanner) {
        priorityBanner.className = `triage-banner priority-level-${sev}`;
    }

    if (priorityBadge) {
        priorityBadge.textContent = `${data.priority_level} PRIORITY`;
    }

    if (severityTitle) {
        severityTitle.textContent = data.severity_label;
    }

    if (protocolText) {
        protocolText.textContent = data.response_protocol;
    }

    if (latencyPill) {
        latencyPill.textContent = `Inference: ${data.inference_time_ms} ms`;
    }

    // Risk Factor Chips
    const riskContainer = document.getElementById("riskFactorsContainer");
    if (riskContainer) {
        riskContainer.innerHTML = "";
        const factors = data.contributing_risk_factors || [];
        factors.forEach(f => {
            const tag = document.createElement("div");
            tag.className = "risk-tag";
            tag.textContent = f;
            riskContainer.appendChild(tag);
        });
    }

    // Mandatory Action Directives
    const dispatchList = document.getElementById("dispatchList");
    if (dispatchList) {
        dispatchList.innerHTML = "";
        const actions = data.dispatch_recommendation || [];
        actions.forEach(action => {
            const li = document.createElement("li");
            li.textContent = action;
            dispatchList.appendChild(li);
        });
    }

    // Probability Distribution Bars
    const probContainer = document.getElementById("probabilityBars");
    if (probContainer) {
        probContainer.innerHTML = "";
        const probs = data.probabilities || {};

        Object.keys(probs).sort().forEach(key => {
            const pVal = probs[key];
            const pPct = (pVal * 100).toFixed(1);
            const sevNum = key.replace(/[^0-9]/g, "") || "2";

            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-labels">
                    <span>${key}</span>
                    <span class="mono">${pPct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill sev-${sevNum}" style="width: ${pPct}%"></div>
                </div>
            `;
            probContainer.appendChild(row);
        });
    }
}

/**
 * Official Incident Dispatch Ticket Modal
 */
function initTicketModal() {
    const modal = document.getElementById("ticketModal");
    const openBtn = document.getElementById("openTicketBtn");
    const closeBtn = document.getElementById("closeTicketBtn");
    const dismissBtn = document.getElementById("dismissTicketBtn");
    const printBtn = document.getElementById("printTicketBtn");

    if (openBtn) {
        openBtn.addEventListener("click", () => {
            if (!lastPredictionResult || !lastInputPayload) return;
            populateTicketModal();
            modal.classList.remove("hidden");
        });
    }

    const closeModal = () => modal.classList.add("hidden");
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (dismissBtn) dismissBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeModal();
    });

    if (printBtn) {
        printBtn.addEventListener("click", () => {
            window.print();
        });
    }
}

function populateTicketModal() {
    const refCode = "INC-" + Math.floor(100000 + Math.random() * 900000);
    const now = new Date();
    const timestampStr = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";

    document.getElementById("ticketRefCode").textContent = refCode;
    document.getElementById("ticketTimestamp").textContent = timestampStr;
    document.getElementById("ticketPriorityVal").textContent = 
        `${lastPredictionResult.priority_level} (Level ${lastPredictionResult.predicted_severity})`;

    document.getElementById("ticketCoordsVal").textContent = 
        `Lat: ${lastInputPayload.Start_Lat.toFixed(4)}, Lng: ${lastInputPayload.Start_Lng.toFixed(4)}`;

    document.getElementById("ticketProtocolVal").textContent = lastPredictionResult.response_protocol;

    const unitsList = document.getElementById("ticketUnitsVal");
    unitsList.innerHTML = "";
    (lastPredictionResult.dispatch_recommendation || []).forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        unitsList.appendChild(li);
    });

    const risksList = document.getElementById("ticketRisksVal");
    risksList.innerHTML = "";
    (lastPredictionResult.contributing_risk_factors || []).forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        risksList.appendChild(li);
    });
}

/**
 * System Health Probe & Telemetry Ping
 */
async function checkSystemHealth() {
    const statusText = document.getElementById("systemStatus");
    const latencyTag = document.getElementById("telemetryLatency");

    const t0 = performance.now();
    try {
        const res = await fetch("/api/v1/health");
        const t1 = performance.now();
        const ping = Math.round(t1 - t0);

        if (res.ok) {
            const data = await res.json();
            if (statusText) {
                statusText.textContent = data.model_loaded ? "Pipeline Online" : "Model Missing";
            }
            if (latencyTag) {
                latencyTag.textContent = `${ping} ms`;
            }
        }
    } catch {
        if (statusText) statusText.textContent = "Offline";
        if (latencyTag) latencyTag.textContent = "-- ms";
    }
}