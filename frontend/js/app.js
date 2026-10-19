/**
 * RescuGuard AI — Emergency Response Prioritization Logic
 * INSA Cyber Talent Summer Camp 2026
 * -----------------------------------------------------------
 * Clean, emoji-free enterprise logic for geospatial mapping,
 * heatmap analysis, incident presets, and dispatch protocols.
 */

// Global State
let map = null;
let incidentMarker = null;
let heatLayer = null;
let corridorsLayerGroup = null;
let hotspotsLayerGroup = null;
let lastPredictionResult = null;

// Historical Accident Density Hotspots [lat, lng, intensity]
const ACCIDENT_HEAT_POINTS = [
    // Addis Ababa Ring Road & High-Density Urban Hotspots
    [8.9897, 38.7523, 1.0],  // Gotera Interchange
    [8.9880, 38.7510, 0.8],
    [8.9910, 38.7540, 0.85],
    [8.9610, 38.7615, 0.9],   // Saris Abo Junction
    [8.9580, 38.7600, 0.75],
    [8.9220, 38.7610, 0.95],  // Kality Roundabout & Truck Exit
    [8.9200, 38.7590, 0.85],
    [8.9950, 38.7880, 0.85],  // Bole Ring Road
    [8.9980, 38.7910, 0.75],
    [9.0205, 38.8020, 0.9],   // Megenagna Hub
    [9.0230, 38.8050, 0.8],
    [9.0190, 38.8250, 0.75],  // Gurad Shola
    [9.0110, 38.7240, 0.85],  // Tor Hailoch
    [9.0090, 38.7210, 0.75],
    [8.9850, 38.7050, 0.8],   // Ayer Tena Roundabout
    [9.0340, 38.7380, 0.9],   // Merkato Corridor
    [9.0300, 38.7510, 0.75],  // Piassa Churchill Ave
    [9.0108, 38.7616, 0.85],  // Meskel Square

    // Addis–Djibouti Freight Logistics Artery
    [8.8850, 38.7950, 0.85],  // Akaki Toll Gate
    [8.7950, 38.9050, 0.8],   // Dukem Industrial Link
    [8.7500, 38.9800, 0.9],   // Bishoftu Bypass
    [8.5925, 39.1245, 1.0],   // Modjo Dry Port Freight Interchange
    [8.5950, 39.1260, 0.9],
    [8.5400, 39.2700, 0.9],   // Adama Expressway Exit
    [8.6650, 39.4350, 0.75],  // Welenchiti Curve
    [8.8950, 39.9150, 0.85],  // Metehara Transit
    [9.0050, 40.1650, 0.95],  // Awash Arba Gorge
    [10.1650, 40.6450, 0.8],  // Gewane Freight Sector
    [11.4150, 40.7580, 0.9],  // Mille Junction
    [11.7150, 41.8350, 0.85], // Galafi Border Crossing

    // Entoto Pass Hazards
    [9.0750, 38.7620, 0.9],   // Entoto Pass Base
    [9.0820, 38.7620, 0.85],  // Hairpin Curve
    [9.1050, 38.7580, 0.8],   // Sululta Transition

    // US Benchmark
    [39.8651, -84.0587, 0.95],
    [39.8600, -84.0700, 0.85],
    [39.8700, -84.0400, 0.9]
];

// High-Risk Route Corridors
const ACCIDENT_CORRIDORS = [
    {
        name: "Addis Ababa Ring Road Belt",
        color: "#f97316",
        risk: "High Collision Density",
        coords: [
            [8.9850, 38.7050],
            [9.0110, 38.7240],
            [9.0340, 38.7380],
            [9.0300, 38.7510],
            [9.0108, 38.7616],
            [8.9897, 38.7523],
            [8.9610, 38.7615],
            [8.9220, 38.7610],
            [8.9950, 38.7880],
            [9.0205, 38.8020],
            [9.0190, 38.8250]
        ]
    },
    {
        name: "Addis–Djibouti Freight Corridor",
        color: "#ef4444",
        risk: "Critical Freight Route",
        coords: [
            [8.9220, 38.7610],
            [8.8850, 38.7950],
            [8.7950, 38.9050],
            [8.7500, 38.9800],
            [8.5925, 39.1245],
            [8.5400, 39.2700],
            [8.6650, 39.4350],
            [8.8950, 39.9150],
            [9.0050, 40.1650]
        ]
    },
    {
        name: "Entoto Mountain Pass",
        color: "#eab308",
        risk: "Steep Grade & Fog Zone",
        coords: [
            [9.0300, 38.7510],
            [9.0550, 38.7600],
            [9.0750, 38.7620],
            [9.0820, 38.7620],
            [9.1100, 38.7550]
        ]
    }
];

// Presets
const PRESETS = {
    addis_gotera: {
        sector: "Addis Ababa: Gotera Interchange",
        lat: 8.9897,
        lng: 38.7523,
        temp_f: 66.0,
        humidity: 68,
        visibility: 9.0,
        bump: false,
        crossing: true,
        traffic_signal: true,
        weather: "Clear",
        solar: "Day"
    },
    modjo_corridor: {
        sector: "Addis–Djibouti Corridor: Modjo Freight Route",
        lat: 8.5925,
        lng: 39.1245,
        temp_f: 58.0,
        humidity: 88,
        visibility: 3.5,
        bump: false,
        crossing: false,
        traffic_signal: false,
        weather: "Rain",
        solar: "Night"
    },
    bole_rain: {
        sector: "Addis Ababa: Bole Ring Road",
        lat: 8.9950,
        lng: 38.7880,
        temp_f: 61.0,
        humidity: 94,
        visibility: 2.2,
        bump: false,
        crossing: true,
        traffic_signal: true,
        weather: "Heavy Rain",
        solar: "Day"
    },
    entoto_fog: {
        sector: "Entoto Mountain Pass",
        lat: 9.0750,
        lng: 38.7620,
        temp_f: 46.5,
        humidity: 98,
        visibility: 0.6,
        bump: true,
        crossing: false,
        traffic_signal: false,
        weather: "Fog",
        solar: "Night"
    },
    us_benchmark: {
        sector: "US Highway Benchmark: Ohio I-70",
        lat: 39.8651,
        lng: -84.0587,
        temp_f: 36.9,
        humidity: 91,
        visibility: 1.5,
        bump: false,
        crossing: true,
        traffic_signal: true,
        weather: "Fog",
        solar: "Day"
    }
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initGeospatialMap();
    initMapControls();
    initPresetListeners();
    initModalHandlers();
    checkSystemHealth();
});

// Map Setup
function initGeospatialMap() {
    const defaultLat = parseFloat(document.getElementById('start_lat').value) || 8.9897;
    const defaultLng = parseFloat(document.getElementById('start_lng').value) || 38.7523;

    map = L.map('incidentMap', {
        zoomControl: true,
        attributionControl: false
    }).setView([defaultLat, defaultLng], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19
    }).addTo(map);

    // Heatmap Layer
    if (typeof L.heatLayer === 'function') {
        heatLayer = L.heatLayer(ACCIDENT_HEAT_POINTS, {
            radius: 22,
            blur: 14,
            maxZoom: 15,
            minOpacity: 0.35,
            gradient: {
                0.2: '#38bdf8',
                0.5: '#eab308',
                0.8: '#f97316',
                1.0: '#ef4444'
            }
        }).addTo(map);
    }

    corridorsLayerGroup = L.layerGroup().addTo(map);
    hotspotsLayerGroup = L.layerGroup().addTo(map);

    renderCorridors();
    renderHotspotBadges();

    // Incident Marker
    incidentMarker = L.marker([defaultLat, defaultLng], {
        draggable: true
    }).addTo(map);

    incidentMarker.on('dragend', () => {
        const pos = incidentMarker.getLatLng();
        updateCoordinateInputs(pos.lat, pos.lng);
    });

    map.on('click', (e) => {
        incidentMarker.setLatLng(e.latlng);
        updateCoordinateInputs(e.latlng.lat, e.latlng.lng);
    });

    document.getElementById('start_lat').addEventListener('change', syncInputsToMap);
    document.getElementById('start_lng').addEventListener('change', syncInputsToMap);
}

function renderCorridors() {
    ACCIDENT_CORRIDORS.forEach(corridor => {
        const poly = L.polyline(corridor.coords, {
            color: corridor.color,
            weight: 3.5,
            opacity: 0.85,
            dashArray: '6, 4'
        }).addTo(corridorsLayerGroup);

        const tooltip = `<strong>${escapeHtml(corridor.name)}</strong><br><span style="color: ${corridor.color};">${corridor.risk}</span>`;
        poly.bindTooltip(tooltip, { className: 'leaflet-tooltip-dark', sticky: true });

        poly.on('click', (e) => {
            incidentMarker.setLatLng(e.latlng);
            updateCoordinateInputs(e.latlng.lat, e.latlng.lng);
            document.getElementById('activeSectorLabel').innerText = corridor.name;
        });
    });
}

function renderHotspotBadges() {
    const keyHotspots = [
        { name: "Gotera Interchange", lat: 8.9897, lng: 38.7523, risk: "Critical Density" },
        { name: "Modjo Dry Port Freight Junction", lat: 8.5925, lng: 39.1245, risk: "High Freight Hazard" },
        { name: "Kality Industrial Interchange", lat: 8.9220, lng: 38.7610, risk: "Frequent Collisions" },
        { name: "Entoto Mountain Hairpin", lat: 9.0820, lng: 38.7620, risk: "Fog / Steep Slope" }
    ];

    keyHotspots.forEach(spot => {
        const marker = L.circleMarker([spot.lat, spot.lng], {
            radius: 6,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.9
        }).addTo(hotspotsLayerGroup);

        marker.bindTooltip(`<strong>${escapeHtml(spot.name)}</strong><br>${spot.risk}`, { className: 'leaflet-tooltip-dark' });

        marker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            incidentMarker.setLatLng([spot.lat, spot.lng]);
            updateCoordinateInputs(spot.lat, spot.lng);
            document.getElementById('activeSectorLabel').innerText = spot.name;
        });
    });
}

function initMapControls() {
    const heatmapBtn = document.getElementById('toggleHeatmapBtn');
    const corridorsBtn = document.getElementById('toggleCorridorsBtn');
    const tileBtn = document.getElementById('toggleMapTileBtn');
    const recenterBtn = document.getElementById('recenterMapBtn');
    const mapElement = document.getElementById('incidentMap');

    heatmapBtn.addEventListener('click', () => {
        if (!heatLayer) return;
        if (map.hasLayer(heatLayer)) {
            map.removeLayer(heatLayer);
            heatmapBtn.classList.remove('active');
            heatmapBtn.innerText = 'Heatmap: Off';
        } else {
            map.addLayer(heatLayer);
            heatmapBtn.classList.add('active');
            heatmapBtn.innerText = 'Heatmap: On';
        }
    });

    corridorsBtn.addEventListener('click', () => {
        if (map.hasLayer(corridorsLayerGroup)) {
            map.removeLayer(corridorsLayerGroup);
            map.removeLayer(hotspotsLayerGroup);
            corridorsBtn.classList.remove('active');
            corridorsBtn.innerText = 'Corridors: Off';
        } else {
            map.addLayer(corridorsLayerGroup);
            map.addLayer(hotspotsLayerGroup);
            corridorsBtn.classList.add('active');
            corridorsBtn.innerText = 'Corridors: On';
        }
    });

    tileBtn.addEventListener('click', () => {
        mapElement.classList.toggle('clean-osm');
        const isClean = mapElement.classList.contains('clean-osm');
        tileBtn.innerText = isClean ? 'Tile: Light' : 'Tile: Dark';
        tileBtn.classList.toggle('active', isClean);
    });

    recenterBtn.addEventListener('click', () => {
        map.flyTo([8.9897, 38.7523], 12, { duration: 0.8 });
    });
}

function updateCoordinateInputs(lat, lng) {
    document.getElementById('start_lat').value = lat.toFixed(4);
    document.getElementById('start_lng').value = lng.toFixed(4);
}

function syncInputsToMap() {
    const lat = parseFloat(document.getElementById('start_lat').value);
    const lng = parseFloat(document.getElementById('start_lng').value);
    if (!isNaN(lat) && !isNaN(lng) && map && incidentMarker) {
        incidentMarker.setLatLng([lat, lng]);
        map.panTo([lat, lng]);
    }
}

// Preset Scenario Buttons
function initPresetListeners() {
    const buttons = document.querySelectorAll('.preset-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.getAttribute('data-preset');
            const preset = PRESETS[presetKey];
            if (!preset) return;

            document.getElementById('start_lat').value = preset.lat;
            document.getElementById('start_lng').value = preset.lng;
            document.getElementById('temp_f').value = preset.temp_f;
            document.getElementById('humidity').value = preset.humidity;
            document.getElementById('visibility').value = preset.visibility;
            document.getElementById('bump').checked = preset.bump;
            document.getElementById('crossing').checked = preset.crossing;
            document.getElementById('traffic_signal').checked = preset.traffic_signal;
            document.getElementById('weather_condition').value = preset.weather;
            document.getElementById('sunrise_sunset').value = preset.solar;

            document.getElementById('activeSectorLabel').innerText = preset.sector;

            if (map && incidentMarker) {
                incidentMarker.setLatLng([preset.lat, preset.lng]);
                map.flyTo([preset.lat, preset.lng], 13, { duration: 1.0 });
            }

            document.getElementById('accidentForm').dispatchEvent(new Event('submit'));
        });
    });
}

// Health Check
async function checkSystemHealth() {
    const statusLabel = document.getElementById('systemStatus');
    try {
        const response = await fetch('/api/v1/health');
        if (response.ok) {
            const data = await response.json();
            statusLabel.innerText = data.model_loaded ? 'Online' : 'Model Pending';
        }
    } catch (err) {
        statusLabel.innerText = 'Offline';
    }
}

// Form Submit Handler
document.getElementById('accidentForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const originalText = submitBtn.innerText;
    submitBtn.innerText = 'Evaluating Incident...';
    submitBtn.disabled = true;

    const payload = {
        "Start_Lat": parseFloat(document.getElementById('start_lat').value),
        "Start_Lng": parseFloat(document.getElementById('start_lng').value),
        "Temperature(F)": parseFloat(document.getElementById('temp_f').value),
        "Humidity(%)": parseFloat(document.getElementById('humidity').value),
        "Visibility(mi)": parseFloat(document.getElementById('visibility').value),
        "Bump": document.getElementById('bump').checked ? 1 : 0,
        "Crossing": document.getElementById('crossing').checked ? 1 : 0,
        "Traffic_Signal": document.getElementById('traffic_signal').checked ? 1 : 0,
        "Weather_Condition": document.getElementById('weather_condition').value,
        "Sunrise_Sunset": document.getElementById('sunrise_sunset').value,
    };

    try {
        const start = performance.now();
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Inference failure.');
        }

        const data = await response.json();
        const latency = (performance.now() - start).toFixed(1);

        lastPredictionResult = {
            ...data,
            inputPayload: payload,
            latency: latency,
            timestamp: new Date().toISOString()
        };

        renderResults(data, latency);

    } catch (err) {
        alert('Assessment Error: ' + err.message);
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
});

// Render Results
function renderResults(data, latency) {
    document.getElementById('placeholder').classList.add('hidden');
    const output = document.getElementById('resultContent');
    output.classList.remove('hidden');

    const banner = document.getElementById('priorityBanner');
    banner.className = `triage-banner ${data.priority_level}`;

    const badge = document.getElementById('priorityBadge');
    badge.innerText = data.priority_level;
    badge.className = `priority-tag ${data.priority_level}`;

    const latText = `${data.inference_time_ms ?? latency} ms`;
    document.getElementById('inferenceLatencyPill').innerText = latText;
    document.getElementById('telemetryLatency').innerText = latText;

    document.getElementById('severityTitle').innerText = data.severity_label;
    document.getElementById('protocolText').innerText = data.response_protocol;

    // Risk Factors
    const riskContainer = document.getElementById('riskFactorsContainer');
    riskContainer.innerHTML = '';
    const risks = data.contributing_risk_factors || [];

    if (risks.length === 0) {
        riskContainer.innerHTML = '<div class="risk-tag neutral">Standard ambient conditions</div>';
    } else {
        risks.forEach(r => {
            const el = document.createElement('div');
            el.className = 'risk-tag';
            el.innerText = r;
            riskContainer.appendChild(el);
        });
    }

    // Dispatch Actions
    const list = document.getElementById('dispatchList');
    list.innerHTML = '';
    data.dispatch_recommendation.forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        list.appendChild(li);
    });

    // Probability Bars
    const bars = document.getElementById('probabilityBars');
    bars.innerHTML = '';
    Object.entries(data.probabilities).forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
            <span class="bar-name">${cls}</span>
            <div class="bar-track">
                <div class="bar-fill ${data.priority_level}" style="width: ${pct}%;"></div>
            </div>
            <span class="bar-val">${pct}%</span>
        `;
        bars.appendChild(row);
    });
}

// Dispatch Ticket Modal
function initModalHandlers() {
    const modal = document.getElementById('ticketModal');
    const openBtn = document.getElementById('openTicketBtn');
    const closeBtn = document.getElementById('closeTicketBtn');
    const dismissBtn = document.getElementById('dismissTicketBtn');
    const printBtn = document.getElementById('printTicketBtn');

    openBtn.addEventListener('click', () => {
        if (!lastPredictionResult) return;
        populateTicket(lastPredictionResult);
        modal.classList.remove('hidden');
    });

    const close = () => modal.classList.add('hidden');
    closeBtn.addEventListener('click', close);
    dismissBtn.addEventListener('click', close);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) close();
    });

    printBtn.addEventListener('click', () => {
        window.print();
    });
}

function populateTicket(res) {
    const rand = Math.floor(1000 + Math.random() * 9000);
    document.getElementById('ticketRefCode').innerText = `INSA-RGA-${rand}`;
    document.getElementById('ticketTimestamp').innerText = new Date().toLocaleString();
    
    document.getElementById('ticketPriorityVal').innerText = `${res.priority_level} (${res.severity_label})`;

    const sector = document.getElementById('activeSectorLabel').innerText;
    document.getElementById('ticketCoordsVal').innerText = 
        `${res.inputPayload.Start_Lat.toFixed(4)} N, ${res.inputPayload.Start_Lng.toFixed(4)} E [${sector}]`;

    document.getElementById('ticketProtocolVal').innerText = res.response_protocol;

    const units = document.getElementById('ticketUnitsVal');
    units.innerHTML = '';
    res.dispatch_recommendation.forEach(u => {
        const li = document.createElement('li');
        li.innerText = u;
        units.appendChild(li);
    });

    const risks = document.getElementById('ticketRisksVal');
    risks.innerHTML = '';
    (res.contributing_risk_factors || []).forEach(r => {
        const li = document.createElement('li');
        li.innerText = r;
        risks.appendChild(li);
    });
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}