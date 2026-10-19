document.getElementById('accidentForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    submitBtn.innerText = 'Analyzing Severity & Route Response...';
    submitBtn.disabled = true;

    // Send payload aligned with US_Accidents_March23 schema
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
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Failed to analyze accident parameters.');
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        submitBtn.innerText = 'Predict Severity & Dispatch Strategy';
        submitBtn.disabled = false;
    }
});

function renderResults(data) {
    document.getElementById('placeholder').classList.add('hidden');
    const resultContent = document.getElementById('resultContent');
    resultContent.classList.remove('hidden');

    const badge = document.getElementById('priorityBadge');
    badge.innerText = data.priority_level;
    badge.className = `badge ${data.priority_level}`;

    document.getElementById('severityTitle').innerText = data.severity_label;
    document.getElementById('protocolText').innerText = data.response_protocol;

    const dispatchList = document.getElementById('dispatchList');
    dispatchList.innerHTML = '';
    data.dispatch_recommendation.forEach(item => {
        const li = document.createElement('li');
        li.innerText = `✓ ${item}`;
        dispatchList.appendChild(li);
    });

    const probBars = document.getElementById('probabilityBars');
    probBars.innerHTML = '';
    Object.entries(data.probabilities).forEach(([key, value]) => {
        const percent = (value * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'prob-item';
        div.innerHTML = `
            <span style="width: 85px;">${key}</span>
            <div class="prob-bar-outer">
                <div class="prob-bar-inner" style="width: ${percent}%;"></div>
            </div>
            <span style="width: 45px; text-align: right;">${percent}%</span>
        `;
        probBars.appendChild(div);
    });
}