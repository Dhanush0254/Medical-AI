import os
import joblib
import pandas as pd
import warnings
import logging
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
from extractor import extract_data

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
MODELS_DIR = BASE_DIR / 'models'

UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'csv', 'json'}

model_cache = {}
def get_model(name):
    if name not in model_cache:
        path = MODELS_DIR / f"{name}_model.joblib"
        model_cache[name] = joblib.load(path) if path.exists() else None
    return model_cache[name]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_risk(data):
    preds, suggs = [], []
    has_data = False
    age = data.get('age', 45)

    # --- 1. DIABETES CHECK ---
    gluc, hba1c = data.get('glucose'), data.get('hba1c')
    status, reason = "Unknown", "Insufficient Data"
    
    if gluc or hba1c:
        has_data = True
        status, reason = "Healthy", "Normal Levels"
        
        # Rule-based Check
        if (gluc and gluc > 140) or (hba1c and hba1c > 6.5):
            status, reason = "High Risk", "Elevated Glucose/A1C"
            suggs.append("⚠️ Diabetes: Consult a Diabetologist. Reduce sugar intake and monitor blood glucose.")
        else:
            # AI Model Check
            m = get_model('diabetes')
            if m and m.predict(pd.DataFrame([[gluc or 100, hba1c or 5.5, age, 25]], columns=['glucose','hba1c','age','bmi']))[0] == 1:
                status, reason = "High Risk", "AI Pattern Detection"
                suggs.append("⚠️ Diabetes Pattern: AI detects subtle patterns. Consider a preventive checkup.")
    
    preds.append({"condition": "Diabetes", "risk": status, "reason": reason})

    # --- 2. HEART CHECK ---
    chol, ldl = data.get('cholesterol'), data.get('ldl')
    status, reason = "Unknown", "Insufficient Data"
    
    if chol or ldl:
        has_data = True
        status, reason = "Healthy", "Normal Lipid Profile"
        
        # Rule-based
        if (chol and chol > 240) or (ldl and ldl > 160):
            status, reason = "High Risk", "High Cholesterol/LDL"
            suggs.append("❤️ Heart: Limit saturated fats (red meat, fried food). Consider cardio exercises.")
        else:
            # AI Model
            m = get_model('cardio')
            if m and m.predict(pd.DataFrame([[chol or 180, ldl or 100, data.get('hdl',50), data.get('triglycerides',150), age]], columns=['cholesterol','ldl','hdl','triglycerides','age']))[0] == 1:
                status, reason = "High Risk", "AI Anomaly Detected"
                suggs.append("❤️ Heart Pattern: AI found potential risks. Monitor blood pressure and lipids.")

    preds.append({"condition": "Heart", "risk": status, "reason": reason})

    # --- 3. ANEMIA CHECK ---
    hb = data.get('hemoglobin')
    status, reason = "Unknown", "Insufficient Data"

    if hb:
        has_data = True
        if hb < 13: # Simplified threshold
            status, reason = "High Risk", f"Low Hemoglobin ({hb})"
            suggs.append("🩸 Anemia: Increase iron-rich foods (spinach, dates, red meat). Consult a doctor.")
        else:
            status, reason = "Healthy", "Normal Levels"
            # AI Model
            m = get_model('anemia')
            if m and m.predict(pd.DataFrame([[hb, data.get('red_blood_cells', 4.5), age]], columns=['hemoglobin','red_blood_cells','age']))[0] == 1:
                status, reason = "High Risk", "AI Flagged"
                suggs.append("🩸 Anemia Pattern: AI suggests further investigation despite normal levels.")

    preds.append({"condition": "Anemia", "risk": status, "reason": reason})
    
    # --- 4. GENERAL SUGGESTIONS ---
    if not has_data: 
        suggs = ["ℹ️ No readable data found. Please enter values manually or upload a clearer file."]
    elif not suggs: 
        suggs.append("🎉 Great News: Your vitals look healthy! Keep up the good lifestyle.")
    
    return preds, suggs

# UI TEMPLATE
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #2563eb; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding: 20px; margin: 0; }
        .container { max-width: 650px; margin: 0 auto; display: flex; flex-direction: column; min-height: 95vh; }
        .main-content { flex: 1; }
        
        .header-section { text-align: center; margin-bottom: 30px; }
        h1 { margin-bottom: 5px; font-size: 2.2rem; }
        .credits { color: #94a3b8; font-size: 0.9rem; letter-spacing: 0.5px; }
        .credits span { color: var(--primary); font-weight: 600; }

        .glass-card { background: var(--card); border: 1px solid #334155; border-radius: 16px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .upload-zone { border: 2px dashed #475569; border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; transition: 0.3s; }
        .upload-zone:hover { border-color: var(--primary); background: rgba(37, 99, 235, 0.1); }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 25px; }
        @media (max-width: 600px) {
            .grid { grid-template-columns: 1fr; }
            h1 { font-size: 1.8rem; }
            .container { padding: 0 5px; }
        }

        input { width: 100%; padding: 12px; background: #020617; border: 1px solid #334155; color: white; border-radius: 8px; box-sizing: border-box; }
        .btn { width: 100%; padding: 16px; border-radius: 10px; border: none; font-weight: 700; background: var(--primary); color: white; margin-top: 20px; cursor: pointer; }
        .badge { padding: 4px 10px; border-radius: 99px; font-size: 0.75rem; font-weight: bold; }

        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155; color: #64748b; font-size: 0.8rem; }

        /* LOADER */
        .loader-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(5px);
            display: none; justify-content: center; align-items: center; flex-direction: column; z-index: 9999;
        }
        .spinner {
            width: 50px; height: 50px; border: 4px solid rgba(37, 99, 235, 0.3);
            border-radius: 50%; border-top: 4px solid var(--primary);
            animation: spin 1s linear infinite; margin-bottom: 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .loading-text { color: var(--text); font-weight: 600; font-size: 1.1rem; }
    </style>
</head>
<body>

<div id="loader" class="loader-overlay">
    <div class="spinner"></div>
    <div class="loading-text" id="loaderText">Processing...</div>
</div>

<div class="container">
    <div class="main-content">
        <div class="header-section">
            <h1>Medical AI 🧬</h1>
            <div class="credits">Developed by <span>Dhanush</span> | <span>Majin Studio</span> | <span>Prince Vegeta</span></div>
        </div>
        
        <div class="glass-card">
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <div id="uText"><strong>Click to Upload</strong><br><small style="color:#94a3b8">IMG, PDF, CSV, JSON</small></div>
                <img id="preview" style="max-width:100%; display:none; border-radius:8px; margin-top:10px;">
            </div>
            <input type="file" id="fileInput" hidden onchange="handleFile(this)">

            <div class="grid">
                <div><label>Glucose</label><input type="number" id="glucose"></div>
                <div><label>HbA1c</label><input type="number" step="0.1" id="hba1c"></div>
                <div><label>Hemoglobin</label><input type="number" step="0.1" id="hemoglobin"></div>
                <div><label>Cholesterol</label><input type="number" id="cholesterol"></div>
                <div><label>LDL</label><input type="number" id="ldl"></div>
                <div><label>Triglycerides</label><input type="number" id="triglycerides"></div>
            </div>

            <button class="btn" onclick="analyze()">RUN DIAGNOSTICS</button>
        </div>

        <div id="results" style="margin-top:20px; display:none;"></div>
    </div>

    <div class="footer">
        <p><strong>Disclaimer:</strong> This application is a virtual demonstration. The prediction values generated are virtual estimates and should not be considered as actual medical advice. Always consult a certified medical professional for diagnosis.</p>
        <p>&copy; 2025 Majin Studio. All rights reserved.</p>
    </div>
</div>

<script>
function showLoader(text) {
    document.getElementById('loaderText').innerText = text;
    document.getElementById('loader').style.display = 'flex';
}
function hideLoader() { document.getElementById('loader').style.display = 'none'; }

async function handleFile(input) {
    const file = input.files[0];
    if (!file) return;

    showLoader("Scanning Document...");
    document.querySelectorAll('input').forEach(i => { i.value = ''; i.style.borderColor = '#334155'; });
    document.getElementById('results').style.display = 'none';
    
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview').src = e.target.result;
            document.getElementById('preview').style.display = 'block';
            document.getElementById('uText').style.display = 'none';
        };
        reader.readAsDataURL(file);
    } else {
        document.getElementById('uText').innerHTML = `<strong>${file.name}</strong>`;
        document.getElementById('preview').style.display = 'none';
        document.getElementById('uText').style.display = 'block';
    }

    const fd = new FormData(); fd.append('file', file);
    try {
        const res = await fetch('/extract', { method: 'POST', body: fd });
        const data = await res.json();
        
        if(data.error) alert(data.error);
        else {
            if (Object.keys(data).length === 0) alert("⚠️ No medical data found.");
            for (const [k, v] of Object.entries(data)) {
                const el = document.getElementById(k);
                if(el) { el.value = v; el.style.borderColor = '#22c55e'; }
            }
        }
    } catch { alert("Extraction Error."); } 
    finally { hideLoader(); }
}

async function analyze() {
    showLoader("Analyzing Health Data...");
    const payload = {};
    document.querySelectorAll('input').forEach(i => { if(i.value) payload[i.id] = parseFloat(i.value); });
    
    try {
        const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json();
        
        let html = '';
        
        // 1. Predictions
        data.predictions.forEach(p => {
            const color = p.risk.includes('High') ? '#ef4444' : '#22c55e';
            html += `<div class="glass-card" style="margin-bottom:10px; border-left: 4px solid ${color}; display:flex; justify-content:space-between; align-items:center;">
                <div><strong>${p.condition}</strong><br><small style="color:#94a3b8">${p.reason}</small></div>
                <span class="badge" style="color:${color}; border:1px solid ${color}">${p.risk}</span>
            </div>`;
        });

        // 2. Recommendations (NEW)
        if(data.suggestions && data.suggestions.length > 0) {
            html += `<div class="glass-card" style="margin-top:15px; border-left: 4px solid #f59e0b;">
                <strong>💡 Recommendations:</strong>
                <ul style="margin: 10px 0 0 20px; padding: 0; color: #cbd5e1; font-size: 0.95rem;">`;
            data.suggestions.forEach(s => {
                html += `<li style="margin-bottom:8px;">${s}</li>`;
            });
            html += `</ul></div>`;
        }

        document.getElementById('results').innerHTML = html;
        document.getElementById('results').style.display = 'block';
    } catch { alert("Analysis Error."); } 
    finally { hideLoader(); }
}
</script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_UI)

@app.route('/extract', methods=['POST'])
def extract():
    file = request.files.get('file')
    if not file: return jsonify({"error": "No file uploaded"}), 400
    if allowed_file(file.filename):
        path = UPLOAD_FOLDER / file.filename
        file.save(path)
        try:
            print(f"📥 Processing: {file.filename}")
            data = extract_data(path)
            print(f"📤 Extracted Data: {data}")
            try: os.remove(path)
            except: pass
            return jsonify(data)
        except Exception as e: return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(dict(zip(['predictions', 'suggestions'], predict_risk(request.json))))

if __name__ == '__main__':
    # Local development server
    app.run(host='0.0.0.0', port=5000, debug=True)
