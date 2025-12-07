from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

app = Flask(__name__)

# --- KONFIGURASI JALUR MUTLAK (SAMA DENGAN TRAIN_MODEL) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

# --- DEFINISI VARIABEL ANTI-CRASH (Inisialisasi Awal) ---
# Kita buat variabel ini DULUAN sebelum load model
# Supaya tidak pernah ada error "NameError: X_min is not defined"
forest = []
feature_names = []
# Default dummy (angka 0) agar perhitungan matematika tidak error
X_min = np.zeros(12) 
X_max = np.ones(12)

# --- LOAD MODEL ---
try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    # Jika berhasil, baru kita timpa variabel dummy tadi
    forest = data.get('forest', [])
    X_min = data.get('X_min', np.zeros(12))
    X_max = data.get('X_max', np.ones(12))
    feature_names = data.get('feature_names', [])
    print(f"INFO: Model berhasil dimuat dari {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    # Jika gagal, dia akan tetap jalan pakai variabel dummy (tidak crash)
    print(f"WARNING: Gagal load model dari {MODEL_PATH}. Error: {e}", file=sys.stderr)

# --- FUNGSI PENDUKUNG ---
def predict_single_tree(tree, x):
    if not isinstance(tree, dict): return 0
    if 'label' in tree: return tree['label']
    if tree['feature'] >= len(x): return 0
    if x[tree['feature']] < tree['threshold']: return predict_single_tree(tree['left'], x)
    else: return predict_single_tree(tree['right'], x)

def get_tree_image(tree, feature_names, title):
    try:
        plt.figure(figsize=(6, 4))
        ax = plt.gca(); ax.set_title(title); ax.axis("off")
        def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
            if 'label' in node:
                val = 0
                if not np.isnan(node['label']): val = int(node['label'])
                ax.text(x, y, f"Leaf:{val}", ha='center', bbox=dict(boxstyle="round", fc="lightgreen"))
                return
            fname = str(node['feature'])
            if feature_names and node['feature'] < len(feature_names): fname = feature_names[node['feature']]
            ax.text(x, y, f"{fname}\n<{node['threshold']:.2f}", ha='center', bbox=dict(boxstyle="round", fc="lightblue"))
            ax.plot([x, x-dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(node['left'], x-dx, y-dy, dx*0.5, dy)
            ax.plot([x, x+dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(node['right'], x+dx, y-dy, dx*0.5, dy)
        recurse(tree)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0); plt.close()
        return base64.b64encode(img.getvalue()).decode()
    except: return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = []
    
    if request.method == 'POST':
        try:
            # Cek apakah model sudah terisi
            if not forest:
                return render_template('index.html', prediction_text=f"<h3 style='color:red'>Error: Model file tidak ditemukan di {MODEL_PATH}. Jalankan python train_model.py di server!</h3>")

            # Input Form (12 Fitur)
            raw_vals = [
                float(request.form.get('gender', 0)), float(request.form.get('age', 30)),
                float(request.form.get('occupation', 2)), float(request.form.get('sleep_duration', 7)),
                float(request.form.get('quality_sleep', 7)), float(request.form.get('phys_activity', 60)),
                float(request.form.get('stress', 5)), float(request.form.get('bmi', 0)),
                float(request.form.get('heart_rate', 70)), float(request.form.get('daily_steps', 5000)),
                float(request.form.get('systolic', 120)), float(request.form.get('diastolic', 80))
            ]
            
            input_arr = np.array(raw_vals)
            # Scaling aman karena X_min sudah pasti ada (meski dummy)
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            votes = []
            debug_info = "<div style='background:#f9f9f9; padding:10px; font-size:12px; margin-top:10px; border:1px solid #ccc'><b>🔍 DEBUG MODE:</b><br>"
            
            for i, t in enumerate(forest):
                v = predict_single_tree(t, input_scaled)
                val = 0
                if v is not None and not np.isnan(v): val = int(v)
                votes.append(val)
                debug_info += f"Pohon {i+1}: Prediksi {val}<br>"
            
            final_pred = 0
            if votes: final_pred = int(np.round(np.mean(votes)))
            
            debug_info += f"<br><b>Rata-rata: {np.mean(votes):.2f}</b><br><b>Keputusan Akhir: {final_pred}</b></div>"

            status = "NORMAL (SEHAT)" if final_pred == 0 else "TERDETEKSI GANGGUAN TIDUR"
            color = "green" if final_pred == 0 else "red"
            prediction_text = f"<h2 style='color:{color}'>{status}</h2>{debug_info}"
            
            for i in range(min(3, len(forest))):
                img = get_tree_image(forest[i], feature_names, f"Struktur Pohon {i+1}")
                if img: tree_plots.append(img)
                        
        except Exception as e:
            prediction_text = f"<span style='color:red'>Error Sistem: {str(e)}</span>"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)