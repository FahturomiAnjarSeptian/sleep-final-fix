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

# --- KONFIGURASI PATH ABSOLUT (PENTING) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

# --- DEFINISI VARIABEL GLOBAL (Agar tidak error 'Not Defined') ---
forest = []
X_min = np.zeros(12) # Default dummy
X_max = np.ones(12)  # Default dummy
feature_names = []

# --- LOAD MODEL ---
try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    forest = data.get('forest', [])
    X_min = data.get('X_min', np.zeros(12))
    X_max = data.get('X_max', np.ones(12))
    feature_names = data.get('feature_names', [])
    print(f"Model berhasil dimuat dari {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"GAGAL LOAD MODEL dari {MODEL_PATH}: {e}", file=sys.stderr)

# --- FUNGSI ---
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
            # PENTING: Jika model kosong, beri peringatan tapi JANGAN CRASH
            if not forest:
                return render_template('index.html', prediction_text="<h3 style='color:red'>Error: Model file tidak ditemukan atau kosong. Cek Log Error.</h3>")

            raw_vals = [
                float(request.form.get('gender', 0)), float(request.form.get('age', 30)),
                float(request.form.get('occupation', 2)), float(request.form.get('sleep_duration', 7)),
                float(request.form.get('quality_sleep', 7)), float(request.form.get('phys_activity', 60)),
                float(request.form.get('stress', 5)), float(request.form.get('bmi', 0)),
                float(request.form.get('heart_rate', 70)), float(request.form.get('daily_steps', 5000)),
                float(request.form.get('systolic', 120)), float(request.form.get('diastolic', 80))
            ]
            
            input_arr = np.array(raw_vals)
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            votes = []
            debug_info = "<div style='background:#eee; padding:10px; font-size:12px; margin-top:10px'><b>DEBUG:</b><br>"
            for i, t in enumerate(forest):
                v = predict_single_tree(t, input_scaled)
                if v is not None and not np.isnan(v):
                    votes.append(int(v))
                    debug_info += f"Tree {i+1}: {int(v)}<br>"
            
            final_pred = 0
            if votes: final_pred = int(np.round(np.mean(votes)))
            debug_info += f"<b>Final: {final_pred}</b></div>"

            status = "NORMAL (SEHAT)" if final_pred == 0 else "TERDETEKSI GANGGUAN TIDUR"
            color = "green" if final_pred == 0 else "red"
            prediction_text = f"<h2 style='color:{color}'>{status}</h2>{debug_info}"
            
            for i in range(min(3, len(forest))):
                img = get_tree_image(forest[i], feature_names, f"Tree {i+1}")
                if img: tree_plots.append(img)
                        
        except Exception as e:
            prediction_text = f"Error System: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)