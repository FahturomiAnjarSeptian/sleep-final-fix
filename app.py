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

# --- KUNCI LOKASI PASTI (SAMA DENGAN TRAIN) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

# VARIABEL DEFAULT (PENTING AGAR TIDAK ERROR 'NOT DEFINED')
forest = []
X_min = np.zeros(12)
X_max = np.ones(12)
feature_names = []

# LOAD MODEL
try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    forest = data.get('forest', [])
    X_min = data.get('X_min', np.zeros(12))
    X_max = data.get('X_max', np.ones(12))
    feature_names = data.get('feature_names', [])
    print(f"INFO: Model dimuat dari {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"ERROR LOAD MODEL: {e}", file=sys.stderr)

# FUNGSI PENDUKUNG
def predict_tree(node, x):
    if not isinstance(node, dict): return 0
    if 'label' in node: return node['label']
    if node['feature'] >= len(x): return 0
    if x[node['feature']] < node['threshold']: return predict_tree(node['left'], x)
    else: return predict_tree(node['right'], x)

def get_tree_image(tree, fnames, title):
    try:
        plt.figure(figsize=(6, 4))
        ax = plt.gca(); ax.set_title(title); ax.axis("off")
        def recurse(n, x=0.5, y=1.0, dx=0.25, dy=0.15):
            if 'label' in n:
                val = int(n['label']) if not np.isnan(n['label']) else 0
                ax.text(x, y, f"L:{val}", ha='center', bbox=dict(boxstyle="round", fc="#90EE90"))
                return
            fn = str(n['feature'])
            if fnames and n['feature'] < len(fnames): fn = fnames[n['feature']]
            ax.text(x, y, f"{fn}<{n['threshold']:.1f}", ha='center', bbox=dict(boxstyle="round", fc="#ADD8E6"))
            ax.plot([x, x-dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(n['left'], x-dx, y-dy, dx*0.5, dy)
            ax.plot([x, x+dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(n['right'], x+dx, y-dy, dx*0.5, dy)
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
            # Cek Model
            if not forest:
                return render_template('index.html', prediction_text=f"<h3 style='color:red'>Error: File Model Tidak Ada di {MODEL_PATH}. Jalankan 'python train_model.py' di Console!</h3>")

            raw = [
                float(request.form.get('gender', 0)), float(request.form.get('age', 30)),
                float(request.form.get('occupation', 2)), float(request.form.get('sleep_duration', 7)),
                float(request.form.get('quality_sleep', 7)), float(request.form.get('phys_activity', 60)),
                float(request.form.get('stress', 5)), float(request.form.get('bmi', 0)),
                float(request.form.get('heart_rate', 70)), float(request.form.get('daily_steps', 5000)),
                float(request.form.get('systolic', 120)), float(request.form.get('diastolic', 80))
            ]
            
            x = (np.array(raw) - X_min) / (X_max - X_min + 1e-8)
            
            votes = []
            for t in forest:
                v = predict_tree(t, x)
                if not np.isnan(v): votes.append(v)
            
            final = 0
            if votes: final = int(np.round(np.mean(votes)))

            res = "NORMAL / SEHAT" if final == 0 else "TERDETEKSI GANGGUAN TIDUR"
            col = "green" if final == 0 else "red"
            prediction_text = f"<h2 style='color:{col}'>{res}</h2>"
            
            for i in range(min(3, len(forest))):
                img = get_tree_image(forest[i], feature_names, f"Tree {i+1}")
                if img: tree_plots.append(img)
                        
        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)