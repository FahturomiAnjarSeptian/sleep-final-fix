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

# --- CONFIG ---
# Gunakan path absolut agar aman di server
# Ganti 'anjarranjayy' dengan username PythonAnywhere Anda jika berbeda
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

# --- LOAD MODEL ---
forest = []
X_min = 0
X_max = 1
feature_names = []

try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    forest = data.get('forest', [])
    X_min = data.get('X_min', 0)
    X_max = data.get('X_max', 1)
    feature_names = data.get('feature_names', [])
    print("Model berhasil dimuat.", file=sys.stderr)
except Exception as e:
    print(f"ERROR MEMUAT MODEL: {e}", file=sys.stderr)

# --- FUNGSI PREDIKSI ---
def predict_single_tree(tree, x):
    if not isinstance(tree, dict): return 0
    if 'label' in tree:
        return tree['label']
    
    # Cek apakah fitur valid
    if tree['feature'] >= len(x): return 0
    
    if x[tree['feature']] < tree['threshold']:
        return predict_single_tree(tree['left'], x)
    else:
        return predict_single_tree(tree['right'], x)

def get_tree_image(tree, feature_names, title):
    try:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        
        def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
            if 'label' in node:
                # Pastikan label angka valid
                val = 0
                if not np.isnan(node['label']): val = int(node['label'])
                
                ax.text(x, y, f"Leaf: {val}", ha='center', va='center', 
                        bbox=dict(boxstyle="round", fc="#90EE90"))
                return
            
            fname = str(node['feature'])
            if feature_names and node['feature'] < len(feature_names):
                fname = feature_names[node['feature']]
                
            ax.text(x, y, f"{fname}\n< {node['threshold']:.2f}", 
                    ha='center', va='center', bbox=dict(boxstyle="round", fc="#ADD8E6"))
            
            ax.plot([x, x-dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(node['left'], x-dx, y-dy, dx*0.5, dy)
            ax.plot([x, x+dx], [y-0.02, y-dy+0.02], 'k-')
            recurse(node['right'], x+dx, y-dy, dx*0.5, dy)

        recurse(tree)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Gagal gambar tree: {e}", file=sys.stderr)
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = []
    
    if request.method == 'POST':
        try:
            # URUTAN HARUS SAMA PERSIS DENGAN TRAIN_MODEL.PY (12 FITUR)
            # ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
            #  'Physical Activity Level', 'Stress Level', 'BMI Category', 
            #  'Heart Rate', 'Daily Steps', 'systolic', 'diastolic']
            
            vals = [
                float(request.form.get('gender', 0)),
                float(request.form.get('age', 30)),
                float(request.form.get('occupation', 2)),
                float(request.form.get('sleep_duration', 7)),
                float(request.form.get('quality_sleep', 7)),
                float(request.form.get('phys_activity', 60)),
                float(request.form.get('stress', 5)),
                float(request.form.get('bmi', 0)),
                float(request.form.get('heart_rate', 70)),
                float(request.form.get('daily_steps', 5000)),
                float(request.form.get('systolic', 120)),
                float(request.form.get('diastolic', 80))
            ]
            
            input_arr = np.array(vals)
            # Normalisasi
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            # Prediksi Voting (Anti-Crash)
            votes = []
            if forest:
                for t in forest:
                    v = predict_single_tree(t, input_scaled)
                    if v is not None and not np.isnan(v):
                        votes.append(v)
            
            # Hitung rata-rata voting
            final_pred = 0
            if votes:
                avg = np.mean(votes)
                if not np.isnan(avg):
                    final_pred = int(np.round(avg))
            
            # Tampilkan Hasil
            status = "NORMAL (SEHAT)" if final_pred == 0 else "TERDETEKSI GANGGUAN TIDUR"
            color = "green" if final_pred == 0 else "red"
            prediction_text = f"<h3 style='color:{color}'>{status}</h3>"
            
            # Gambar visualisasi
            if forest:
                for i, tree in enumerate(forest):
                    # Hanya gambar 3 pohon pertama agar loading tidak terlalu lama
                    if i < 3: 
                        img = get_tree_image(tree, feature_names, f"Pohon Keputusan #{i+1}")
                        if img: tree_plots.append(img)
                        
        except Exception as e:
            prediction_text = f"<span class='text-danger'>Error: {str(e)}</span>"
            print(f"Error di proses prediksi: {e}", file=sys.stderr)

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)