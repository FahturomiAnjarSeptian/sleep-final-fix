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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    forest = data.get('forest', [])
    X_min = data.get('X_min', 0)
    X_max = data.get('X_max', 1)
    feature_names = data.get('feature_names', [])
except:
    print("Model Error", file=sys.stderr)

def predict_single_tree(tree, x):
    if not isinstance(tree, dict): return 0
    if 'label' in tree: return tree['label']
    if tree['feature'] >= len(x): return 0
    if x[tree['feature']] < tree['threshold']:
        return predict_single_tree(tree['left'], x)
    else:
        return predict_single_tree(tree['right'], x)

def get_tree_image(tree, feature_names, title):
    try:
        plt.figure(figsize=(6, 4))
        ax = plt.gca(); ax.set_title(title); ax.axis("off")
        def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
            if 'label' in node:
                ax.text(x, y, f"Leaf:{int(node['label'])}", ha='center', bbox=dict(boxstyle="round", fc="lightgreen"))
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
            # INPUT
            raw_vals = [
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
            
            input_arr = np.array(raw_vals)
            # Normalisasi
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            # Prediksi & DEBUG LOG
            votes = []
            debug_info = "<div style='font-size:12px; text-align:left; margin-top:10px; background:#f0f0f0; padding:10px;'>"
            debug_info += "<b>--- DEBUG INFO (Screenshot Ini) ---</b><br>"
            
            for i, t in enumerate(forest):
                v = predict_single_tree(t, input_scaled)
                # Cek hasil vote
                vote_res = "NaN"
                if v is not None and not np.isnan(v):
                    vote_res = int(v)
                    votes.append(vote_res)
                debug_info += f"Pohon {i+1}: {vote_res}<br>"
            
            # Hitung Final
            final_pred = 0
            avg_vote = 0
            if votes:
                avg_vote = np.mean(votes)
                final_pred = int(np.round(avg_vote))
            
            debug_info += f"<b>Rata-rata Vote: {avg_vote:.2f} -> Hasil Akhir: {final_pred}</b><br>"
            debug_info += f"Input Scaled (Contoh Age): {input_scaled[1]:.2f} (Harusnya 0.0 - 1.0)<br>"
            debug_info += "</div>"

            # TAMPILAN HASIL
            status = "NORMAL (SEHAT)" if final_pred == 0 else "TERDETEKSI GANGGUAN TIDUR!"
            color = "green" if final_pred == 0 else "red"
            
            # KITA TEMPEL DEBUG INFO DI BAWAH HASIL
            prediction_text = f"<h2 style='color:{color}'>{status}</h2>{debug_info}"
            
            # Gambar (Limit 3 pohon)
            if forest:
                for i in range(min(3, len(forest))):
                    img = get_tree_image(forest[i], feature_names, f"Pohon {i+1}")
                    if img: tree_plots.append(img)
                        
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)