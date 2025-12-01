from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sys

app = Flask(__name__)

# --- LOAD MODEL ---
model_path = '/home/anjarranjayy/sleep-final-fix/model_sleep.pkl'
# (Path lengkap agar aman, sesuaikan username jika perlu)

data = {}
try:
    with open('model_sleep.pkl', 'rb') as f:
        data = pickle.load(f)
except:
    # Coba path absolute jika relative gagal
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"ERROR LOAD MODEL: {e}", file=sys.stderr)

forest = data.get('forest', [])
X_min = data.get('X_min', np.zeros(12))
X_max = data.get('X_max', np.ones(12))
feature_names = data.get('feature_names', [])

def predict_single_tree(node, x):
    if 'label' in node:
        return node['label']
    
    # Cek input feature index
    feat_idx = node['feature']
    threshold = node['threshold']
    
    if x[feat_idx] < threshold:
        return predict_single_tree(node['left'], x)
    else:
        return predict_single_tree(node['right'], x)

def get_tree_image(tree, feature_names, title):
    try:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.set_title(title)
        ax.axis("off")
        
        def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
            if 'label' in node:
                ax.text(x, y, f"Leaf:{int(node['label'])}", ha='center', bbox=dict(boxstyle="round", fc="lightgreen"))
                return
            
            fname = feature_names[node['feature']] if feature_names else str(node['feature'])
            ax.text(x, y, f"{fname}\n<{node['threshold']:.2f}", ha='center', bbox=dict(boxstyle="round", fc="lightblue"))
            
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
    except:
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = []
    
    if request.method == 'POST':
        try:
            # URUTAN INI HARUS SAMA PERSIS DENGAN train_model.py
            # 1. Gender, 2. Age, 3. Occ, 4. SleepDur, 5. QualSleep, 
            # 6. PhysAct, 7. Stress, 8. BMI, 9. Heart, 10. Steps, 
            # 11. SysBP, 12. DiaBP
            
            vals = [
                float(request.form.get('gender')),
                float(request.form.get('age')),
                float(request.form.get('occupation')),
                float(request.form.get('sleep_duration')),
                float(request.form.get('quality_sleep')),
                float(request.form.get('phys_activity')),
                float(request.form.get('stress')),
                float(request.form.get('bmi')),
                float(request.form.get('heart_rate')),
                float(request.form.get('daily_steps')),
                float(request.form.get('systolic')),
                float(request.form.get('diastolic'))
            ]
            
            input_arr = np.array(vals)
            
            # Debug: Print input ke Error Log
            print(f"INPUT USER: {input_arr}", file=sys.stderr)
            
            # Normalisasi
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            # Prediksi Voting
            votes = []
            for t in forest:
                votes.append(predict_single_tree(t, input_scaled))
            
            final_pred = int(np.round(np.mean(votes)))
            print(f"HASIL VOTING: {votes} -> {final_pred}", file=sys.stderr)
            
            status = "NORMAL (SEHAT)" if final_pred == 0 else "TERDETEKSI GANGGUAN TIDUR"
            color = "green" if final_pred == 0 else "red"
            prediction_text = f"<h3 style='color:{color}'>{status}</h3>"
            
            # Visualisasi
            for i, tree in enumerate(forest):
                plot = get_tree_image(tree, feature_names, f"Tree {i+1}")
                if plot: tree_plots.append(plot)

        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            print(f"ERROR APP: {e}", file=sys.stderr)

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)