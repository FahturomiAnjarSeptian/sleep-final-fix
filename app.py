from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # Wajib untuk server
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- LOAD MODEL (Versi Simple) ---
forest = []
X_min = 0; X_max = 1
feature_names = []

try:
    # Kita coba load langsung. Jika gagal, nanti web tetap jalan tapi error text.
    with open('model_sleep.pkl', 'rb') as f:
        data = pickle.load(f)
    forest = data['forest']
    X_min = data['X_min']
    X_max = data['X_max']
    feature_names = data['feature_names']
except:
    print("Model belum ditemukan, jalankan train_model.py dulu.")

# --- FUNGSI PREDIKSI ---
def predict_tree(tree, x):
    if 'label' in tree: return tree['label']
    if x[tree['feature']] < tree['threshold']: return predict_tree(tree['left'], x)
    else: return predict_tree(tree['right'], x)

def predict_forest(trees, input_x):
    if not trees: return 0
    preds = [predict_tree(t, input_x[0]) for t in trees]
    return int(np.round(np.mean(preds)))

def get_tree_image(tree, feature_names, title):
    plt.figure(figsize=(6, 4))
    ax = plt.gca(); ax.set_title(title); ax.axis("off")
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
    img.seek(0); plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = []
    
    if request.method == 'POST':
        try:
            # Ambil data form
            vals = [
                float(request.form.get('gender')), float(request.form.get('age')),
                float(request.form.get('occupation')), float(request.form.get('sleep_duration')),
                float(request.form.get('quality_sleep')), float(request.form.get('phys_activity')),
                float(request.form.get('stress')), float(request.form.get('bmi')),
                float(request.form.get('heart_rate')), float(request.form.get('daily_steps')),
                float(request.form.get('systolic')), float(request.form.get('diastolic'))
            ]
            
            input_arr = np.array([vals])
            input_scaled = (input_arr - X_min) / (X_max - X_min + 1e-8)
            
            pred = predict_forest(forest, input_scaled)
            
            res_str = "NORMAL (SEHAT)" if pred == 0 else "GANGGUAN TIDUR"
            color = "green" if pred == 0 else "red"
            prediction_text = f"<h2 style='color:{color}'>{res_str}</h2>"
            
            if forest:
                for i in range(min(3, len(forest))):
                    img = get_tree_image(forest[i], feature_names, f"Pohon {i+1}")
                    tree_plots.append(img)
                    
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)