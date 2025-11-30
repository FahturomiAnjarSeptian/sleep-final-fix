from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # PENTING AGAR TIDAK ERROR DI SERVER
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load Model
model_path = 'model_sleep.pkl'
forest = []
X_min = 0
X_max = 1
feature_names = []

try:
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    forest = saved_data['forest']
    X_min = saved_data['X_min']
    X_max = saved_data['X_max']
    feature_names = saved_data['feature_names']
except:
    print("Model belum ada. Jalankan train_model.py dulu.")

def predict_tree(tree, x):
    if 'label' in tree: return tree['label']
    if x[tree['feature']] < tree['threshold']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def predict_forest(trees, X_input):
    preds = [predict_tree(t, X_input[0]) for t in trees]
    return int(np.round(np.mean(preds)))

def get_tree_image(tree, feature_names, title):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    
    def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
        if 'label' in node:
            ax.text(x, y, f"Leaf: {int(node['label'])}", ha='center', va='center', bbox=dict(boxstyle="round", fc="#90EE90"))
            return
        feat_name = feature_names[node['feature']]
        ax.text(x, y, f"{feat_name}\n< {node['threshold']:.2f}", ha='center', va='center', bbox=dict(boxstyle="round", fc="#ADD8E6"))
        ax.plot([x, x-dx], [y-0.02, y-dy+0.02], 'k-')
        recurse(node['left'], x-dx, y-dy, dx*0.5, dy)
        ax.plot([x, x+dx], [y-0.02, y-dy+0.02], 'k-')
        recurse(node['right'], x+dx, y-dy, dx*0.5, dy)

    recurse(tree)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = []
    
    if request.method == 'POST':
        try:
            # Ambil input form dan ubah ke float
            inputs = [
                float(request.form.get(name)) for name in 
                ['gender', 'age', 'occupation', 'sleep_duration', 'quality_sleep',
                 'phys_activity', 'stress', 'bmi', 'heart_rate', 'daily_steps', 
                 'systolic', 'diastolic']
            ]
            
            input_data = np.array([inputs])
            input_scaled = (input_data - X_min) / (X_max - X_min + 1e-8)
            
            pred = predict_forest(forest, input_scaled)
            res_str = "Gangguan Tidur Terdeteksi" if pred == 1 else "Tidur Normal"
            prediction_text = f"Hasil: {res_str}"
            
            # Gambar semua pohon
            for i, tree in enumerate(forest):
                tree_plots.append(get_tree_image(tree, feature_names, f"Pohon {i+1}"))
                
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)