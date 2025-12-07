import numpy as np
import pandas as pd
import pickle
import os

# Setup Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Sleep_health_and_lifestyle_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model_sleep.pkl')

print("=== MULAI TRAINING MODEL CERDAS ===")

# 1. LOAD DATA
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("Error: Dataset tidak ditemukan.")
    exit()

# 2. PREPROCESSING
if 'Person ID' in df.columns: df = df.drop(columns=['Person ID'])

# Blood Pressure Split
try:
    if 'Blood Pressure' in df.columns:
        bp = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df['systolic'] = bp[0]
        df['diastolic'] = bp[1]
    else:
        df['systolic'] = 120.0; df['diastolic'] = 80.0
except:
    df['systolic'] = 120.0; df['diastolic'] = 80.0

# Encoding
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).astype(float)
df['Occupation'] = df['Occupation'].astype('category').cat.codes.astype(float)
df['BMI Category'] = df['BMI Category'].astype('category').cat.codes.astype(float)
df['Sleep Disorder'] = df['Sleep Disorder'].replace({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 1}).fillna(0).astype(int)

# --- TEKNIK OVERSAMPLING (PENTING!) ---
# Agar model tidak bias ke "Normal", kita perbanyak data "Sakit"
data_normal = df[df['Sleep Disorder'] == 0]
data_sakit = df[df['Sleep Disorder'] == 1]

# Kita copy data sakit sebanyak 2x lipat agar seimbang
df_balanced = pd.concat([data_normal, data_sakit, data_sakit], axis=0).reset_index(drop=True)

print(f"Data Awal: {len(df)} baris")
print(f"Data Setelah Balancing: {len(df_balanced)} baris (Lebih peka penyakit)")

# Urutan Fitur Baku
feature_order = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 
    'Daily Steps', 'systolic', 'diastolic'
]

# Isi kolom yang hilang dengan 0
for col in feature_order:
    if col not in df_balanced.columns: df_balanced[col] = 0.0

X = df_balanced[feature_order].values.astype(float)
y = df_balanced['Sleep Disorder'].values

# Scaling
X_min = X.min(axis=0)
X_max = X.max(axis=0)

# 3. RANDOM FOREST MANUAL
def split_data(X, y, feat, thr):
    mask = X[:, feat] < thr
    return X[mask], y[mask], X[~mask], y[~mask]

def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def best_split(X, y):
    best_gini = 1
    best_feat, best_thr = None, None
    n_feats = X.shape[1]
    for f in range(n_feats):
        thrs = np.unique(X[:, f])
        if len(thrs) > 20: thrs = np.percentile(thrs, np.linspace(0, 100, 20)) # Optimasi
        for t in thrs:
            xl, yl, xr, yr = split_data(X, y, f, t)
            if len(yl) == 0 or len(yr) == 0: continue
            g = (len(yl)*gini(yl) + len(yr)*gini(yr)) / len(y)
            if g < best_gini: best_gini, best_feat, best_thr = g, f, t
    return best_feat, best_thr

def build_tree(X, y, depth=0, max_depth=5): # Depth dinaikkan jadi 5 biar lebih detail
    if len(set(y)) == 1 or depth == max_depth or len(y) < 2:
        return {'label': np.round(np.mean(y))}
    feat, thr = best_split(X, y)
    if feat is None: return {'label': np.round(np.mean(y))}
    xl, yl, xr, yr = split_data(X, y, feat, thr)
    return {'feature': feat, 'threshold': thr, 'left': build_tree(xl, yl, depth+1, max_depth), 'right': build_tree(xr, yr, depth+1, max_depth)}

print("Sedang melatih 7 pohon...", end="")
forest = []
for i in range(7): # 7 Pohon
    idx = np.random.choice(len(X), len(X), replace=True)
    forest.append(build_tree(X[idx], y[idx]))
    print(".", end="")

# 4. SIMPAN
data = {'forest': forest, 'X_min': X_min, 'X_max': X_max, 'feature_names': feature_order}
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(data, f)

print(f"\nSUKSES! Model disimpan di: {MODEL_PATH}")