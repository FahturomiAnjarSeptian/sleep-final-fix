import numpy as np
import pandas as pd
import pickle

print("=== MEMULAI TRAINING (DENGAN OVERSAMPLING) ===")

# 1. LOAD DATA
try:
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
except FileNotFoundError:
    print("ERROR: File csv tidak ditemukan!")
    exit()

# 2. PREPROCESSING
if 'Person ID' in df.columns: df = df.drop(columns=['Person ID'])

# Pecah Blood Pressure
try:
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df['systolic'] = bp_split[0]
        df['diastolic'] = bp_split[1]
    else:
        df['systolic'] = 120.0
        df['diastolic'] = 80.0
except:
    df['systolic'] = 120.0
    df['diastolic'] = 80.0

# Encoding
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).astype(float)
df['Occupation'] = df['Occupation'].astype('category').cat.codes.astype(float)
df['BMI Category'] = df['BMI Category'].astype('category').cat.codes.astype(float)
# Target: 0=Sehat, 1=Sakit
df['Sleep Disorder'] = df['Sleep Disorder'].replace({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 1}).fillna(0).astype(int)

# --- OVERSAMPLING (Teknik agar tidak selalu prediksi Normal) ---
# Kita duplikasi data orang yang sakit agar jumlahnya seimbang
data_sakit = df[df['Sleep Disorder'] == 1]
data_sehat = df[df['Sleep Disorder'] == 0]
# Gabungkan data sehat + data sakit (dikali 2 biar banyak)
df_balanced = pd.concat([data_sehat, data_sakit, data_sakit], axis=0).reset_index(drop=True)
print(f"Data asli: {len(df)}, Data setelah oversampling: {len(df_balanced)}")

# Urutan Fitur Baku
feature_order = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 
    'Daily Steps', 'systolic', 'diastolic'
]

for col in feature_order:
    if col not in df_balanced.columns: df_balanced[col] = 0.0

X = df_balanced[feature_order].values.astype(float)
y = df_balanced['Sleep Disorder'].values

# Scaling
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)

# 3. RANDOM FOREST MANUAL
def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def split_data(X, y, feature, threshold):
    mask = X[:, feature] < threshold
    return X[mask], y[mask], X[~mask], y[~mask]

def best_split(X, y):
    best_gini = 1
    best_feat, best_thr = None, None
    n_features = X.shape[1]
    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        if len(thresholds) > 30: # Optimasi
            thresholds = np.percentile(thresholds, np.linspace(0, 100, 30))
        for thr in thresholds:
            X_l, y_l, X_r, y_r = split_data(X, y, feat, thr)
            if len(y_l) == 0 or len(y_r) == 0: continue
            g = (len(y_l)*gini(y_l) + len(y_r)*gini(y_r)) / len(y)
            if g < best_gini:
                best_gini, best_feat, best_thr = g, feat, thr
    return best_feat, best_thr

def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1 or depth == max_depth or len(y) < 2:
        return {'label': np.round(np.mean(y))}
    feat, thr = best_split(X, y)
    if feat is None:
        return {'label': np.round(np.mean(y))}
    X_l, y_l, X_r, y_r = split_data(X, y, feat, thr)
    return {
        'feature': feat, 'threshold': thr,
        'left': build_tree(X_l, y_l, depth+1, max_depth),
        'right': build_tree(X_r, y_r, depth+1, max_depth)
    }

trees = []
print("Melatih 7 Pohon...", end="")
for i in range(7): # Naikkan jadi 7 pohon
    idx = np.random.choice(len(X), len(X), replace=True)
    trees.append(build_tree(X[idx], y[idx]))
    print(".", end="")

data_to_save = {
    'forest': trees, 'X_min': X_min, 'X_max': X_max, 'feature_names': feature_order
}
with open('model_sleep.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
print("\n=== SUKSES! Model disimpan ===")