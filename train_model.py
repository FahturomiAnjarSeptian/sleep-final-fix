import numpy as np
import pandas as pd
import pickle

# 1. LOAD DATA (Pastikan file csv ada di sebelah file ini)
try:
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'Sleep_health_and_lifestyle_dataset.csv' tidak ditemukan.")
    exit()

# 2. PREPROCESSING SEDERHANA
if 'Person ID' in df.columns: df = df.drop(columns=['Person ID'])

# Blood Pressure split
try:
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df['systolic'] = bp_split[0]
        df['diastolic'] = bp_split[1]
        df = df.drop(columns=['Blood Pressure'])
    else:
        df['systolic'] = 120; df['diastolic'] = 80
except:
    pass

# Encoding
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df['Occupation'] = df['Occupation'].astype('category').cat.codes
df['BMI Category'] = df['BMI Category'].astype('category').cat.codes
df['Sleep Disorder'] = df['Sleep Disorder'].replace({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 1}).fillna(0).astype(int)

# Fitur Baku (12 Kolom)
feature_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
                'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 
                'Daily Steps', 'systolic', 'diastolic']

# Pastikan kolom lengkap
for col in feature_cols:
    if col not in df.columns: df[col] = 0

X = df[feature_cols].values.astype(float)
y = df['Sleep Disorder'].values

# Scaling
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)

# 3. RANDOM FOREST MANUAL (Versi Standar)
def split_data(X, y, feature, threshold):
    mask = X[:, feature] < threshold
    return X[mask], y[mask], X[~mask], y[~mask]

def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def best_split(X, y):
    best_gini = 1
    best_feat, best_thr = None, None
    n_features = X.shape[1]
    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        # Optimasi sederhana
        if len(thresholds) > 20: thresholds = np.percentile(thresholds, np.linspace(0, 100, 20))
        for thr in thresholds:
            X_l, y_l, X_r, y_r = split_data(X, y, feat, thr)
            if len(y_l) == 0 or len(y_r) == 0: continue
            g = (len(y_l)*gini(y_l) + len(y_r)*gini(y_r)) / len(y)
            if g < best_gini: best_gini, best_feat, best_thr = g, feat, thr
    return best_feat, best_thr

def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth or len(y) < 2: return {'label': np.round(np.mean(y))}
    feat, thr = best_split(X, y)
    if feat is None: return {'label': np.round(np.mean(y))}
    X_l, y_l, X_r, y_r = split_data(X, y, feat, thr)
    return {'feature': feat, 'threshold': thr, 'left': build_tree(X_l, y_l, depth+1, max_depth), 'right': build_tree(X_r, y_r, depth+1, max_depth)}

trees = []
print("Melatih model...", end="")
for i in range(5): # 5 Pohon cukup
    idx = np.random.choice(len(X), len(X), replace=True)
    trees.append(build_tree(X[idx], y[idx]))
    print(".", end="")

# 4. SIMPAN (Simple Pickle)
data_to_save = {'forest': trees, 'X_min': X_min, 'X_max': X_max, 'feature_names': feature_cols}
with open('model_sleep.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("\nSUKSES! Model disimpan sebagai 'model_sleep.pkl'")