import numpy as np
import pandas as pd
import pickle

# 1. LOAD DATA
try:
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
except FileNotFoundError:
    print("Error: Dataset tidak ditemukan.")
    exit()

# 2. PREPROCESSING
if 'Person ID' in df.columns: df = df.drop(columns=['Person ID'])
if 'Blood Pressure' in df.columns:
    try:
        bp_split = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df['Systolic BP'] = bp_split[0]
        df['Diastolic BP'] = bp_split[1]
        df = df.drop(columns=['Blood Pressure'])
    except: pass

df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df['Occupation'] = df['Occupation'].astype('category').cat.codes
df['BMI Category'] = df['BMI Category'].astype('category').cat.codes
if 'Age Group' in df.columns: df['Age Group'] = df['Age Group'].astype('category').cat.codes
df['Sleep Disorder'] = df['Sleep Disorder'].replace({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 1}).fillna(0).astype(int)

numeric_cols = [c for c in df.columns if c not in ['Sleep Disorder'] and np.issubdtype(df[c].dtype, np.number)]
X = df[numeric_cols].values.astype(float)
y = df['Sleep Disorder'].values

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)

# 3. RANDOM FOREST MANUAL
def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def split_data(X, y, feature, threshold):
    left_mask = X[:, feature] < threshold
    return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

def best_split(X, y):
    best_gini = 1
    best_feat, best_thr = None, None
    n_features = X.shape[1]
    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        for thr in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feat, thr)
            if len(y_left) == 0 or len(y_right) == 0: continue
            g = (len(y_left)*gini(y_left) + len(y_right)*gini(y_right)) / len(y)
            if g < best_gini:
                best_gini, best_feat, best_thr = g, feat, thr
    return best_feat, best_thr

def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth or len(y) == 0:
        return {'label': np.round(np.mean(y)) if len(y) > 0 else 0}
    feat, thr = best_split(X, y)
    if feat is None:
        return {'label': np.round(np.mean(y))}
    X_left, y_left, X_right, y_right = split_data(X, y, feat, thr)
    return {
        'feature': feat, 'threshold': thr,
        'left': build_tree(X_left, y_left, depth+1, max_depth),
        'right': build_tree(X_right, y_right, depth+1, max_depth)
    }

def random_forest(X, y, n_trees=5, max_depth=3):
    trees = []
    print("Sedang melatih Trees", end="")
    for _ in range(n_trees):
        idx = np.random.choice(len(X), len(X), replace=True)
        trees.append(build_tree(X[idx], y[idx], depth=0, max_depth=max_depth))
        print(".", end="")
    print(" Selesai.")
    return trees

# Eksekusi Training
print("Mulai Training...")
forest_model = random_forest(X_scaled, y, n_trees=5, max_depth=3)

# 4. SIMPAN MODEL
data_to_save = {
    'forest': forest_model,
    'X_min': X_min,
    'X_max': X_max,
    'feature_names': numeric_cols
}

with open('model_sleep.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
print("SUKSES! File 'model_sleep.pkl' berhasil dibuat.")