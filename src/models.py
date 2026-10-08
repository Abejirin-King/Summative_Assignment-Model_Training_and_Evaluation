from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def make_preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == "object"]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                          ("scaler",StandardScaler())]), num),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), cat)
    ])

def traditional_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000,class_weight="balanced",C=0.1),
        "RandomForest": RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=2,
                                               class_weight="balanced",random_state=42,n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.1,max_leaf_nodes=15,l2_regularization=1,random_state=42)
    }

def make_pipeline(X, model):
    return Pipeline([("preprocessor",make_preprocessor(X)),("model",model)])
