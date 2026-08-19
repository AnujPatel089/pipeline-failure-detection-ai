"""Preprocessing and baseline model construction."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor(
    scale_numeric: bool,
    feature_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted transformer for approved numerical and categorical data."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    selected = set(feature_columns or (NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    numeric_features = [feature for feature in NUMERIC_FEATURES if feature in selected]
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature in selected]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(steps), numeric_features))
    if categorical_features:
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", categorical, categorical_features))
    return ColumnTransformer(transformers)


def build_random_forest_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build the baseline Random Forest for an explicitly approved feature subset."""
    return Pipeline([
        ("preprocessor", build_preprocessor(False, feature_columns)),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def build_model_pipelines() -> dict[str, Pipeline]:
    """Return fresh, end-to-end baseline pipelines."""
    return {
        "dummy": Pipeline([("preprocessor", build_preprocessor(False)), ("classifier", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))]),
        "logistic_regression": Pipeline([("preprocessor", build_preprocessor(True)), ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))]),
        "random_forest": build_random_forest_pipeline(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
    }
