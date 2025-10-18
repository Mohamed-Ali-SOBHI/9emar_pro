#!/usr/bin/env python3
"""
Realistic training pipeline with time-aware evaluation and richer model search.

This version replaces the previous single-split logistic baseline with a
time-based train/test split, cross-validation grouped by season, and a set of
competitive estimators (regularised logistic regression, random forest and
gradient boosting). The full preprocessing chain is embedded inside sklearn
pipelines so the persisted model can be reused directly at prediction time.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def find_dataset_path() -> Path:
    """Return the first available enhanced dataset path."""
    candidates = [
        Path("preprocessed_data_enhanced_no_uncertainty.csv"),
        Path("preprocessed_data_enhanced.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate an enhanced dataset. Expected one of: "
        "preprocessed_data_enhanced_no_uncertainty.csv or preprocessed_data_enhanced.csv"
    )


def load_enhanced_dataset():
    """Load data, enforce ordering, and keep only usable numeric features."""
    dataset_path = find_dataset_path()
    print(f"[1/6] Loading enhanced dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if "result" not in df.columns:
        raise ValueError("Enhanced dataset is missing the 'result' target column.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
            errors="coerce",
        )

    df["month"] = df["month"] if "month" in df.columns else df["date"].dt.month
    df["year"] = df["year"] if "year" in df.columns else df["date"].dt.year
    df["season"] = np.where(df["month"] >= 7, df["year"], df["year"] - 1)

    df = df[df["result"].notna()].copy()
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    exclude = {
        "result",
        "team_id",
        "opponent_id",
        "team_name",
        "opponent_name",
        "league",
        "date",
    }
    exclude.update({col for col in ["favorite", "favorite_wins"] if col in df.columns})

    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].copy()

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    feature_cols = X.columns.tolist()

    y = df["result"].astype(str).reset_index(drop=True)
    meta = df[["date", "season", "year", "month"]].reset_index(drop=True)

    print(
        f"[INFO] Samples: {len(df)}, usable numeric features: {len(feature_cols)}, "
        f"seasons: {meta['season'].nunique()}"
    )
    print("[INFO] Target distribution:")
    for label, count in y.value_counts().sort_index().items():
        print(f"   {label}: {count} ({count / len(y) * 100:.1f}%)")

    return X.reset_index(drop=True), y, meta, feature_cols


def time_based_train_test_split(X, y, meta, holdout_seasons=1):
    """Hold out the most recent seasons to mimic production usage."""
    seasons = meta["season"]
    unique_seasons = sorted(seasons.dropna().unique())

    if len(unique_seasons) < 2:
        split_index = int(len(X) * 0.8)
        print(
            "[WARN] Less than two seasons found. Falling back to chronological 80/20 split."
        )
        return (
            X.iloc[:split_index],
            X.iloc[split_index:],
            y.iloc[:split_index],
            y.iloc[split_index:],
            seasons.iloc[:split_index],
            seasons.iloc[split_index:],
            unique_seasons[-1:],
        )

    holdout = max(1, min(holdout_seasons, len(unique_seasons) - 1))
    test_seasons = unique_seasons[-holdout:]
    test_mask = seasons.isin(test_seasons)
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_seasons, test_seasons_series = seasons[train_mask], seasons[test_mask]

    if len(X_test) < len(X) * 0.1:
        split_index = int(len(X) * 0.85)
        print(
            "[WARN] Holdout season too small. "
            "Switching to chronological 85/15 split to keep enough test data."
        )
        return (
            X.iloc[:split_index],
            X.iloc[split_index:],
            y.iloc[:split_index],
            y.iloc[split_index:],
            seasons.iloc[:split_index],
            seasons.iloc[split_index:],
            unique_seasons[-1:],
        )

    print(
        f"[INFO] Train seasons: {sorted(train_seasons.unique())}, "
        f"Test seasons: {sorted(test_seasons)}"
    )
    print(
        f"[INFO] Train size: {len(X_train)} rows, test size: {len(X_test)} rows "
        f"({len(X_test) / len(X) * 100:.1f}% holdout)"
    )
    return X_train, X_test, y_train, y_test, train_seasons, test_seasons_series, test_seasons


def build_cv_strategy(train_seasons, y_train):
    """Create a cross-validation strategy that respects temporal grouping."""
    unique_groups = np.unique(train_seasons.dropna())

    if len(unique_groups) >= 3:
        n_splits = min(5, len(unique_groups))
        print(f"[INFO] Using GroupKFold with {n_splits} splits grouped by season.")
        return GroupKFold(n_splits=n_splits), train_seasons

    min_class = int(y_train.value_counts().min())
    n_splits = max(2, min(5, min_class))
    print(f"[INFO] Using StratifiedKFold with {n_splits} splits (insufficient seasons).")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None


def build_model_candidates(k_linear, k_tree):
    """Prepare model pipelines with consistent preprocessing."""
    k_linear = max(1, k_linear)
    k_tree = max(1, k_tree)

    def make_linear_selector():
        return SelectKBest(score_func=f_classif, k=k_linear)

    def make_tree_selector():
        return SelectKBest(score_func=f_classif, k=k_tree)

    def make_imputer():
        return SimpleImputer(strategy="median")

    models = {
        "LogReg_balanced_C1.5": Pipeline(
            [
                ("imputer", make_imputer()),
                ("scaler", StandardScaler()),
                ("selector", make_linear_selector()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.5,
                        solver="lbfgs",
                        max_iter=2000,
                        multi_class="ovr",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LogReg_balanced_C0.7": Pipeline(
            [
                ("imputer", make_imputer()),
                ("scaler", StandardScaler()),
                ("selector", make_linear_selector()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=0.7,
                        solver="lbfgs",
                        max_iter=2000,
                        multi_class="ovr",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LogReg_draw_prioritized": Pipeline(
            [
                ("imputer", make_imputer()),
                ("scaler", StandardScaler()),
                ("selector", make_linear_selector()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        multi_class="ovr",
                        class_weight={"w": 1.0, "d": 1.6, "l": 1.05},
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LogReg_balanced_C2.5": Pipeline(
            [
                ("imputer", make_imputer()),
                ("scaler", StandardScaler()),
                ("selector", make_linear_selector()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=2.5,
                        solver="lbfgs",
                        max_iter=2000,
                        multi_class="ovr",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest_balanced": Pipeline(
            [
                ("imputer", make_imputer()),
                ("selector", make_tree_selector()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=5,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", make_imputer()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=600,
                        max_depth=8,
                        min_samples_leaf=40,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting_Tuned": Pipeline(
            [
                ("imputer", make_imputer()),
                ("selector", make_tree_selector()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=900,
                        max_depth=6,
                        min_samples_leaf=25,
                        l2_regularization=0.2,
                        max_bins=255,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Stacked_LogReg_HGB": Pipeline(
            [
                ("imputer", make_imputer()),
                ("selector", make_tree_selector()),
                (
                    "clf",
                    StackingClassifier(
                        estimators=[
                            (
                                "logreg_pipeline",
                                make_pipeline(
                                    StandardScaler(),
                                    LogisticRegression(
                                        penalty="l2",
                                        C=0.9,
                                        solver="lbfgs",
                                        max_iter=3000,
                                        multi_class="ovr",
                                        class_weight="balanced",
                                        random_state=42,
                                    ),
                                ),
                            ),
                            (
                                "hist_gb",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.05,
                                    max_iter=700,
                                    max_depth=7,
                                    min_samples_leaf=30,
                                    random_state=42,
                                ),
                            ),
                        ],
                        final_estimator=LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            solver="lbfgs",
                            max_iter=2000,
                            multi_class="ovr",
                            class_weight="balanced",
                            random_state=42,
                        ),
                        stack_method="predict_proba",
                        passthrough=True,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def compute_draw_overprediction(y_true, y_pred):
    """Return the draw over-prediction count used as a penalty component."""
    pred_counts = pd.Series(y_pred).value_counts()
    actual_counts = pd.Series(y_true).value_counts()
    return pred_counts.get("d", 0) - actual_counts.get("d", 0)


def realistic_model_training(
    X_train, X_test, y_train, y_test, train_seasons, k_features
):
    """Train multiple models, score them, and keep the best performer."""
    print("[3/6] Model selection and training...")
    n_features = X_train.shape[1]
    k_linear = min(max(120, int(n_features * 0.75)), n_features)
    k_tree = min(max(80, int(n_features * 0.5)), n_features)
    candidates = build_model_candidates(k_linear, k_tree)

    cv_strategy, cv_groups = build_cv_strategy(train_seasons, y_train)

    results = {}
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, estimator in candidates.items():
        print(f"\n--- Training {name} ---")
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        draw_metrics = report.get("d", {})
        draw_precision = float(draw_metrics.get("precision", 0.0))
        draw_recall = float(draw_metrics.get("recall", 0.0))
        draw_f1 = float(draw_metrics.get("f1-score", 0.0))
        draw_over = compute_draw_overprediction(y_test, y_pred)
        draw_gap_ratio = abs(draw_over) / max(1, (y_test == "d").sum())
        overall_score = (test_acc * 0.65) + (test_f1_macro * 0.35) - (draw_gap_ratio * 0.05)

        try:
            cv_scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv_strategy,
                groups=cv_groups,
                scoring="accuracy",
                n_jobs=-1,
            )
        except Exception as exc:
            print(f"[WARN] CV failed for {name}: {exc}")
            cv_scores = np.array([])

        results[name] = {
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1_macro),
            "test_f1_weighted": float(test_f1_weighted),
            "draw_precision": draw_precision,
            "draw_recall": draw_recall,
            "draw_f1": draw_f1,
            "draw_overprediction": int(draw_over),
            "draw_gap_ratio": float(draw_gap_ratio),
            "overall_score": float(overall_score),
            "cv_scores": cv_scores,
            "cv_accuracy": float(cv_scores.mean()) if cv_scores.size else None,
            "cv_std": float(cv_scores.std()) if cv_scores.size else None,
            "classification_report": classification_report(
                y_test, y_pred, digits=4, zero_division=0
            ),
        }

        print(
            f"[RESULT] {name}: acc={test_acc:.4f}, macro_f1={test_f1_macro:.4f}, "
            f"draw_recall={draw_recall:.4f}, draw_over={draw_over:+d}"
        )
        if cv_scores.size:
            print(
                f"[CV] mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}, "
                f"splits={len(cv_scores)}"
            )

        if overall_score > best_score:
            best_score = overall_score
            best_name = name
            best_model = estimator

    if best_name is None:
        raise RuntimeError("No model could be trained successfully.")

    print(f"\n[BEST MODEL] {best_name} with composite score {best_score:.4f}")
    return best_model, best_name, results[best_name], results


def final_realistic_evaluation(model, X_test, y_test, model_name, test_seasons):
    """Print a detailed evaluation for the chosen model on the holdout set."""
    print("\n[4/6] Final evaluation on holdout seasons:", test_seasons)
    y_pred = model.predict(X_test)

    print("\nFinal classification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    actual_dist = y_test.value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index()

    print("\nRealism check (distribution gap):")
    print("Class    Actual   Predicted   Difference")
    for cls in ["d", "l", "w"]:
        actual_pct = actual_dist.get(cls, 0.0) * 100
        pred_pct = pred_dist.get(cls, 0.0) * 100
        diff = pred_pct - actual_pct
        print(f"  {cls}     {actual_pct:5.1f}%      {pred_pct:5.1f}%      {diff:+5.1f}%")


def save_model_artifacts(model, metrics, feature_cols, all_results):
    """Persist the trained pipeline and its metadata."""
    print("\n[6/6] Saving model artifacts...")
    models_dir = Path("models/optimal_model")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "trained_model_realistic.pkl")

    metrics_to_save = metrics.copy()
    cv_scores = metrics_to_save.pop("cv_scores", None)
    if cv_scores is not None:
        metrics_to_save["cv_scores"] = [float(score) for score in cv_scores]

    def to_serializable(value):
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, np.ndarray):
            return [to_serializable(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {k: to_serializable(v) for k, v in value.items()}
        return value

    metrics_serializable = {
        key: to_serializable(val) for key, val in metrics_to_save.items()
    }

    with open(models_dir / "metrics_realistic.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_serializable, fh, indent=2)

    with open(models_dir / "features_realistic.json", "w", encoding="utf-8") as fh:
        json.dump(sorted(feature_cols), fh, indent=2)

    search_results = {
        name: {
            k: to_serializable(v)
            for k, v in metrics.items()
            if k not in {"classification_report"}
        }
        for name, metrics in all_results.items()
    }

    with open(models_dir / "model_search_results.json", "w", encoding="utf-8") as fh:
        json.dump(search_results, fh, indent=2)


def main():
    """Main entrypoint for the enhanced realistic training pipeline."""
    print("=" * 80)
    print("REALISTIC TRAINING PIPELINE - TIME AWARE VERSION")
    print("=" * 80)

    X, y, meta, feature_cols = load_enhanced_dataset()

    X_train, X_test, y_train, y_test, train_seasons, test_seasons_series, test_seasons = (
        time_based_train_test_split(X, y, meta)
    )

    best_model, best_name, best_metrics, all_results = realistic_model_training(
        X_train, X_test, y_train, y_test, train_seasons, len(feature_cols)
    )

    best_metrics.update(
        {
            "model_type": best_name,
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_seasons": sorted(train_seasons.unique()),
            "test_seasons": test_seasons,
        }
    )

    final_realistic_evaluation(best_model, X_test, y_test, best_name, test_seasons)

    save_model_artifacts(best_model, best_metrics, feature_cols, all_results)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best model: {best_name}")
    print(f"Holdout seasons: {test_seasons}")
    print(f"Test accuracy: {best_metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {best_metrics['test_f1_macro']:.4f}")
    if best_metrics.get("cv_accuracy") is not None:
        print(
            f"CV accuracy: {best_metrics['cv_accuracy']:.4f} "
            f"(std {best_metrics['cv_std']:.4f})"
        )
    print(f"Draw recall: {best_metrics['draw_recall']:.4f}")
    print(f"Draw over-prediction: {best_metrics['draw_overprediction']:+d}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
