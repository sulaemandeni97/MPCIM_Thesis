"""
Reusable prediction utilities shared by Streamlit pages and the API layer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd


MODEL_REGISTRY = {
    "dual_logistic": {
        "label": "Logistic Regression (Dual-Dimensional)",
        "path": "results/baseline_models/dual_dimensional_model.pkl",
        "type": "Baseline",
    },
    "random_forest": {
        "label": "Random Forest Classifier",
        "path": "results/advanced_models/random_forest_model.pkl",
        "type": "Advanced",
    },
    "xgboost": {
        "label": "XGBoost Classifier",
        "path": "results/advanced_models/xgboost_model.pkl",
        "type": "Advanced",
    },
    "neural_network": {
        "label": "Neural Network (MLP)",
        "path": "results/advanced_models/neural_network_model.pkl",
        "type": "Advanced",
    },
}


@dataclass
class PredictionResult:
    prediction: int
    probability: float
    label: str
    derived_signals: Dict[str, Any]
    model_key: str


class PredictionService:
    """Load model artifacts once and provide helper methods for inference."""

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.repo_root / "data"
        self.results_dir = self.repo_root / "results"
        self.feature_columns = self._load_feature_columns()
        self.scaler = self._load_scaler()
        self.reference_assets = self._load_reference_assets()
        self.available_models = self._load_available_models()

    def _load_scaler(self):
        path = self.data_dir / "processed" / "scaler.pkl"
        if not path.exists():
            return None
        return joblib.load(path)

    def _load_feature_columns(self):
        path = self.data_dir / "processed" / "X_train.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path, nrows=0)
        return df.columns.tolist()

    def _load_reference_assets(self):
        processed_path = self.data_dir / "processed" / "full_dataset_processed.csv"
        final_path = self.data_dir / "final" / "integrated_performance_behavioral.csv"

        processed_df = pd.read_csv(processed_path) if processed_path.exists() else None
        final_df = pd.read_csv(final_path) if final_path.exists() else None

        category_options = {}
        category_mappings = {}
        category_defaults = {}

        categorical_columns = [
            "gender",
            "marital_status",
            "is_permanent",
            "performance_rating",
            "tenure_category",
            "performance_level",
            "behavioral_level",
        ]

        if processed_df is not None:
            for col in categorical_columns:
                enc_col = f"{col}_encoded"
                if col in processed_df.columns and enc_col in processed_df.columns:
                    category_options[col] = sorted(processed_df[col].dropna().unique().tolist())
                    mapping = (
                        processed_df[[col, enc_col]]
                        .drop_duplicates()
                        .set_index(col)[enc_col]
                        .to_dict()
                    )
                    category_mappings[col] = mapping
                    category_defaults[col] = mapping.get("unknown", next(iter(mapping.values()), 0))

        benchmark = None
        slider_defaults = None
        if final_df is not None:
            cols = ["performance_score", "behavior_avg", "tenure_years"]
            promoted = final_df[final_df["has_promotion"] == 1][cols].mean()
            not_promoted = final_df[final_df["has_promotion"] == 0][cols].mean()
            overall = final_df[cols].mean()
            benchmark = {"promoted": promoted, "not_promoted": not_promoted, "overall": overall}
            slider_defaults = final_df[cols].mean()

        return {
            "category_options": category_options,
            "category_mappings": category_mappings,
            "category_defaults": category_defaults,
            "benchmark": benchmark,
            "slider_defaults": slider_defaults,
            "processed_df": processed_df,
        }

    def _load_available_models(self):
        available = {}
        for key, config in MODEL_REGISTRY.items():
            model_path = self.repo_root / config["path"]
            if model_path.exists():
                available[key] = {**config, "abs_path": model_path}
        return available

    def get_benchmark(self):
        return self.reference_assets.get("benchmark")

    def list_models(self):
        return self.available_models

    def load_model(self, model_key: str):
        if model_key not in self.available_models:
            raise FileNotFoundError(f"Model '{model_key}' tidak ditemukan.")
        return joblib.load(self.available_models[model_key]["abs_path"])

    @staticmethod
    def categorize_performance(score: float) -> str:
        if score < 70:
            return "low"
        if score < 90:
            return "medium"
        return "high"

    @staticmethod
    def categorize_behavioral(score: float) -> str:
        if score < 85:
            return "low"
        if score < 95:
            return "medium"
        return "high"

    @staticmethod
    def categorize_tenure(years: float) -> str:
        if years <= 2:
            return "junior"
        if years <= 7:
            return "mid"
        return "senior"

    def encode_category(self, value: str, column: str) -> int:
        mapping = self.reference_assets["category_mappings"].get(column, {})
        defaults = self.reference_assets["category_defaults"]

        if value in mapping:
            return mapping[value]

        normalized = str(value).strip().lower()
        if normalized in mapping:
            return mapping[normalized]
        return defaults.get(column, 0)

    def prepare_features(self, inputs: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        perf_score = float(inputs["performance_score"])
        beh_score = float(inputs["behavior_avg"])
        tenure = float(inputs["tenure_years"])

        tenure_category = self.categorize_tenure(tenure)
        performance_level = self.categorize_performance(perf_score)
        behavioral_level = self.categorize_behavioral(beh_score)

        # Quick Assessment features (with defaults if not provided)
        psychological_score = float(inputs.get("psychological_score", 23.5))  # Mean from QA data
        drive_score = float(inputs.get("drive_score", 43.0))
        mental_strength_score = float(inputs.get("mental_strength_score", 43.1))
        adaptability_score = float(inputs.get("adaptability_score", 43.2))
        collaboration_score = float(inputs.get("collaboration_score", 7.4))
        has_quick_assessment = int(inputs.get("has_quick_assessment", 0))

        # Derived QA features
        holistic_score = (perf_score * 0.4 + beh_score * 0.3 + psychological_score * 0.3)
        score_alignment = 1 - (np.std([perf_score, beh_score, psychological_score]) / 100)
        leadership_potential = (drive_score * 0.4 + mental_strength_score * 0.3 + collaboration_score * 0.3)

        engineered = {
            "tenure_years": tenure,
            "performance_score": perf_score,
            "behavior_avg": beh_score,
            "perf_beh_ratio": perf_score / beh_score if beh_score else 0.0,
            "combined_score": (perf_score + beh_score) / 2,
            "score_difference": perf_score - beh_score,
            "high_performer": int(performance_level == "high" and behavioral_level == "high"),
            # Quick Assessment features
            "psychological_score": psychological_score,
            "drive_score": drive_score,
            "mental_strength_score": mental_strength_score,
            "adaptability_score": adaptability_score,
            "collaboration_score": collaboration_score,
            "has_quick_assessment": has_quick_assessment,
            "holistic_score": holistic_score,
            "score_alignment": score_alignment,
            "leadership_potential": leadership_potential,
        }

        categorical_values = {
            "gender": inputs["gender"],
            "marital_status": inputs["marital_status"],
            "is_permanent": inputs["is_permanent"],
            "performance_rating": inputs["performance_rating"],
            "tenure_category": tenure_category,
            "performance_level": performance_level,
            "behavioral_level": behavioral_level,
        }

        for col, value in categorical_values.items():
            engineered[f"{col}_encoded"] = self.encode_category(value, col)

        feature_df = pd.DataFrame([engineered])
        missing = set(self.feature_columns) - set(feature_df.columns)
        if missing:
            raise ValueError(f"Fitur tidak lengkap: {', '.join(sorted(missing))}")

        return feature_df[self.feature_columns], {
            "tenure_category": tenure_category,
            "performance_level": performance_level,
            "behavioral_level": behavioral_level,
            "perf_beh_ratio": engineered["perf_beh_ratio"],
            "combined_score": engineered["combined_score"],
            "score_difference": engineered["score_difference"],
            # Quick Assessment signals
            "psychological_score": psychological_score,
            "drive_score": drive_score,
            "mental_strength_score": mental_strength_score,
            "adaptability_score": adaptability_score,
            "collaboration_score": collaboration_score,
            "holistic_score": holistic_score,
            "score_alignment": score_alignment,
            "leadership_potential": leadership_potential,
            "has_quick_assessment": has_quick_assessment,
        }

    @staticmethod
    def _predict(model, scaled_features: np.ndarray) -> Tuple[int, float]:
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(scaled_features)[0][1])
        else:
            logits = float(model.decision_function(scaled_features)[0])
            probability = 1 / (1 + np.exp(-logits))
        prediction = int(probability >= 0.5)
        return prediction, probability

    def predict_single(self, payload: Dict[str, Any], model_key: str) -> PredictionResult:
        if self.scaler is None or not self.feature_columns:
            raise RuntimeError("Feature engineering artifacts belum tersedia.")

        if model_key not in self.available_models:
            raise FileNotFoundError(f"Model '{model_key}' tidak tersedia.")

        feature_df, derived = self.prepare_features(payload)
        scaled = self.scaler.transform(feature_df)
        model = self.load_model(model_key)
        prediction, probability = self._predict(model, scaled)
        label = "PROMOTED" if prediction == 1 else "NOT_PROMOTED"

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            label=label,
            derived_signals=derived,
            model_key=model_key,
        )

    def batch_predict(self, df: pd.DataFrame, model_key: str) -> pd.DataFrame:
        outputs = []
        for _, row in df.iterrows():
            record = row.to_dict()
            features, _ = self.prepare_features(record)
            scaled = self.scaler.transform(features)
            model = self.load_model(model_key)
            prediction, probability = self._predict(model, scaled)
            record["prediction"] = prediction
            record["probability"] = probability
            outputs.append(record)
        return pd.DataFrame(outputs)


def load_service() -> PredictionService:
    """Lazy loader for FastAPI / Streamlit usage."""
    return PredictionService()
