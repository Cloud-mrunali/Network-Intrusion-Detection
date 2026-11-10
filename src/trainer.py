"""
trainer.py
----------
Model training, evaluation, and persistence for the NIDS.
Trains a Random Forest + XGBoost ensemble and saves the best model.
"""

import os
import json
import logging
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class NIDSTrainer:
    """Train and evaluate multiple classifiers; persist the best one."""

    def __init__(self, model_dir: str = "models", n_classes: int = None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.n_classes = n_classes
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    # ── Model definitions ─────────────────────────────────────────────────────

    def _build_models(self) -> dict:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        )

        dt = DecisionTreeClassifier(
            max_depth=20,
            class_weight="balanced",
            random_state=42
        )

        # Soft-voting ensemble of RF + XGB
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
            n_jobs=-1
        )

        return {
            "RandomForest": rf,
            "XGBoost": xgb,
            "DecisionTree": dt,
            "Ensemble_RF_XGB": ensemble,
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray,  y_test: np.ndarray,
        label_names: list = None,
    ):
        """
        Train all models, evaluate on the test set, and keep the best one
        (highest macro F1-score).
        """
        models = self._build_models()
        best_f1 = -1.0

        for name, model in models.items():
            logger.info("Training %s …", name)
            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test, name, label_names)
            self.metrics[name] = metrics

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                self.best_model = model
                self.best_model_name = name

        logger.info(
            "Best model: %s  (macro F1 = %.4f)", self.best_model_name, best_f1
        )
        self._save_best()
        self._save_metrics()
        return self.best_model, self.metrics

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(
        self, model, X_test, y_test, name: str, label_names=None
    ) -> dict:
        y_pred = model.predict(X_test)
        acc   = accuracy_score(y_test, y_pred)
        mac_f1 = f1_score(y_test, y_pred, average="macro",  zero_division=0)
        mic_f1 = f1_score(y_test, y_pred, average="micro",  zero_division=0)
        wgt_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC (one-vs-rest, only for proba-capable models)
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                )
            except Exception:
                pass

        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "model": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(mac_f1, 4),
            "micro_f1": round(mic_f1, 4),
            "weighted_f1": round(wgt_f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        logger.info(
            "%s → Acc=%.4f | MacroF1=%.4f | ROC-AUC=%s",
            name, acc, mac_f1,
            f"{roc_auc:.4f}" if roc_auc else "N/A"
        )
        return metrics

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self, feature_names: list) -> dict:
        """Return top-20 feature importances from the best model (if supported)."""
        model = self.best_model
        # Unwrap VotingClassifier → use first estimator
        if hasattr(model, "estimators_"):
            model = model.estimators_[0]

        if not hasattr(model, "feature_importances_"):
            return {}

        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        return {
            feature_names[i]: round(float(importances[i]), 6)
            for i in idx
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_best(self):
        path = os.path.join(self.model_dir, "nids_model.pkl")
        joblib.dump(self.best_model, path)
        logger.info("Best model saved → %s", path)

    def _save_metrics(self):
        path = os.path.join(self.model_dir, "metrics.json")
        # confusion_matrix is already a list; classification_report is a dict
        with open(path, "w") as fh:
            json.dump(self.metrics, fh, indent=2)
        logger.info("Metrics saved → %s", path)

    @staticmethod
    def load_model(model_dir: str = "models"):
        path = os.path.join(model_dir, "nids_model.pkl")
        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model

    # ── Inference ─────────────────────────────────────────────────────────────

    @staticmethod
    def predict(model, X: np.ndarray):
        """Return (predicted_class, probabilities) tuple."""
        pred = model.predict(X)
        proba = (
            model.predict_proba(X)
            if hasattr(model, "predict_proba") else None
        )
        return pred, proba
