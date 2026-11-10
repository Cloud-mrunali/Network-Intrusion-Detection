"""
train.py
--------
CLI entry-point to preprocess data and train the NIDS model.

Usage
-----
    python train.py --data data/Train_data.csv
    python train.py --data data/Train_data.csv --no-smote
    python train.py --data data/Train_data.csv --upload-cos   # also push to IBM COS
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessor import NIDSPreprocessor
from trainer import NIDSTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train the NIDS machine learning model."
    )
    parser.add_argument(
        "--data", default="data/Train_data.csv",
        help="Path to the training CSV file (default: data/Train_data.csv)"
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Skip SMOTE oversampling (faster but imbalanced)"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save model artefacts (default: models/)"
    )
    parser.add_argument(
        "--upload-cos", action="store_true",
        help="Upload trained model to IBM Cloud Object Storage after training"
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  NIDS Training Pipeline")
    logger.info("=" * 60)
    logger.info("Dataset : %s", args.data)
    logger.info("SMOTE   : %s", not args.no_smote)

    prep = NIDSPreprocessor(model_dir=args.model_dir)
    X_train, X_test, y_train, y_test = prep.run(
        filepath=args.data,
        use_smote=not args.no_smote,
    )

    label_names = list(prep.label_encoder_target.classes_)
    n_classes   = len(label_names)
    logger.info("Classes (%d): %s", n_classes, label_names)

    # ── 2. Training ──────────────────────────────────────────────────────────
    trainer = NIDSTrainer(model_dir=args.model_dir, n_classes=n_classes)
    best_model, metrics = trainer.train_all(
        X_train, y_train, X_test, y_test,
        label_names=label_names,
    )

    # ── 3. Feature importance ────────────────────────────────────────────────
    from preprocessor import COLUMN_NAMES, CATEGORICAL_COLS
    feature_names = [c for c in COLUMN_NAMES[:-1]]
    importances = trainer.get_feature_importance(feature_names)
    if importances:
        logger.info("Top-10 features:")
        for feat, imp in list(importances.items())[:10]:
            logger.info("  %-35s %.6f", feat, imp)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    best = metrics[trainer.best_model_name]
    logger.info("=" * 60)
    logger.info("Best model  : %s", trainer.best_model_name)
    logger.info("Accuracy    : %.4f", best["accuracy"])
    logger.info("Macro F1    : %.4f", best["macro_f1"])
    logger.info("Weighted F1 : %.4f", best["weighted_f1"])
    if best.get("roc_auc"):
        logger.info("ROC-AUC     : %.4f", best["roc_auc"])
    logger.info("=" * 60)

    # ── 5. Optional COS upload ────────────────────────────────────────────────
    if args.upload_cos:
        from ibm_watson import IBMObjectStorage
        cos = IBMObjectStorage()
        for fname in ["nids_model.pkl", "scaler.pkl", "label_encoder.pkl",
                      "enc_protocol.pkl", "enc_service.pkl", "enc_flag.pkl",
                      "metrics.json"]:
            fpath = os.path.join(args.model_dir, fname)
            if os.path.exists(fpath):
                cos.upload(fpath, f"models/{fname}")
        logger.info("Artefacts uploaded to IBM Cloud Object Storage.")

    logger.info("Training complete. Artefacts saved to %s/", args.model_dir)


if __name__ == "__main__":
    main()
