"""
preprocessor.py
---------------
Data preprocessing pipeline for Network Intrusion Detection System.
Handles the KDD Cup 1999 / Sampadab17 Kaggle dataset.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

logger = logging.getLogger(__name__)

# ── Column definitions (KDD Cup 1999 41-feature schema) ──────────────────────
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
TARGET_COL = "label"

# Attack-type family mapping (binary + 5-class)
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L",
    "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}


class NIDSPreprocessor:
    """End-to-end preprocessing pipeline for the NIDS dataset."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.label_encoder_protocol = LabelEncoder()
        self.label_encoder_service = LabelEncoder()
        self.label_encoder_flag = LabelEncoder()
        self.label_encoder_target = LabelEncoder()
        self.scaler = StandardScaler()

        # Feature columns surviving after clean() — populated during run()
        self.feature_cols: list[str] = []

        self._is_fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV; auto-detect whether header row is present."""
        logger.info("Loading dataset from %s", filepath)
        df = pd.read_csv(filepath)

        # If the first row looks like column names, drop it
        if df.iloc[0, 0] == "duration" or str(df.iloc[0, 0]).isalpha():
            df = pd.read_csv(filepath, header=0)
        else:
            df.columns = COLUMN_NAMES

        # Strip trailing dot from label if present (KDD convention)
        df[TARGET_COL] = df[TARGET_COL].str.rstrip(".")
        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, handle infinities, and remove irrelevant cols."""
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Remove constant / near-zero-variance columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_var = [c for c in numeric_cols
                   if df[c].std() < 1e-6 and c != TARGET_COL]
        if low_var:
            logger.debug("Dropping low-variance cols: %s", low_var)
            df.drop(columns=low_var, inplace=True)

        logger.info("After cleaning: %d rows × %d columns", *df.shape)
        return df

    def add_attack_family(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append an 'attack_family' column (normal / DoS / Probe / R2L / U2R)."""
        df = df.copy()
        df["attack_family"] = df[TARGET_COL].map(ATTACK_MAP).fillna("Other")
        logger.info(
            "Attack family distribution:\n%s",
            df["attack_family"].value_counts()
        )
        return df

    def encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns and the target."""
        df = df.copy()

        for col, enc in zip(
            CATEGORICAL_COLS,
            [self.label_encoder_protocol,
             self.label_encoder_service,
             self.label_encoder_flag],
        ):
            if col not in df.columns:
                continue
            if fit:
                df[col] = enc.fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories gracefully
                known = set(enc.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known else enc.classes_[0]
                )
                df[col] = enc.transform(df[col].astype(str))

        # Encode the target
        if fit:
            df[TARGET_COL] = self.label_encoder_target.fit_transform(
                df[TARGET_COL]
            )
        else:
            known_labels = set(self.label_encoder_target.classes_)
            df[TARGET_COL] = df[TARGET_COL].apply(
                lambda x: x if x in known_labels
                else self.label_encoder_target.classes_[0]
            )
            df[TARGET_COL] = self.label_encoder_target.transform(df[TARGET_COL])

        return df

    def scale(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Standardise features."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def get_X_y(self, df: pd.DataFrame):
        """Split DataFrame into feature matrix and target vector."""
        drop_cols = [TARGET_COL]
        if "attack_family" in df.columns:
            drop_cols.append("attack_family")
        X = df.drop(columns=drop_cols).values
        y = df[TARGET_COL].values
        return X, y

    def split(
        self, X: np.ndarray, y: np.ndarray,
        test_size: float = 0.20, random_state: int = 42
    ):
        """Stratified train/test split."""
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    def balance(self, X_train: np.ndarray, y_train: np.ndarray):
        """Apply SMOTE to handle class imbalance (U2R / R2L are tiny)."""
        logger.info("Applying SMOTE to balance classes …")
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE: %d samples (was %d)", len(y_res), len(y_train)
        )
        return X_res, y_res

    def run(self, filepath: str, use_smote: bool = True):
        """Full pipeline: load → clean → encode → scale → split → (smote)."""
        df = self.load_data(filepath)
        df = self.clean(df)
        df = self.add_attack_family(df)
        df = self.encode(df, fit=True)

        # Record exactly which feature columns survived cleaning
        drop_cols = [TARGET_COL, "attack_family"]
        self.feature_cols = [c for c in df.columns if c not in drop_cols]

        X, y = self.get_X_y(df)
        X_scaled = self.scale(X, fit=True)

        X_train, X_test, y_train, y_test = self.split(X_scaled, y)

        if use_smote:
            X_train, y_train = self.balance(X_train, y_train)

        self._is_fitted = True
        self.save()

        logger.info(
            "Pipeline complete. Train: %d  Test: %d", len(y_train), len(y_test)
        )
        return X_train, X_test, y_train, y_test

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        """Persist encoders and scaler to disk."""
        joblib.dump(self.scaler,                  os.path.join(self.model_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder_protocol,  os.path.join(self.model_dir, "enc_protocol.pkl"))
        joblib.dump(self.label_encoder_service,   os.path.join(self.model_dir, "enc_service.pkl"))
        joblib.dump(self.label_encoder_flag,      os.path.join(self.model_dir, "enc_flag.pkl"))
        joblib.dump(self.label_encoder_target,    os.path.join(self.model_dir, "label_encoder.pkl"))
        joblib.dump(self.feature_cols,            os.path.join(self.model_dir, "feature_cols.pkl"))
        logger.info("Preprocessor artefacts saved to %s/", self.model_dir)

    @classmethod
    def load(cls, model_dir: str = "models") -> "NIDSPreprocessor":
        """Restore a previously fitted preprocessor from disk."""
        obj = cls(model_dir=model_dir)
        obj.scaler                   = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        obj.label_encoder_protocol   = joblib.load(os.path.join(model_dir, "enc_protocol.pkl"))
        obj.label_encoder_service    = joblib.load(os.path.join(model_dir, "enc_service.pkl"))
        obj.label_encoder_flag       = joblib.load(os.path.join(model_dir, "enc_flag.pkl"))
        obj.label_encoder_target     = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        # Load surviving feature list; fall back to deriving from scaler if missing
        fc_path = os.path.join(model_dir, "feature_cols.pkl")
        if os.path.exists(fc_path):
            obj.feature_cols = joblib.load(fc_path)
        obj._is_fitted = True
        logger.info("Preprocessor loaded from %s/", model_dir)
        return obj

    # ── Inference helper ──────────────────────────────────────────────────────

    def transform_single(self, record: dict) -> np.ndarray:
        """
        Transform a single JSON record (dict) into a scaled feature vector
        ready for model inference.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded first.")

        df = pd.DataFrame([record])

        # Fill every known column with zeros if absent
        for col in COLUMN_NAMES[:-1]:
            if col not in df.columns:
                df[col] = 0

        # Encode categorical columns
        for col, enc in zip(
            CATEGORICAL_COLS,
            [self.label_encoder_protocol,
             self.label_encoder_service,
             self.label_encoder_flag],
        ):
            known = set(enc.classes_)
            df[col] = df[col].apply(
                lambda x: x if str(x) in known else enc.classes_[0]
            )
            df[col] = enc.transform(df[col].astype(str))

        # Use the exact feature columns the scaler was fitted on.
        # Fall back to all 41-minus-label columns if feature_cols not saved yet.
        cols = self.feature_cols if self.feature_cols else [
            c for c in COLUMN_NAMES[:-1] if c in df.columns
        ]
        X = df[cols].values.astype(float)
        return self.scaler.transform(X)

    def decode_prediction(self, encoded_label: int) -> str:
        """Convert integer prediction back to the original attack label."""
        return self.label_encoder_target.inverse_transform([encoded_label])[0]
