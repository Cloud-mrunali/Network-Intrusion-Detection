"""
ibm_watson.py
-------------
IBM Cloud Lite Services integration for NIDS:
  - Watson Machine Learning (WML) — deploy & score the trained model
  - IBM Cloud Object Storage (COS) — upload/download dataset and model artefacts
  - IBM Cloudant — log detection events to a NoSQL database
"""

import os
import io
import json
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Optional heavy imports (only fail at call-time, not import-time) ─────────
try:
    from ibm_watson_machine_learning import APIClient as WMLClient
    _WML_AVAILABLE = True
except ImportError:
    _WMLClient_available = False
    _WML_AVAILABLE = False

try:
    import ibm_boto3
    from ibm_botocore.client import Config
    _COS_AVAILABLE = True
except ImportError:
    _COS_AVAILABLE = False

try:
    from ibmcloudant import CouchDbSessionAuthenticator
    from ibmcloudant.cloudant_v1 import CloudantV1
    _CLOUDANT_AVAILABLE = True
except ImportError:
    _CLOUDANT_AVAILABLE = False


# =============================================================================
# Watson Machine Learning
# =============================================================================

class WatsonMLService:
    """
    Wraps IBM Watson Machine Learning (WML) Lite-tier operations.

    Workflow
    --------
    1. `connect()`        – authenticate to WML
    2. `store_model()`    – upload trained scikit-learn model to WML repository
    3. `deploy()`         – create an online deployment
    4. `predict()`        – call the deployment scoring endpoint
    5. `delete_deploy()`  – clean up (Lite has a 5-deployment cap)
    """

    def __init__(self):
        self.wml_url         = os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
        self.api_key         = os.getenv("IBM_CLOUD_API_KEY")
        self.space_id        = os.getenv("IBM_WML_SPACE_ID")
        self.deployment_id   = os.getenv("IBM_WML_DEPLOYMENT_ID")
        self.client          = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Authenticate and return True on success."""
        if not _WML_AVAILABLE:
            logger.error("ibm-watson-machine-learning package not installed.")
            return False
        if not self.api_key:
            logger.error("IBM_CLOUD_API_KEY is not set in .env")
            return False

        wml_credentials = {
            "url":    self.wml_url,
            "apikey": self.api_key,
        }
        try:
            self.client = WMLClient(wml_credentials)
            if self.space_id:
                self.client.set.default_space(self.space_id)
            logger.info("Connected to IBM WML at %s", self.wml_url)
            return True
        except Exception as exc:
            logger.error("WML connection failed: %s", exc)
            return False

    # ── Store model ───────────────────────────────────────────────────────────

    def store_model(
        self,
        model_path: str,
        model_name: str = "NIDS-RandomForest",
        software_spec: str = "runtime-23.1-py3.10",
    ) -> str | None:
        """
        Upload a local .pkl model file to the WML repository.
        Returns the stored model UID, or None on failure.
        """
        if not self.client:
            logger.error("Call connect() first.")
            return None

        meta = {
            self.client.repository.ModelMetaNames.NAME: model_name,
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.client.software_specifications.get_uid_by_name(software_spec),
        }
        try:
            import joblib, pickle
            model_obj = joblib.load(model_path)
            stored = self.client.repository.store_model(
                model=model_obj, meta_props=meta
            )
            uid = self.client.repository.get_model_id(stored)
            logger.info("Model stored in WML repository. UID: %s", uid)
            return uid
        except Exception as exc:
            logger.error("Failed to store model: %s", exc)
            return None

    # ── Deploy ────────────────────────────────────────────────────────────────

    def deploy(self, model_uid: str, deployment_name: str = "NIDS-Online") -> str | None:
        """Create an online deployment; returns deployment UID."""
        if not self.client:
            logger.error("Call connect() first.")
            return None

        meta = {
            self.client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            self.client.deployments.ConfigurationMetaNames.ONLINE: {},
        }
        try:
            deploy_details = self.client.deployments.create(
                artifact_uid=model_uid, meta_props=meta
            )
            dep_uid = self.client.deployments.get_uid(deploy_details)
            logger.info("Deployment created. UID: %s", dep_uid)
            return dep_uid
        except Exception as exc:
            logger.error("Deployment failed: %s", exc)
            return None

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, feature_vectors: list) -> dict | None:
        """
        Score a list of feature vectors against the deployed WML endpoint.

        Parameters
        ----------
        feature_vectors : list of list[float]  – shape (n_samples, n_features)

        Returns
        -------
        dict with 'predictions' list on success, None on failure.
        """
        dep_id = self.deployment_id
        if not dep_id:
            logger.error("IBM_WML_DEPLOYMENT_ID not set in .env")
            return None
        if not self.client:
            if not self.connect():
                return None

        payload = {
            "input_data": [{"values": feature_vectors}]
        }
        try:
            result = self.client.deployments.score(dep_id, payload)
            logger.info("WML scoring succeeded.")
            return result
        except Exception as exc:
            logger.error("WML scoring failed: %s", exc)
            return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def delete_deploy(self, deployment_uid: str) -> bool:
        if not self.client:
            return False
        try:
            self.client.deployments.delete(deployment_uid)
            logger.info("Deployment %s deleted.", deployment_uid)
            return True
        except Exception as exc:
            logger.error("Delete deployment failed: %s", exc)
            return False


# =============================================================================
# IBM Cloud Object Storage
# =============================================================================

class IBMObjectStorage:
    """
    Upload / download files to IBM COS (Lite plan).
    Useful for sharing the dataset and model artefacts across environments.
    """

    def __init__(self):
        self.api_key       = os.getenv("IBM_COS_API_KEY")
        self.instance_crn  = os.getenv("IBM_COS_INSTANCE_CRN")
        self.endpoint      = os.getenv("IBM_COS_ENDPOINT",
                                        "https://s3.us-south.cloud-object-storage.appdomain.cloud")
        self.auth_endpoint = os.getenv("IBM_COS_AUTH_ENDPOINT",
                                        "https://iam.cloud.ibm.com/oidc/token")
        self.bucket        = os.getenv("IBM_COS_BUCKET_NAME", "nids-dataset-bucket")
        self._client       = None

    def _get_client(self):
        if self._client:
            return self._client
        if not _COS_AVAILABLE:
            raise RuntimeError("ibm_boto3 not installed. Run: pip install ibm-cos-sdk")
        self._client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.api_key,
            ibm_service_instance_id=self.instance_crn,
            ibm_auth_endpoint=self.auth_endpoint,
            config=Config(signature_version="oauth"),
            endpoint_url=self.endpoint,
        )
        return self._client

    def upload(self, local_path: str, object_key: str = None) -> bool:
        """Upload a local file to COS."""
        if object_key is None:
            object_key = os.path.basename(local_path)
        try:
            cos = self._get_client()
            cos.upload_file(local_path, self.bucket, object_key)
            logger.info("Uploaded %s → COS://%s/%s", local_path, self.bucket, object_key)
            return True
        except Exception as exc:
            logger.error("COS upload failed: %s", exc)
            return False

    def download(self, object_key: str, local_path: str) -> bool:
        """Download a COS object to a local path."""
        try:
            cos = self._get_client()
            cos.download_file(self.bucket, object_key, local_path)
            logger.info("Downloaded COS://%s/%s → %s", self.bucket, object_key, local_path)
            return True
        except Exception as exc:
            logger.error("COS download failed: %s", exc)
            return False

    def list_objects(self) -> list:
        """List all objects in the configured bucket."""
        try:
            cos = self._get_client()
            response = cos.list_objects_v2(Bucket=self.bucket)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as exc:
            logger.error("COS list failed: %s", exc)
            return []


# =============================================================================
# IBM Cloudant — detection event logging
# =============================================================================

class CloudantLogger:
    """
    Persist every NIDS detection event to an IBM Cloudant database (Lite plan).
    The database 'nids_alerts' is created on first use if it doesn't exist.
    """

    def __init__(self):
        self.url       = os.getenv("IBM_CLOUDANT_URL")
        self.api_key   = os.getenv("IBM_CLOUDANT_APIKEY")
        self.db_name   = os.getenv("IBM_CLOUDANT_DB_NAME", "nids_alerts")
        self._service  = None

    def _get_service(self):
        if self._service:
            return self._service
        if not _CLOUDANT_AVAILABLE:
            raise RuntimeError(
                "ibmcloudant not installed. Run: pip install ibmcloudant"
            )
        from ibmcloudant.cloudant_v1 import CloudantV1
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        auth = IAMAuthenticator(self.api_key)
        svc  = CloudantV1(authenticator=auth)
        svc.set_service_url(self.url)
        self._service = svc
        # Ensure DB exists
        try:
            svc.put_database(db=self.db_name).get_result()
        except Exception:
            pass  # DB already exists
        return svc

    def log_event(self, event: dict) -> bool:
        """
        Persist an event document to Cloudant.

        Expected keys in `event`:
            source_ip, destination_ip, protocol, service, flag,
            prediction_label, attack_family, confidence, timestamp
        """
        if not self.url or not self.api_key:
            logger.warning("Cloudant credentials not configured – skipping log.")
            return False
        try:
            svc = self._get_service()
            doc = {
                "timestamp": event.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat()
                ),
                **event,
            }
            svc.post_document(db=self.db_name, document=doc).get_result()
            logger.info("Event logged to Cloudant: %s", doc.get("prediction_label"))
            return True
        except Exception as exc:
            logger.error("Cloudant logging failed: %s", exc)
            return False

    def fetch_recent(self, limit: int = 50) -> list:
        """Return the most recent alert documents."""
        if not self.url or not self.api_key:
            return []
        try:
            svc = self._get_service()
            result = svc.post_all_docs(
                db=self.db_name,
                include_docs=True,
                limit=limit,
                descending=True,
            ).get_result()
            return [row["doc"] for row in result.get("rows", [])]
        except Exception as exc:
            logger.error("Cloudant fetch failed: %s", exc)
            return []
