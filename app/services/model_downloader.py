import os
import shutil
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

ROOT_DIR = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT_DIR / "tmp"


MODEL_SPECS = [
    {
        "name": "embedding model",
        "env_key": "S3_EMBEDDING_MODEL_KEY",
        "target_dir": MODELS_DIR / "embedding_model",
        "required_file": "embedding_classifier.pkl",
        "archive_name": "embedding_model.zip",
    },
    {
        "name": "IT role model",
        "env_key": "S3_IT_ROLE_MODEL_KEY",
        "target_dir": MODELS_DIR / "it_role_model",
        "required_file": "embedding_classifier.pkl",
        "archive_name": "it_role_model.zip",
    },
    {
        "name": "NER model",
        "env_key": "S3_NER_MODEL_KEY",
        "target_dir": MODELS_DIR / "resume_bert_ner_model_chunked",
        "required_file": "model.safetensors",
        "archive_name": "resume_bert_ner_model_chunked.zip",
    },
]


def build_s3_client():
    endpoint_url = os.getenv("S3_ENDPOINT_URL") or None
    region_name = os.getenv("AWS_DEFAULT_REGION") or None

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if access_key and secret_key:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        config=Config(signature_version=UNSIGNED),
    )


def download_from_s3(bucket: str, key: str, target_path: Path) -> None:
    s3_client = build_s3_client()

    s3_client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(target_path),
    )


def extract_archive(archive_path: Path, target_root: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_root)


def model_exists(target_dir: Path, required_file: str) -> bool:
    return target_dir.exists() and (target_dir / required_file).exists()


def ensure_model_exists(model_spec: dict[str, object]) -> bool:
    target_dir = model_spec["target_dir"]
    required_file = model_spec["required_file"]

    if model_exists(target_dir, required_file):
        print(f"{model_spec['name']} already exists.")
        return True

    bucket = os.getenv("S3_MODEL_BUCKET")
    key = os.getenv(str(model_spec["env_key"]))

    if not bucket or not key:
        print(f"S3 settings for {model_spec['name']} are not set.")
        return False

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = TMP_DIR / str(model_spec["archive_name"])

    print(f"Downloading {model_spec['name']} from s3://{bucket}/{key}...")

    download_from_s3(
        bucket=bucket,
        key=key,
        target_path=archive_path,
    )

    print(f"Extracting {model_spec['name']} archive...")

    extract_archive(
        archive_path=archive_path,
        target_root=MODELS_DIR,
    )

    if not model_exists(target_dir, required_file):
        print(f"{model_spec['name']} was not found after extraction: {target_dir}")
        return False

    print(f"{model_spec['name']} is ready.")

    return True


def ensure_all_models_exist() -> bool:
    result = True

    for model_spec in MODEL_SPECS:
        model_ready = ensure_model_exists(model_spec)

        if not model_ready:
            result = False

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    return result
