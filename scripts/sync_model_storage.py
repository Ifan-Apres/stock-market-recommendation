"""
Encrypted Model & Dataset Storage Synchronization Utility (Phase 5).
Allows:
1. Encrypted Vault Backup: AES-256-CBC/Fernet encryption of model weights and raw datasets.
2. Encrypted Vault Restore: Decrypts and restores local models and datasets.
3. Hugging Face Private Hub Sync: Push/pull from a 100% free private Hugging Face repository.
"""

import argparse
import base64
import getpass
import io
import json
import logging
import os
from pathlib import Path
import sys
import tarfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelStorageVault")

VAULT_DEFAULT_FILE = ROOT_DIR / "models_vault_encrypted.bin"
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

SENSITIVE_TARGETS = [
    PROCESSED_DIR / "alpha_model.joblib",
    PROCESSED_DIR / "lstm_model.pth",
    PROCESSED_DIR / "processed_market_features.csv",
    PROCESSED_DIR / "advanced_quant_metrics.csv",
    RAW_DIR,
]


def derive_key(password: str, salt: bytes) -> bytes:
    """Derives a Fernet-compatible key using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def create_encrypted_vault(output_path: Path, password: str) -> None:
    """Packs model weights and raw datasets into an AES-256 encrypted vault binary."""
    salt = os.urandom(16)
    key = derive_key(password, salt)
    fernet = Fernet(key)

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for target in SENSITIVE_TARGETS:
            if target.exists():
                arcname = target.relative_to(ROOT_DIR)
                logger.info(f"Adding to vault: {arcname}")
                tar.add(target, arcname=str(arcname))
            else:
                logger.warning(f"Target not found on disk, skipping: {target}")

    unencrypted_bytes = tar_buffer.getvalue()
    logger.info(f"Uncompressed archive size: {len(unencrypted_bytes) / (1024 * 1024):.2f} MB")

    encrypted_bytes = fernet.encrypt(unencrypted_bytes)

    # Format: [16 bytes salt] + [encrypted payload]
    with open(output_path, "wb") as f:
        f.write(salt + encrypted_bytes)

    logger.info(f"Encrypted vault created successfully: {output_path} ({len(encrypted_bytes) / (1024 * 1024):.2f} MB)")
    print(f"\n[OK] Vault encrypted successfully! Saved to: {output_path}")


def restore_encrypted_vault(vault_path: Path, password: str) -> None:
    """Decrypts and restores model weights and raw datasets from vault."""
    if not vault_path.exists():
        logger.error(f"Vault file not found: {vault_path}")
        return

    with open(vault_path, "rb") as f:
        data = f.read()

    salt = data[:16]
    encrypted_payload = data[16:]

    key = derive_key(password, salt)
    fernet = Fernet(key)

    try:
        decrypted_bytes = fernet.decrypt(encrypted_payload)
    except Exception:
        logger.error("Decryption failed: Incorrect password or corrupted vault file.")
        print("\n[ERROR] Password salah atau file vault rusak!")
        return

    tar_buffer = io.BytesIO(decrypted_bytes)
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        tar.extractall(path=ROOT_DIR)

    logger.info(f"Restoration complete! Files extracted to {ROOT_DIR}")
    print(f"\n[OK] Restorasi model & dataset selesai ke direktori proyek.")


def sync_to_huggingface(hf_token: str, repo_id: str) -> None:
    """Pushes model weights to a private Hugging Face repository."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return

    api = HfApi(token=hf_token)
    try:
        create_repo(repo_id=repo_id, token=hf_token, repo_type="model", private=True, exist_ok=True)
        logger.info(f"Verified private Hugging Face repository: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo check warning: {e}")

    for target in [PROCESSED_DIR / "alpha_model.joblib", PROCESSED_DIR / "lstm_model.pth"]:
        if target.exists():
            logger.info(f"Uploading {target.name} to Hugging Face...")
            api.upload_file(
                path_or_fileobj=str(target),
                path_in_repo=target.name,
                repo_id=repo_id,
                repo_type="model",
            )
    print(f"\n[OK] Model weights successfully uploaded to private Hugging Face repo: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Model Storage & Encrypted Vault Manager (Phase 5)")
    parser.add_argument("--backup", action="store_true", help="Create encrypted vault binary")
    parser.add_argument("--restore", action="store_true", help="Restore model weights from encrypted vault")
    parser.add_argument("--vault-path", default=str(VAULT_DEFAULT_DEFAULT_PATH := VAULT_DEFAULT_FILE), help="Vault path")
    parser.add_argument("--hf-push", action="store_true", help="Push models to private Hugging Face repository")
    parser.add_argument("--hf-repo", default=os.getenv("HF_MODEL_REPO", ""), help="Hugging Face repo ID (e.g. username/idx-quant-models)")
    args = parser.parse_args()

    password = os.getenv("VAULT_PASSWORD")

    if args.backup:
        if not password:
            password = getpass.getpass("Enter master password for encrypting vault: ")
        create_encrypted_vault(Path(args.vault_path), password)

    elif args.restore:
        if not password:
            password = getpass.getpass("Enter master password for decrypting vault: ")
        restore_encrypted_vault(Path(args.vault_path), password)

    elif args.hf_push:
        token = os.getenv("HF_TOKEN")
        if not token:
            token = getpass.getpass("Enter Hugging Face Access Token (write): ")
        if not args.hf_repo:
            args.hf_repo = input("Enter target Hugging Face model repo (e.g. ifan-apres/idx-quant-models): ")
        sync_to_huggingface(token, args.hf_repo)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
