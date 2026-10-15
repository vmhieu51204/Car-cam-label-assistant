"""Private B2 artifacts with hash checks and short-lived GPU worker URLs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

_SAFE_BUCKET = re.compile(r"^[a-zA-Z0-9-]{6,63}$")
_SAFE_STAGE = re.compile(r"^[a-z0-9_-]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Storage failure without credentials or signed URLs in the message."""


@dataclass(frozen=True)
class ArtifactReceipt:
    key: str
    sha256: str
    size_bytes: int


class B2ArtifactStore:
    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        s3_client: Any = None,
    ) -> None:
        parsed = urlsplit(endpoint_url)
        if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".backblazeb2.com"):
            raise ValueError("B2 endpoint must be an HTTPS Backblaze S3 endpoint")
        if not _SAFE_BUCKET.fullmatch(bucket):
            raise ValueError("B2 bucket name is invalid")
        if s3_client is None:
            if not access_key_id or not secret_access_key:
                raise ValueError("B2 application key ID and secret are required")
            import boto3
            from botocore.config import Config

            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url.rstrip("/"),
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
        self.bucket = bucket
        self.s3 = s3_client

    @staticmethod
    def key(job_id: str, stage: str, name: str) -> str:
        try:
            UUID(job_id)
        except ValueError as error:
            raise ValueError("Artifact job ID is invalid") from error
        if not _SAFE_STAGE.fullmatch(stage) or name not in ("request.zip", "result.json"):
            raise ValueError("Artifact stage or name is invalid")
        return f"autolabel/jobs/{job_id}/{stage}/{name}"

    def upload(self, path: Path, key: str, *, max_bytes: int = 2_147_483_648) -> ArtifactReceipt:
        self._check_key(key)
        digest, size = self._file_hash(path, max_bytes=max_bytes)
        try:
            self.s3.upload_file(str(path), self.bucket, key)
        except Exception as error:
            raise ArtifactError("B2 artifact upload failed") from error
        return ArtifactReceipt(key, digest, size)

    def download_verified(
        self,
        key: str,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
        max_bytes: int = 2_147_483_648,
    ) -> ArtifactReceipt:
        self._check_key(key)
        if not _SHA256.fullmatch(expected_sha256) or not 0 < expected_bytes <= max_bytes:
            raise ValueError("Artifact receipt is invalid")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.downloading.{uuid4().hex}")
        try:
            metadata = self.s3.head_object(Bucket=self.bucket, Key=key)
            if metadata.get("ContentLength") != expected_bytes:
                raise ArtifactError("B2 artifact size does not match receipt")
            self.s3.download_file(self.bucket, key, str(temporary))
            digest, size = self._file_hash(temporary, max_bytes=max_bytes)
            if digest != expected_sha256 or size != expected_bytes:
                raise ArtifactError("B2 artifact hash does not match receipt")
            temporary.replace(path)
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError("B2 artifact download failed") from error
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactReceipt(key, digest, size)

    def signed_url(self, key: str, *, method: str, expires_s: int = 3600) -> str:
        self._check_key(key)
        if method not in ("get_object", "put_object") or not 60 <= expires_s <= 86400:
            raise ValueError("Artifact URL method or expiry is invalid")
        try:
            url = self.s3.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_s,
            )
        except Exception as error:
            raise ArtifactError("B2 artifact URL generation failed") from error
        if urlsplit(url).scheme != "https":
            raise ArtifactError("B2 artifact URL is not HTTPS")
        return url

    @staticmethod
    def _check_key(key: str) -> None:
        if not re.fullmatch(r"autolabel/jobs/[0-9a-f-]{36}/[a-z0-9_-]{1,64}/(request\.zip|result\.json)", key):
            raise ValueError("Artifact key is invalid")
        UUID(key.split("/")[2])

    @staticmethod
    def _file_hash(path: Path, *, max_bytes: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactError("Artifact exceeds configured size limit")
                digest.update(chunk)
        if size == 0:
            raise ArtifactError("Artifact is empty")
        return digest.hexdigest(), size
