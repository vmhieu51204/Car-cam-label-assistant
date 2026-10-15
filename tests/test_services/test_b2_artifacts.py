"""Artifact digests and pre-signed URLs stay bounded at the cloud boundary."""

from pathlib import Path
from uuid import uuid4

import pytest

from src.services.cloud.artifacts import ArtifactError, B2ArtifactStore


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, path: str, _bucket: str, key: str) -> None:
        self.objects[key] = Path(path).read_bytes()

    def head_object(self, **kwargs) -> dict[str, int]:
        assert kwargs["Bucket"] == "private-bucket"
        return {"ContentLength": len(self.objects[kwargs["Key"]])}

    def download_file(self, _bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[key])

    def generate_presigned_url(self, method: str, **kwargs) -> str:
        params = kwargs["Params"]
        return f"https://s3.us-west-004.backblazeb2.com/{params['Bucket']}/{params['Key']}?operation={method}&ttl={kwargs['ExpiresIn']}"


def test_b2_upload_signed_urls_and_verified_download(tmp_path):
    fake = FakeS3()
    store = B2ArtifactStore(
        endpoint_url="https://s3.us-west-004.backblazeb2.com",
        bucket="private-bucket",
        access_key_id="",
        secret_access_key="",
        s3_client=fake,
    )
    source = tmp_path / "request.zip"
    source.write_bytes(b"bounded input")
    key = store.key(str(uuid4()), "sam3_seed", "request.zip")
    receipt = store.upload(source, key)
    assert receipt.size_bytes == len(b"bounded input")
    assert "operation=get_object" in store.signed_url(key, method="get_object", expires_s=900)
    destination = tmp_path / "download.zip"
    assert (
        store.download_verified(
            key,
            destination,
            expected_sha256=receipt.sha256,
            expected_bytes=receipt.size_bytes,
        )
        == receipt
    )
    assert destination.read_bytes() == source.read_bytes()

    fake.objects[key] = b"other content"
    with pytest.raises(ArtifactError, match="hash"):
        store.download_verified(key, destination, expected_sha256=receipt.sha256, expected_bytes=receipt.size_bytes)
    assert destination.read_bytes() == source.read_bytes()


def test_b2_artifact_bounds_and_keys(tmp_path):
    store = B2ArtifactStore(
        endpoint_url="https://s3.us-west-004.backblazeb2.com",
        bucket="private-bucket",
        access_key_id="",
        secret_access_key="",
        s3_client=FakeS3(),
    )
    with pytest.raises(ValueError, match="stage or name"):
        store.key(str(uuid4()), "../escape", "request.zip")
    with pytest.raises(ValueError, match="method or expiry"):
        store.signed_url(store.key(str(uuid4()), "vlm_audit", "result.json"), method="put_object", expires_s=1)
    oversized = tmp_path / "large.zip"
    oversized.write_bytes(b"12345")
    with pytest.raises(ArtifactError, match="size limit"):
        store.upload(oversized, store.key(str(uuid4()), "sam3_seed", "request.zip"), max_bytes=4)
