"""Wire format and HTTP helpers shared by the deploy server, client, and probe.

stdlib + numpy only so both sides work in any conda env. Observations travel
as npz archives (images + state + task), chunks as raw .npy bytes.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import numpy as np

IMG_PREFIX = "img_"
PROTOCOL_VERSION = 1


class PolicyServerError(RuntimeError):
    """An expected transport or server failure with operator-friendly context."""


class ProtocolMismatchError(PolicyServerError):
    pass


def encode_observation(
    images: dict[str, np.ndarray], state, task: str,
    consumed: int = -1, delay_ticks: int = 0,
) -> bytes:
    arrays = {IMG_PREFIX + key: np.ascontiguousarray(img) for key, img in images.items()}
    arrays["state"] = np.asarray(state, dtype=np.float32)
    arrays["task"] = np.array(task)  # 0-d unicode array, no pickle involved
    if consumed >= 0 or delay_ticks > 0:
        arrays["consumed"] = np.array(int(consumed), dtype=np.int64)
        arrays["delay_ticks"] = np.array(int(delay_ticks), dtype=np.int64)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def decode_observation(payload: bytes) -> tuple[dict[str, np.ndarray], np.ndarray, str, dict]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        images = {
            key[len(IMG_PREFIX):]: archive[key]
            for key in archive.files
            if key.startswith(IMG_PREFIX)
        }
        state = archive["state"]
        task = str(archive["task"])
        meta = {
            "consumed": int(archive["consumed"]) if "consumed" in archive.files else -1,
            "delay_ticks": int(archive["delay_ticks"]) if "delay_ticks" in archive.files else 0,
        }
    return images, state, task, meta


def encode_chunk(chunk: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(chunk, dtype=np.float32))
    return buf.getvalue()


def decode_chunk(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def http_get_json(url: str, timeout: float = 10.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        raise PolicyServerError(f"server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PolicyServerError(f"cannot reach {url}: {exc.reason}") from exc


def http_post(url: str, body: bytes = b"", timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        raise PolicyServerError(f"server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PolicyServerError(f"cannot reach {url}: {exc.reason}") from exc


class PolicyClient:
    """Small typed boundary around the deploy server's HTTP API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def info(self, timeout: float = 10.0) -> dict:
        info = http_get_json(self.base_url + "/info", timeout)
        version = info.get("protocol_version")
        if version != PROTOCOL_VERSION:
            raise ProtocolMismatchError(
                f"protocol mismatch: client={PROTOCOL_VERSION}, server={version!r}"
            )
        return info

    def reset(self, timeout: float = 15.0) -> None:
        http_post(self.base_url + "/reset", timeout=timeout)

    def predict(self, payload: bytes, timeout: float = 15.0) -> np.ndarray:
        return decode_chunk(http_post(self.base_url + "/predict", payload, timeout))
