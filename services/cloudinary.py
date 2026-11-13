import os
import time
from typing import Optional


def cloudinary_configured() -> bool:
    return all(
        [
            os.getenv("CLOUDINARY_CLOUD_NAME", "").strip(),
            os.getenv("CLOUDINARY_API_KEY", "").strip(),
            os.getenv("CLOUDINARY_API_SECRET", "").strip(),
        ]
    )


def _configure_cloudinary():
    if not cloudinary_configured():
        return False

    try:
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        )
        return True
    except Exception:
        return False


def upload_chart_png(file_path: str, run_id: str, title: str) -> dict:
    if os.getenv('TESTING') or os.getenv('PYTEST_CURRENT_TEST') or not _configure_cloudinary():
        return {"secure_url": None, "public_id": None, "uploaded": False}

    try:
        import cloudinary.uploader

        result = cloudinary.uploader.upload(
            file_path,
            folder=f"anchoriq/{run_id}",
            public_id=title.replace(" ", "_").lower()[:80],
            resource_type="image",
        )
        return {
            "secure_url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "uploaded": True,
        }
    except Exception:
        return {"secure_url": None, "public_id": None, "uploaded": False}


def upload_private_file(file_path: str, run_id: str, asset_name: str) -> dict:
    """Store sensitive run artifacts as authenticated Cloudinary raw assets."""
    if os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST") or not _configure_cloudinary():
        return {"public_id": None, "uploaded": False}

    try:
        import cloudinary.uploader

        extension = os.path.splitext(file_path)[1].lower()
        stem = os.path.basename(asset_name or "asset")
        if extension and not stem.lower().endswith(extension):
            stem = f"{stem}{extension}"
        result = cloudinary.uploader.upload(
            file_path,
            folder=f"anchoriq/{run_id}/private",
            public_id=stem,
            resource_type="raw",
            type="authenticated",
            overwrite=True,
            unique_filename=False,
            use_filename=False,
        )
        return {
            "public_id": result.get("public_id"),
            "format": result.get("format") or extension.lstrip("."),
            "uploaded": bool(result.get("public_id")),
        }
    except Exception:
        return {"public_id": None, "uploaded": False}


def download_private_file(public_id: str, file_format: Optional[str] = None) -> bytes:
    """Fetch an authenticated raw asset with a short-lived signed URL."""
    if not public_id or not _configure_cloudinary():
        raise RuntimeError("Private Cloudinary storage is not configured.")

    import requests
    from cloudinary.utils import private_download_url

    url = private_download_url(
        public_id,
        format=file_format,
        resource_type="raw",
        type="authenticated",
        expires_at=int(time.time()) + 120,
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def delete_cloudinary_image(public_id: Optional[str]) -> bool:
    if not public_id or not _configure_cloudinary():
        return False

    try:
        import cloudinary.uploader

        response = cloudinary.uploader.destroy(public_id)
        return bool(response and response.get("result") in {"ok", "deleted"})
    except Exception:
        return False


def delete_private_file(public_id: Optional[str]) -> bool:
    if not public_id or not _configure_cloudinary():
        return False
    try:
        import cloudinary.uploader

        response = cloudinary.uploader.destroy(
            public_id,
            resource_type="raw",
            type="authenticated",
        )
        return bool(response and response.get("result") in {"ok", "deleted"})
    except Exception:
        return False
