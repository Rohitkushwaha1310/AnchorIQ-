import os
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


def delete_cloudinary_image(public_id: Optional[str]) -> bool:
    if not public_id or not _configure_cloudinary():
        return False

    try:
        import cloudinary.uploader

        response = cloudinary.uploader.destroy(public_id)
        return bool(response and response.get("result") in {"ok", "deleted"})
    except Exception:
        return False
