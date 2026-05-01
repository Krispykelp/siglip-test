from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, model_validator
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import tempfile
import os

from analyzer.engine import run_analysis
from analyzer.schemas import make_compact_analysis_result

app = FastAPI()

PYTHON_ANALYZER_API_KEY = os.getenv("PYTHON_ANALYZER_API_KEY", "")


class AnalyzeRequest(BaseModel):
    post_id: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    claimed_tags: List[str]
    analysis_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_image_source(self):
        if not self.image_path and not self.image_url:
            raise ValueError("Either image_path or image_url must be provided.")
        return self


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"ok": True}


def require_auth(authorization: Optional[str]) -> None:
    if not PYTHON_ANALYZER_API_KEY:
        return

    expected = f"Bearer {PYTHON_ANALYZER_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def download_image_to_temp(image_url: str) -> str:
    tmp_path = None

    try:
        request = Request(
            image_url,
            headers={"User-Agent": "digidachi-analyzer/1.0"},
        )

        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "").lower()

            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL did not return an image. Content-Type was '{content_type}'.",
                )

            suffix = ".jpg"
            if "png" in content_type:
                suffix = ".png"
            elif "webp" in content_type:
                suffix = ".webp"
            elif "jpeg" in content_type or "jpg" in content_type:
                suffix = ".jpg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.read())
                tmp_path = tmp.name

        return tmp_path

    except HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image URL. HTTPError: {e.code}",
        )
    except URLError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image URL. URLError: {e.reason}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while downloading image: {str(e)}",
        )


@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: Optional[str] = Header(None)):
    require_auth(authorization)

    temp_path = None

    try:
        if req.image_path:
            image_source_path = req.image_path
        else:
            temp_path = download_image_to_temp(req.image_url)
            image_source_path = temp_path

        result = run_analysis(
            image_path=image_source_path,
            claimed_tags=req.claimed_tags,
        )

        compact = make_compact_analysis_result(
            result,
            analysis_id=req.analysis_id,
            post_id=req.post_id,
        )

        if req.image_url:
            compact["source_type"] = "image_url"
            compact["source_url"] = req.image_url
        else:
            compact["source_type"] = "image_path"

        return compact

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass