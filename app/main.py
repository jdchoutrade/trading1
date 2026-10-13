import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session
from sqlalchemy import text

from .config import APP_NAME, STATIC_DIR, UPLOAD_DIR, MAX_UPLOAD_MB, GEMINI_API_KEY, DEEPSEEK_API_KEY, FORCE_SIGNAL
from .database import Base, engine, SessionLocal, AnalysisRecord
from .ai_service import analyze_images, chat_with_deepseek
from .trade_engine import build_trade_read
from .schemas import ChatRequest, OutcomeRequest

app = FastAPI(title=APP_NAME, version="1.0.0", description="Image-only, multi-timeframe XAUUSD chart analyzer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.on_event("startup")
def startup() -> None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        # Keep the dashboard available when Neon is unreachable; health/history will show the DB issue.
        print(f"[JD ANALYSIS] Database initialization failed: {str(exc)[:300]}")

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _load_image(upload: UploadFile, raw: bytes) -> tuple[str, dict]:
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Each image must be under {MAX_UPLOAD_MB} MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Use PNG, JPG/JPEG or WEBP chart screenshots.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width < 320 or image.height < 200:
                raise HTTPException(status_code=400, detail="The chart image is too small. Upload a clearer screenshot (at least 320×200).")
            # Preserve a good local screenshot while bounding upload/storage/API size.
            stored = image.copy()
            stored.thumbnail((2200, 1600), Image.Resampling.LANCZOS)
            filename = f"{uuid.uuid4().hex}.jpg"
            out_path = UPLOAD_DIR / filename
            stored.save(out_path, "JPEG", quality=91, optimize=True)
            api_image = image.copy()
            api_image.thumbnail((1500, 1200), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            api_image.save(stream, "JPEG", quality=87, optimize=True)
            payload = {"mime_type": "image/jpeg", "base64": __import__("base64").b64encode(stream.getvalue()).decode("ascii")}
            return filename, payload
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=415, detail="That file is not a readable chart image.") from exc

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # No custom icon asset is required; return an empty response instead of a noisy 404.
    return Response(status_code=204)

@app.get("/api/health")
def health(db: Session = Depends(get_db)) -> dict:
    database = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database = "offline"
    return {"app": APP_NAME, "database": database,
            "providers": {"gemini_configured": bool(GEMINI_API_KEY), "deepseek_configured": bool(DEEPSEEK_API_KEY)},
            "mode": "image-only", "symbol": "XAUUSD", "force_signal": FORCE_SIGNAL}

@app.post("/api/analyze")
async def analyze(
    primary_image: UploadFile = File(...),
    secondary_image: UploadFile = File(...),
    tf_primary: str = Form("M1"),
    tf_secondary: str = Form("M5"),
    db: Session = Depends(get_db),
) -> dict:
    allowed_tfs = {"M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"}
    tf_primary, tf_secondary = tf_primary.upper(), tf_secondary.upper()
    if tf_primary not in allowed_tfs or tf_secondary not in allowed_tfs or tf_primary == tf_secondary:
        raise HTTPException(status_code=400, detail="Choose two different supported timeframes.")
    raw_1, raw_2 = await primary_image.read(), await secondary_image.read()
    if not raw_1 or not raw_2:
        raise HTTPException(status_code=400, detail="Upload both timeframe screenshots before analyzing.")
    name_1, image_1 = _load_image(primary_image, raw_1)
    name_2, image_2 = _load_image(secondary_image, raw_2)
    try:
        ai_results = await analyze_images([image_1, image_2], tf_primary, tf_secondary)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    provider_results = ai_results.get("provider_results", [])
    provider_errors = ai_results.get("provider_errors", {})
    if not provider_results:
        errors = " | ".join(f"{k}: {v}" for k, v in provider_errors.items())
        raise HTTPException(status_code=502, detail="All configured AI providers failed. " + (errors or "Check API keys/model names."))
    analysis = build_trade_read(provider_results, provider_errors, tf_primary, tf_secondary)
    analysis["provider_errors"] = provider_errors
    record = AnalysisRecord(
        created_at=datetime.now(timezone.utc), symbol="XAUUSD", tf_primary=tf_primary, tf_secondary=tf_secondary,
        direction=analysis["direction"], confidence=float(analysis.get("confidence", 0)),
        entry_low=analysis.get("entry_low"), entry_high=analysis.get("entry_high"), stop_loss=analysis.get("stop_loss"),
        take_profit_1=analysis.get("take_profit_1"), take_profit_2=analysis.get("take_profit_2"),
        image_primary=name_1, image_secondary=name_2, payload_json=json.dumps(analysis, ensure_ascii=False),
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        # The analysis has completed; provide an explicit persistence warning instead of throwing away the read.
        return {"id": None, "created_at": datetime.now(timezone.utc).isoformat(), "tf_primary": tf_primary,
                "tf_secondary": tf_secondary, "image_primary": name_1, "image_secondary": name_2,
                "image_primary_url": f"/uploads/{name_1}", "image_secondary_url": f"/uploads/{name_2}",
                "history_saved": False, "history_error": str(exc)[:300], "analysis": analysis}
    return {"id": record.id, "created_at": record.created_at.isoformat(), "tf_primary": tf_primary,
            "tf_secondary": tf_secondary, "image_primary": name_1, "image_secondary": name_2,
            "image_primary_url": f"/uploads/{name_1}", "image_secondary_url": f"/uploads/{name_2}",
            "history_saved": True, "analysis": analysis}

@app.get("/api/history")
def history(limit: int = Query(default=40, ge=1, le=150), db: Session = Depends(get_db)) -> dict:
    try:
        rows = db.query(AnalysisRecord).order_by(AnalysisRecord.id.desc()).limit(limit).all()
        return {"items": [_record_payload(row) for row in rows], "database": "connected"}
    except Exception as exc:
        return {"items": [], "database": "offline", "error": str(exc)[:240]}

def _record_payload(row: AnalysisRecord) -> dict:
    try:
        analysis = json.loads(row.payload_json)
    except (ValueError, TypeError):
        analysis = {}
    return {"id": row.id, "created_at": row.created_at.isoformat() if row.created_at else "",
            "symbol": row.symbol, "tf_primary": row.tf_primary, "tf_secondary": row.tf_secondary,
            "direction": row.direction, "confidence": row.confidence,
            "entry_low": row.entry_low, "entry_high": row.entry_high, "stop_loss": row.stop_loss,
            "take_profit_1": row.take_profit_1, "take_profit_2": row.take_profit_2,
            "image_primary": row.image_primary, "image_secondary": row.image_secondary,
            "image_primary_url": f"/uploads/{row.image_primary}", "image_secondary_url": f"/uploads/{row.image_secondary}",
            "analysis": analysis}

@app.get("/api/history/{record_id}")
def history_detail(record_id: int, db: Session = Depends(get_db)) -> dict:
    try:
        row = db.get(AnalysisRecord, record_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(exc)[:200]}") from exc
    if not row:
        raise HTTPException(status_code=404, detail="Signal history item was not found.")
    return _record_payload(row)

RESULT_R = {"TP1": 1.0, "TP2": 1.8, "SL": -1.0}  # R multiples (SL distance = 1R, TP1 = 1R, TP2 = 1.8R)


@app.post("/api/history/{record_id}/outcome")
def set_outcome(record_id: int, body: OutcomeRequest, db: Session = Depends(get_db)) -> dict:
    """Mark what actually happened to a signal (TP1 / TP2 / SL / SKIP / CLEAR) so the real win rate can be measured."""
    try:
        row = db.get(AnalysisRecord, record_id)
        if not row:
            raise HTTPException(status_code=404, detail="Signal history item was not found.")
        try:
            payload = json.loads(row.payload_json)
        except (ValueError, TypeError):
            payload = {}
        if body.result == "CLEAR":
            payload.pop("outcome", None)
        else:
            payload["outcome"] = {"result": body.result, "at": datetime.now(timezone.utc).isoformat()}
        row.payload_json = json.dumps(payload, ensure_ascii=False)
        db.commit()
        return _record_payload(row)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(exc)[:200]}") from exc


def _bucket() -> dict:
    return {"resolved": 0, "wins": 0, "losses": 0, "total_r": 0.0}


def _finish(b: dict) -> dict:
    n = b["resolved"]
    return {**b, "total_r": round(b["total_r"], 2), "win_rate": round(100 * b["wins"] / n, 1) if n else 0.0,
            "avg_r": round(b["total_r"] / n, 2) if n else 0.0}


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    """Real results of the signals you marked: win rate and average R, overall and per grade / direction."""
    try:
        rows = db.query(AnalysisRecord).order_by(AnalysisRecord.id.desc()).limit(2000).all()
    except Exception as exc:
        return {"resolved": 0, "unresolved": 0, "error": str(exc)[:200], "by_grade": {}, "by_direction": {}}
    total, by_grade, by_dir, unresolved = _bucket(), {}, {}, 0
    for row in rows:
        if row.direction not in ("BUY", "SELL") or row.stop_loss is None:
            continue
        try:
            a = json.loads(row.payload_json)
        except (ValueError, TypeError):
            a = {}
        result = (a.get("outcome") or {}).get("result")
        if result not in RESULT_R:
            if result != "SKIP":
                unresolved += 1
            continue
        for b in (total, by_grade.setdefault(a.get("signal_grade") or "-", _bucket()), by_dir.setdefault(row.direction, _bucket())):
            b["resolved"] += 1
            b["wins"] += 1 if RESULT_R[result] > 0 else 0
            b["losses"] += 1 if RESULT_R[result] < 0 else 0
            b["total_r"] += RESULT_R[result]
    return {**_finish(total), "unresolved": unresolved,
            "by_grade": {k: _finish(v) for k, v in sorted(by_grade.items())},
            "by_direction": {k: _finish(v) for k, v in by_dir.items()}}


@app.post("/api/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> dict:
    context = None
    row = None
    try:
        if request.analysis_id is not None:
            row = db.get(AnalysisRecord, request.analysis_id)
            if row:
                try:
                    context = json.loads(row.payload_json)
                except (ValueError, TypeError):
                    context = None
        if context is None:
            row = db.query(AnalysisRecord).order_by(AnalysisRecord.id.desc()).first()
            if row:
                try:
                    context = json.loads(row.payload_json)
                except (ValueError, TypeError):
                    context = None
    except Exception:
        # Chat remains available without database context if Neon is temporarily unreachable.
        context, row = None, None
    try:
        answer = await chat_with_deepseek(request.message, context)
        return {"answer": answer, "provider": "DeepSeek", "context_analysis_id": request.analysis_id or (row.id if 'row' in locals() and row else None)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
