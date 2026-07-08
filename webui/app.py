"""FastAPI backend for the pipeline dashboard.

Routes are intentionally thin: dataset inspection and config generation
live in ``config_builder``, subprocess/log-streaming lives in
``process_manager``. This file only wires HTTP/WebSocket <-> those two
modules.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import config_builder, preflight, results
from .process_manager import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webui.app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="STM32 YOLO Pipeline Dashboard")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Starlette >= 1.0 requires `request` as the first positional arg
    # (the old `TemplateResponse(name, {"request": request})` form was
    # removed and raises a confusing "unhashable dict" error).
    return templates.TemplateResponse(request, "index.html", {})


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

class SelectPathRequest(BaseModel):
    path: str


@app.post("/api/dataset/select-path")
async def select_path(payload: SelectPathRequest) -> dict[str, Any]:
    root = Path(payload.path).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.is_dir():
        raise HTTPException(400, f"Directory not found: {root}")
    try:
        info = config_builder.inspect_dataset(root)
    except config_builder.DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    return info


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip uploads are supported")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        dest_name = Path(file.filename).stem
        extracted_root = config_builder.extract_zip(tmp_path, dest_name)
        info = config_builder.inspect_dataset(extracted_root)
    except config_builder.DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return info


class SplitDatasetRequest(BaseModel):
    path: str
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42


@app.post("/api/dataset/split")
async def split_dataset(payload: SplitDatasetRequest) -> dict[str, Any]:
    root = Path(payload.path).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.is_dir():
        raise HTTPException(400, f"Directory not found: {root}")
    try:
        config_builder.split_dataset(
            root,
            {"train": payload.train, "val": payload.val, "test": payload.test},
            payload.seed,
        )
        info = config_builder.inspect_dataset(root)
    except config_builder.DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    return info


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

class GenerateConfigRequest(BaseModel):
    dataset_info: dict[str, Any]
    dataset_name: str
    epochs: int
    batch_size: int
    imgsz: int
    device: str = "cpu"
    workers: int = 0
    enable_int8: bool = False
    class_names: str = ""
    deploy_stedge: bool = False
    stedge_target: str = ""
    stedge_username: str = ""
    stedge_password: str = ""


@app.post("/api/config/generate")
async def generate_config(payload: GenerateConfigRequest) -> dict[str, Any]:
    form = payload.model_dump()
    dataset_info = form.pop("dataset_info")
    if not dataset_info or not dataset_info.get("root"):
        raise HTTPException(400, "Scan or upload a dataset before generating a config")

    try:
        config_dict = config_builder.build_config_dict(form, dataset_info)
        config_path = config_builder.write_config_yaml(config_dict, form["dataset_name"])
    except config_builder.DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "config_path": str(config_path),
        "preview": config_builder.redact_preview(config_dict),
    }


# ---------------------------------------------------------------------------
# Pipeline run control
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    config_path: str


@app.post("/api/pipeline/run")
async def run_pipeline(payload: RunRequest) -> dict[str, str]:
    config_path = Path(payload.config_path)
    if not config_path.is_file():
        raise HTTPException(400, f"Config file not found: {config_path}")
    run = await manager.start_run(config_path)
    return {"run_id": run.run_id, "status": run.status}


@app.post("/api/pipeline/stop/{run_id}")
async def stop_pipeline(run_id: str) -> dict[str, Any]:
    stopped = await manager.stop_run(run_id)
    if not stopped:
        raise HTTPException(404, "Run not found or already finished")
    return {"stopped": True}


@app.get("/api/pipeline/status/{run_id}")
async def pipeline_status(run_id: str) -> dict[str, Any]:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run_id")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "returncode": run.returncode,
    }


class PreflightRequest(BaseModel):
    config_path: str


@app.post("/api/preflight")
async def run_preflight(payload: PreflightRequest) -> dict[str, Any]:
    result = await preflight.run_preflight(payload.config_path)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/results")
async def get_results(config_path: str) -> dict[str, Any]:
    try:
        return results.load_results(config_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/ws/logs/{run_id}")
async def ws_logs(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        async for frame in manager.subscribe(run_id):
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        pass
    except Exception:  # pragma: no cover - defensive
        logger.exception("log websocket crashed for run %s", run_id)
