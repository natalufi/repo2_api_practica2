from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
UPLOADS_DIR = ARTIFACTS_DIR / "uploaded_models"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FILTER_PATH = ARTIFACTS_DIR / "filter.pkl"
MODEL_PATH = ARTIFACTS_DIR / "practica2_model.pkl"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

DELEGATION_THRESHOLD = 0.2


class PredictionResponse(BaseModel):
    p_default: float = Field(..., ge=0.0, le=1.0)
    p_low: float = Field(..., ge=0.0, le=1.0)
    p_high: float = Field(..., ge=0.0, le=1.0)
    decision: str
    reason: str


class ModelUploadResponse(BaseModel):
    version: str
    timestamp: str
    active: bool
    source: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_at: str


class ModelState:
    def __init__(self) -> None:
        self.preprocessor = None
        self.feature_filter = None
        self.model = None
        self.raw_features: list[str] = []
        self.model_version = "initial"
        self.loaded_at = ""

    def load_startup_artifacts(self) -> None:
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.feature_filter = joblib.load(FILTER_PATH)
        self.model = _load_and_validate_model(MODEL_PATH)
        schema = json.loads(SCHEMA_PATH.read_text())
        self.raw_features = list(schema["raw_features"])
        self.loaded_at = _utc_now()

    def set_model(self, model: Any, version: str) -> None:
        self.model = model
        self.model_version = version
        self.loaded_at = _utc_now()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_and_validate_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"No existe el artefacto de modelo: {path}")

    model = joblib.load(path)
    expected_methods = ["predict", "predict_proba", "predict_interval"]
    missing = [method for method in expected_methods if not hasattr(model, method)]
    if missing:
        raise ValueError(f"Modelo invalido. Faltan metodos: {missing}")
    return model


def _safe_transform(transformer: Any, data: pd.DataFrame) -> pd.DataFrame:
    transformed = transformer.transform(data)
    if isinstance(transformed, tuple):
        transformed = transformed[0]
    return transformed


def _payload_to_dataframe(payload: dict[str, Any], raw_features: list[str]) -> pd.DataFrame:
    missing = [feature for feature in raw_features if feature not in payload]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise HTTPException(
            status_code=422,
            detail=f"Faltan {len(missing)} features crudas requeridas: {preview}{suffix}",
        )

    row = {feature: payload.get(feature) for feature in raw_features}
    return pd.DataFrame([row])


state = ModelState()

app = FastAPI(
    title="Practica 2 - API de Impago",
    description="API local para servir el modelo calibrado con intervalo de incertidumbre.",
    version="0.1.0",
)


@app.on_event("startup")
def load_artifacts() -> None:
    state.load_startup_artifacts()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        loaded_at=state.loaded_at,
    )


@app.post("/model/upload", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile | None = File(default=None),
    model_path: str | None = Form(default=None),
) -> ModelUploadResponse:
    """Carga un nuevo practica2_model.pkl y lo deja activo en memoria."""

    if file is None and not model_path:
        raise HTTPException(status_code=400, detail="Sube un .pkl o envia model_path.")

    UPLOADS_DIR.mkdir(exist_ok=True)

    if file is not None:
        if not file.filename or not file.filename.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="El fichero debe tener extension .pkl.")
        version = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        destination = UPLOADS_DIR / f"{version}_{Path(file.filename).name}"
        destination.write_bytes(await file.read())
        source = str(destination)
    else:
        destination = Path(model_path).expanduser().resolve()
        version = f"local_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        source = str(destination)

    try:
        model = _load_and_validate_model(destination)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo cargar el modelo: {exc}") from exc

    state.set_model(model, version)
    return ModelUploadResponse(
        version=version,
        timestamp=state.loaded_at,
        active=True,
        source=source,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: dict[str, Any]) -> PredictionResponse:
    """Predice probabilidad de impago con intervalo y decision auto/agent."""

    if state.model is None:
        raise HTTPException(status_code=503, detail="No hay modelo cargado.")

    raw_df = _payload_to_dataframe(payload, state.raw_features)

    try:
        preprocessed = _safe_transform(state.preprocessor, raw_df)
        filtered = _safe_transform(state.feature_filter, preprocessed)
        proba = float(state.model.predict_proba(filtered)[0, 1])
        p_low_arr, p_high_arr = state.model.predict_interval(filtered)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la prediccion: {exc}") from exc

    p_low = float(np.asarray(p_low_arr)[0])
    p_high = float(np.asarray(p_high_arr)[0])
    width = p_high - p_low
    decision = "agent" if width > DELEGATION_THRESHOLD else "auto"
    reason = "p_high - p_low > 0.2" if decision == "agent" else "p_high - p_low <= 0.2"

    return PredictionResponse(
        p_default=round(proba, 6),
        p_low=round(p_low, 6),
        p_high=round(p_high, 6),
        decision=decision,
        reason=reason,
    )
