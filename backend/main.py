# main.py
# Punto de entrada del backend.
# Define la aplicación FastAPI, configura CORS y declara los endpoints.
# La lógica de machine learning vive en services/, no aquí.

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from services.bert_service    import bert_service
from services.roberta_service  import roberta_service
from services.ensemble_service import ensemble_service


# ── Aplicación FastAPI ────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Text Detector",
    description=(
        "Backend para detectar si un texto fue escrito por un humano "
        "o generado por Inteligencia Artificial. "
        "Utiliza modelos BERT y RoBERTa combinados mediante promedio ponderado."
    ),
    version="1.0.0",
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Permite que el frontend TypeScript (corriendo en localhost) se comunique
# con este backend durante el desarrollo.
# Ajustar origins en producción según el dominio real del frontend.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Esquemas de datos (Pydantic) ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Cuerpo esperado en POST /predict."""
    text: str


class BertResult(BaseModel):
    probability_ai:    float
    probability_human: float


class RobertaResult(BaseModel):
    probability_ai:    float
    probability_human: float


class PredictResponse(BaseModel):
    """Respuesta de POST /predict."""
    prediction:        str          # "IA" o "Humano"
    probability_ai:    float
    probability_human: float
    bert:              BertResult
    roberta:           RobertaResult


class HealthResponse(BaseModel):
    """Respuesta de GET /health."""
    status:    str
    mock_mode: bool
    device:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """
    Verifica que el backend está en línea.
    Informa si se está usando MOCK_MODE y qué dispositivo está activo.
    """
    # En modo real tomamos el dispositivo de cualquiera de los dos servicios
    device = bert_service.get_device_name()

    return {
        "status":    "ok",
        "mock_mode": config.MOCK_MODE,
        "device":    device,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Clasificación"])
def predict(request: PredictRequest):
    """
    Recibe un texto y devuelve la predicción: "IA" o "Humano".

    Flujo:
        1. Validar texto.
        2. Clasificar con BERT.
        3. Clasificar con RoBERTa.
        4. Combinar probabilidades (ensemble).
        5. Devolver resultado completo.
    """

    # 1. Validar texto
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="El campo 'text' no puede estar vacío.",
        )

    # 2. Clasificar con BERT
    try:
        bert_result = bert_service.predict(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error en BERT: {e}")

    # 3. Clasificar con RoBERTa
    try:
        roberta_result = roberta_service.predict(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error en RoBERTa: {e}")

    # 4. Combinar probabilidades
    ensemble_result = ensemble_service.combine(
        bert_probability_ai    = bert_result["probability_ai"],
        roberta_probability_ai = roberta_result["probability_ai"],
    )

    # 5. Construir y devolver respuesta
    return {
        "prediction":        ensemble_result["prediction"],
        "probability_ai":    ensemble_result["probability_ai"],
        "probability_human": ensemble_result["probability_human"],
        "bert": {
            "probability_ai":    bert_result["probability_ai"],
            "probability_human": bert_result["probability_human"],
        },
        "roberta": {
            "probability_ai":    roberta_result["probability_ai"],
            "probability_human": roberta_result["probability_human"],
        },
    }
