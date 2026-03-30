from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import gradio as gr

from interface import build_demo
from prediction_service import predict_property


class PredictionRequest(BaseModel):
    type_bien: str = Field(..., examples=["Appartement"])
    commune: str = Field(..., examples=["Autun"])
    surface_bati: float = Field(..., gt=0, examples=[65])
    nb_pieces: float = Field(..., gt=0, examples=[3])
    surface_terrain: float = Field(0, ge=0, examples=[0])
    mois: int = Field(..., ge=1, le=12, examples=[6])
    annee: int = Field(..., ge=2021, le=2035, examples=[2024])
    prix_affiche: float = Field(0, ge=0, examples=[75000])


def _allowed_origins() -> list[str]:
    raw_value = os.getenv("ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


app = FastAPI(
    title="Immo Price API",
    version="1.0.0",
    description=(
        "API REST pour estimer un bien immobilier en Saone-et-Loire et "
        "scorer une annonce depuis un portfolio."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    return predict_property(**payload.model_dump())


demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/gradio")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )
