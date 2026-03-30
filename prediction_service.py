from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import joblib
from huggingface_hub import snapshot_download

from model_components import (
    MODEL_TYPES,
    SCORING_LABELS,
    SEUIL_BAS,
    SEUIL_HAUT,
    artifact_path,
    build_input_frame,
    load_pipeline_artifact,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAE = 40_000.0
DEFAULT_MODEL_REPO_ID = "a126OPS/prediction_immo_soane_et_loire"
MODEL_CACHE_DIR = BASE_DIR / ".hf_model_cache"


def _required_artifact_filenames() -> list[str]:
    filenames = [
        "mae_par_type.joblib",
        "prix_m2_reference.joblib",
        "pipeline_final_dépendance.joblib",
    ]
    for property_type in MODEL_TYPES:
        filenames.append(artifact_path(Path("."), "pipeline_final", property_type).name)
        filenames.append(artifact_path(Path("."), "scoring", property_type).name)
    return filenames


REQUIRED_ARTIFACT_FILENAMES = _required_artifact_filenames()


def _has_local_artifacts(directory: Path) -> bool:
    return all((directory / filename).exists() for filename in REQUIRED_ARTIFACT_FILENAMES)


def _download_hf_artifacts(repo_id: str) -> Path:
    local_dir = MODEL_CACHE_DIR / repo_id.replace("/", "__")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=REQUIRED_ARTIFACT_FILENAMES,
    )
    return local_dir


def resolve_artifacts_dir() -> Path:
    prefer_local = os.getenv("IMMO_PREFER_LOCAL_ARTIFACTS", "0") == "1"
    model_repo_id = os.getenv("HF_MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID).strip()

    if prefer_local and _has_local_artifacts(BASE_DIR):
        return BASE_DIR

    if model_repo_id:
        try:
            downloaded_dir = _download_hf_artifacts(model_repo_id)
            if _has_local_artifacts(downloaded_dir):
                return downloaded_dir
        except Exception:
            pass

        if _has_local_artifacts(BASE_DIR):
            return BASE_DIR
        raise FileNotFoundError(
            f"The Hugging Face model repo '{model_repo_id}' does not contain the expected artifacts yet."
        )

    if _has_local_artifacts(BASE_DIR):
        return BASE_DIR

    raise FileNotFoundError(
        "No model artifacts were found locally and no Hugging Face model repo could be downloaded."
    )


@lru_cache(maxsize=1)
def load_assets() -> dict[str, Any]:
    artifacts_dir = resolve_artifacts_dir()
    regression_models: dict[str, dict[str, Any]] = {}
    scoring_models: dict[str, dict[str, Any]] = {}

    for property_type in MODEL_TYPES:
        regression_path = artifact_path(artifacts_dir, "pipeline_final", property_type)
        if not regression_path.exists():
            raise FileNotFoundError(f"Missing model file: {regression_path.name}")
        regression_models[property_type] = load_pipeline_artifact(regression_path)

        scoring_path = artifact_path(artifacts_dir, "scoring", property_type)
        if scoring_path.exists():
            scoring_models[property_type] = load_pipeline_artifact(scoring_path)

    price_refs = joblib.load(artifacts_dir / "prix_m2_reference.joblib")
    mae_by_type = joblib.load(artifacts_dir / "mae_par_type.joblib")
    communes = sorted(
        {
            commune
            for refs_by_commune in price_refs.values()
            for commune in refs_by_commune.keys()
        }
    )

    return {
        "regression_models": regression_models,
        "scoring_models": scoring_models,
        "price_refs": price_refs,
        "mae_by_type": mae_by_type,
        "communes": communes,
        "artifacts_dir": str(artifacts_dir),
    }


def get_communes() -> list[str]:
    return load_assets()["communes"]


def get_default_commune() -> str:
    communes = get_communes()
    if "Autun" in communes:
        return "Autun"
    return communes[0] if communes else ""


def _validate_inputs(
    type_bien: str,
    commune: str,
    surface_bati: float,
    nb_pieces: float,
    surface_terrain: float,
    mois: int,
    annee: int,
    prix_affiche: float,
) -> None:
    if type_bien not in MODEL_TYPES:
        raise ValueError(f"type_bien must be one of: {', '.join(MODEL_TYPES)}")
    if not commune or not commune.strip():
        raise ValueError("commune is required")
    if float(surface_bati) <= 0:
        raise ValueError("surface_bati must be greater than 0")
    if float(nb_pieces) <= 0:
        raise ValueError("nb_pieces must be greater than 0")
    if float(surface_terrain) < 0:
        raise ValueError("surface_terrain must be greater than or equal to 0")
    if int(mois) < 1 or int(mois) > 12:
        raise ValueError("mois must be between 1 and 12")
    if int(annee) < 2021 or int(annee) > 2035:
        raise ValueError("annee must be between 2021 and 2035")
    if float(prix_affiche) < 0:
        raise ValueError("prix_affiche must be greater than or equal to 0")


def predict_property(
    type_bien: str,
    commune: str,
    surface_bati: float,
    nb_pieces: float,
    surface_terrain: float,
    mois: int,
    annee: int,
    prix_affiche: float = 0.0,
) -> dict[str, Any]:
    _validate_inputs(
        type_bien=type_bien,
        commune=commune,
        surface_bati=surface_bati,
        nb_pieces=nb_pieces,
        surface_terrain=surface_terrain,
        mois=mois,
        annee=annee,
        prix_affiche=prix_affiche,
    )

    assets = load_assets()
    property_frame = build_input_frame(
        type_bien=type_bien,
        commune=commune,
        surface_bati=surface_bati,
        nb_pieces=nb_pieces,
        surface_terrain=surface_terrain,
        mois=mois,
        annee=annee,
    )

    regression_pipeline = assets["regression_models"][type_bien]["pipeline"]
    prix_estime = max(0.0, float(regression_pipeline.predict(property_frame)[0]))
    mae = float(assets["mae_by_type"].get(type_bien, DEFAULT_MAE))
    prix_m2_estime = prix_estime / float(surface_bati)
    prix_m2_reference = assets["price_refs"].get(type_bien, {}).get(commune)

    result: dict[str, Any] = {
        "input": {
            "type_bien": type_bien,
            "commune": commune,
            "surface_bati": float(surface_bati),
            "nb_pieces": float(nb_pieces),
            "surface_terrain": float(surface_terrain),
            "mois": int(mois),
            "annee": int(annee),
            "prix_affiche": float(prix_affiche),
        },
        "estimation": {
            "prix_estime": prix_estime,
            "borne_basse": max(0.0, prix_estime - mae),
            "borne_haute": prix_estime + mae,
            "mae": mae,
            "prix_m2_estime": prix_m2_estime,
            "prix_m2_reference_commune": prix_m2_reference,
        },
        "scoring": None,
        "warnings": [],
    }

    if float(prix_affiche) <= 0:
        return result

    if type_bien not in assets["scoring_models"]:
        result["warnings"].append(
            "Scoring model is not available for this property type."
        )
        return result

    scoring_pipeline = assets["scoring_models"][type_bien]["pipeline"]
    probabilities = scoring_pipeline.predict_proba(property_frame)[0]
    ratio = float(prix_affiche) / prix_estime if prix_estime > 0 else 1.0
    diff_pct = (ratio - 1.0) * 100.0
    diff_eur = float(prix_affiche) - prix_estime

    if ratio < SEUIL_BAS:
        label_id = 0
    elif ratio <= SEUIL_HAUT:
        label_id = 1
    else:
        label_id = 2

    result["scoring"] = {
        "label": SCORING_LABELS[label_id],
        "label_id": label_id,
        "probabilite": float(probabilities[label_id]),
        "probabilites": {
            SCORING_LABELS[index]: float(value)
            for index, value in enumerate(probabilities)
        },
        "ratio_prix_affiche_sur_estime": ratio,
        "ecart_prix_eur": diff_eur,
        "ecart_prix_pct": diff_pct,
        "seuil_bas": SEUIL_BAS,
        "seuil_haut": SEUIL_HAUT,
    }
    return result


def _format_currency(value: float) -> str:
    return f"{value:,.0f} EUR"


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_prediction_markdown(result: dict[str, Any]) -> str:
    estimation = result["estimation"]
    payload = result["input"]
    scoring = result["scoring"]

    lines = [
        "## Estimation du prix",
        "",
        f"- Borne basse: {_format_currency(estimation['borne_basse'])}",
        f"- Estimation centrale: {_format_currency(estimation['prix_estime'])}",
        f"- Borne haute: {_format_currency(estimation['borne_haute'])}",
        f"- Prix au m2 estime: {_format_currency(estimation['prix_m2_estime'])}",
    ]

    if estimation["prix_m2_reference_commune"] is not None:
        lines.append(
            "- Reference commune: "
            + _format_currency(estimation["prix_m2_reference_commune"])
        )

    lines.extend(
        [
            "",
            "## Recapitulatif",
            "",
            f"- Type de bien: {payload['type_bien']}",
            f"- Commune: {payload['commune']}",
            f"- Surface habitable: {payload['surface_bati']:.0f} m2",
            f"- Nombre de pieces: {payload['nb_pieces']:.0f}",
            f"- Surface terrain: {payload['surface_terrain']:.0f} m2",
            f"- Periode: {payload['mois']:02d}/{payload['annee']}",
            "",
            f"*Fourchette d'incertitude approx.: +/- {_format_currency(estimation['mae'])}*",
        ]
    )

    if scoring:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Scoring de l'annonce",
                "",
                f"- Prix affiche: {_format_currency(payload['prix_affiche'])}",
                f"- Ecart: {_format_currency(scoring['ecart_prix_eur'])} ({scoring['ecart_prix_pct']:+.1f}%)",
                f"- Verdict: {scoring['label']}",
                f"- Probabilite dominante: {_format_percentage(scoring['probabilite'])}",
                "",
                "### Probabilites par classe",
                "",
            ]
        )
        for label, probability in scoring["probabilites"].items():
            lines.append(f"- {label}: {_format_percentage(probability)}")
    elif result["warnings"]:
        lines.extend(["", "---", "", f"Attention: {result['warnings'][0]}"])
    else:
        lines.extend(
            [
                "",
                "---",
                "",
                "Ajoute un prix affiche pour activer le scoring de l'annonce.",
            ]
        )

    return "\n".join(lines)


def predict_markdown(
    type_bien: str,
    commune: str,
    surface_bati: float,
    nb_pieces: float,
    surface_terrain: float,
    mois: int,
    annee: int,
    prix_affiche: float = 0.0,
) -> str:
    try:
        result = predict_property(
            type_bien=type_bien,
            commune=commune,
            surface_bati=surface_bati,
            nb_pieces=nb_pieces,
            surface_terrain=surface_terrain,
            mois=mois,
            annee=annee,
            prix_affiche=prix_affiche,
        )
        return format_prediction_markdown(result)
    except Exception as exc:
        return f"Erreur: {exc}"
