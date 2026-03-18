from pathlib import Path

import gradio as gr
import joblib

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


def load_regression_models():
    models = {}
    for type_bien in MODEL_TYPES:
        path = artifact_path(BASE_DIR, "pipeline_final", type_bien)
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path.name}")
        models[type_bien] = load_pipeline_artifact(path)
    return models


def load_scoring_models():
    models = {}
    for type_bien in MODEL_TYPES:
        path = artifact_path(BASE_DIR, "scoring", type_bien)
        if not path.exists():
            return {}
        models[type_bien] = load_pipeline_artifact(path)
    return models


REGRESSION_MODELS = load_regression_models()
SCORING_MODELS = load_scoring_models()
PRIX_M2_REF = joblib.load(BASE_DIR / "prix_m2_reference.joblib")
MAE_PAR_TYPE = joblib.load(BASE_DIR / "mae_par_type.joblib")
COMMUNES_71 = sorted(
    {
        commune
        for refs_by_commune in PRIX_M2_REF.values()
        for commune in refs_by_commune.keys()
    }
)
DEFAULT_COMMUNE = "Autun" if "Autun" in COMMUNES_71 else COMMUNES_71[0]


def build_scoring_block(prix_estime, prix_affiche, proba):
    ratio = prix_affiche / prix_estime if prix_estime > 0 else 1.0
    diff_pct = (ratio - 1) * 100
    diff_eur = prix_affiche - prix_estime

    if ratio < SEUIL_BAS:
        label = SCORING_LABELS[0]
        explication = (
            f"Le prix affiche est {abs(diff_pct):.1f}% en dessous du prix estime."
        )
    elif ratio <= SEUIL_HAUT:
        label = SCORING_LABELS[1]
        explication = f"Le prix affiche est proche du prix estime ({diff_pct:+.1f}%)."
    else:
        label = SCORING_LABELS[2]
        explication = f"Le prix affiche est {diff_pct:.1f}% au dessus du prix estime."

    return f"""
---

## Scoring de l'annonce

| Prix affiche | Prix estime | Ecart |
|:------------:|:-----------:|:-----:|
| {prix_affiche:,.0f} EUR | {prix_estime:,.0f} EUR | {diff_eur:+,.0f} EUR ({diff_pct:+.1f}%) |

### {label}
_{explication}_

| Classe | Probabilite |
|:-------|:-----------:|
| {SCORING_LABELS[0]} | {proba[0] * 100:.1f}% |
| {SCORING_LABELS[1]} | {proba[1] * 100:.1f}% |
| {SCORING_LABELS[2]} | {proba[2] * 100:.1f}% |
"""


def predire(
    type_bien,
    commune,
    surface_bati,
    nb_pieces,
    surface_terrain,
    mois,
    annee,
    prix_affiche,
):
    try:
        bien = build_input_frame(
            type_bien=type_bien,
            commune=commune,
            surface_bati=surface_bati,
            nb_pieces=nb_pieces,
            surface_terrain=surface_terrain,
            mois=mois,
            annee=annee,
        )

        regression_pipeline = REGRESSION_MODELS[type_bien]["pipeline"]
        prix_estime = max(0.0, float(regression_pipeline.predict(bien)[0]))
        mae = float(MAE_PAR_TYPE.get(type_bien, 40_000))
        prix_m2 = prix_estime / float(surface_bati) if float(surface_bati) > 0 else 0.0
        prix_m2_ref = PRIX_M2_REF.get(type_bien, {}).get(commune)

        resultat = f"""
## Estimation du prix

| Bas | Estimation centrale | Haut |
|:---:|:-------------------:|:----:|
| {max(0, prix_estime - mae):,.0f} EUR | **{prix_estime:,.0f} EUR** | {prix_estime + mae:,.0f} EUR |

**Prix au m2 estime** : {prix_m2:,.0f} EUR/m2
"""

        if prix_m2_ref is not None:
            resultat += f"\n**Reference commune** : {prix_m2_ref:,.0f} EUR/m2\n"

        resultat += f"""

---

## Recapitulatif

| Champ | Valeur |
|:------|:-------|
| Type de bien | {type_bien} |
| Commune | {commune} |
| Surface habitable | {float(surface_bati):.0f} m2 |
| Nombre de pieces | {float(nb_pieces):.0f} |
| Surface terrain | {float(surface_terrain):.0f} m2 |
| Periode | {int(mois):02d}/{int(annee)} |

---
*Fourchette d'incertitude approx. : +/- {mae:,.0f} EUR*
"""

        prix_affiche = float(prix_affiche)
        if prix_affiche > 0:
            if type_bien not in SCORING_MODELS:
                return (
                    resultat
                    + "\n\nScoring indisponible : les modeles de scoring ne sont pas encore generes."
                )

            scoring_pipeline = SCORING_MODELS[type_bien]["pipeline"]
            proba = scoring_pipeline.predict_proba(bien)[0]
            resultat += build_scoring_block(prix_estime, prix_affiche, proba)

        return resultat
    except Exception as exc:
        return f"Erreur : {exc}"


with gr.Blocks(title="Estimateur immobilier Saone-et-Loire") as demo:
    gr.Markdown(
        """
# Estimateur immobilier
## Saone-et-Loire - DVF 2021-2023
"""
    )

    with gr.Row():
        with gr.Column():
            type_bien = gr.Dropdown(
                choices=MODEL_TYPES,
                value="Appartement",
                label="Type de bien",
            )
            commune = gr.Dropdown(
                choices=COMMUNES_71,
                value=DEFAULT_COMMUNE,
                label="Commune",
            )
            annee = gr.Slider(2021, 2026, value=2024, step=1, label="Annee")
            mois = gr.Slider(1, 12, value=6, step=1, label="Mois")

        with gr.Column():
            surface_bati = gr.Number(
                value=70,
                minimum=9,
                label="Surface habitable (m2)",
            )
            nb_pieces = gr.Slider(
                1,
                10,
                value=3,
                step=1,
                label="Nombre de pieces",
            )
            surface_terrain = gr.Number(
                value=0,
                minimum=0,
                label="Surface terrain (m2)",
            )
            prix_affiche = gr.Number(
                value=0,
                minimum=0,
                label="Prix affiche (EUR, optionnel)",
            )

    btn = gr.Button("Estimer", variant="primary")
    output = gr.Markdown("Remplis le formulaire puis clique sur Estimer.")

    btn.click(
        fn=predire,
        inputs=[
            type_bien,
            commune,
            surface_bati,
            nb_pieces,
            surface_terrain,
            mois,
            annee,
            prix_affiche,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)
