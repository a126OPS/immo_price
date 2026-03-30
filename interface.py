from __future__ import annotations

import gradio as gr

from model_components import MODEL_TYPES
from prediction_service import (
    get_communes,
    get_default_commune,
    predict_markdown,
    predict_property,
)


def build_demo() -> gr.Blocks:
    communes = get_communes()
    default_commune = get_default_commune()

    with gr.Blocks(
        title="Estimateur immobilier Saone-et-Loire",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """
# Estimateur et scoring immobilier
## Saone-et-Loire - DVF 2021-2023

Cette interface permet d'estimer le prix d'un bien et, si un prix affiche est fourni,
de scorer l'annonce.
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
                    choices=communes,
                    value=default_commune,
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

        gr.Markdown(
            """
### API portfolio

L'interface expose aussi une API publique `predict_property`.
Elle peut etre consommee par un portfolio via `@gradio/client` ou par l'application REST dans `api.py`.
"""
        )

        button = gr.Button("Estimer", variant="primary")
        output = gr.Markdown("Remplis le formulaire puis clique sur Estimer.")
        api_output = gr.JSON(visible=False)
        api_trigger = gr.Button(visible=False)

        button.click(
            fn=predict_markdown,
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
            queue=False,
            api_name=False,
        )

        gr.Examples(
            examples=[
                ["Appartement", default_commune, 65, 3, 0, 6, 2024, 0],
                ["Maison", "Chalon-sur-Saône", 100, 4, 300, 5, 2024, 0],
                ["Appartement", default_commune, 65, 3, 0, 6, 2024, 75000],
            ],
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
        )

        api_trigger.click(
            fn=predict_property,
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
            outputs=api_output,
            api_name="predict_property",
            queue=False,
            show_api=True,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(show_api=True)
