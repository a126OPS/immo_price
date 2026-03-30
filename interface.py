# Chargement de tous les modèles

## 🎨 Cellule 25 — Interface Gradio finale avec scoring

# L'interface combine les deux modules :
# 1. **Estimation** : prix prédit + fourchette (module régression)
# 2. **Scoring** : l'utilisateur saisit le prix affiché → le modèle juge si c'est une bonne affaire + probabilité

# Si le champ 'prix affiché' est laissé à 0 → seulement l'estimation, pas de scoring.

pipelines_regr_final   = {}
pipelines_score_final  = {}

for type_bien in ['Maison', 'Appartement', 'Local commercial']:
    # Régression
    nom_r = f'pipeline_final_{type_bien.lower().replace(" ", "_")}.joblib'
    pipelines_regr_final[type_bien]  = joblib.load(nom_r)['pipeline']
    # Scoring (calibré)
    nom_s = f'scoring_{type_bien.lower().replace(" ", "_")}.joblib'
    pipelines_score_final[type_bien] = joblib.load(nom_s)

prix_m2_ref = joblib.load('prix_m2_reference.joblib')
MAE_CHARGEE = joblib.load('mae_par_type.joblib')
print('✅ Tous les modèles chargés')


def predire_complet(type_bien, commune, surface_bati,
                    nb_pieces, surface_terrain,
                    mois, annee, prix_affiche):
    """
    Combine régression (estimation) et scoring (évaluation du prix affiché).

    Si prix_affiche = 0 → affiche seulement l'estimation.
    Si prix_affiche > 0 → affiche estimation + scoring.
    """
    try:
        bien = pd.DataFrame([{
            'type_local'               : type_bien,
            'nom_commune'              : commune,
            'surface_reelle_bati'      : float(surface_bati),
            'nombre_pieces_principales': float(nb_pieces),
            'surface_terrain'          : float(surface_terrain),
            'longitude'                : 4.5,
            'latitude'                 : 46.5,
            'date_mutation'            : f'{int(annee)}-{int(mois):02d}-01',
            'annee_mutation'           : str(int(annee)),
            'code_postal'              : '71000'
        }])

        # -- MODULE 1 : Estimation (régression) --------------
        prix_estime = max(0, pipelines_regr_final[type_bien].predict(bien)[0])
        mae         = MAE_CHARGEE.get(type_bien, 40_000)
        prix_m2     = prix_estime / float(surface_bati) if float(surface_bati) > 0 else 0
        nb_tr       = len(df_principal[df_principal['type_local'] == type_bien])

        bloc_estimation = f"""
## 💰 Estimation du prix

| À la baisse | **Estimation centrale** | À la hausse |
|:-----------:|:----------------------:|:-----------:|
| {max(0, prix_estime - mae):,.0f} € | **{prix_estime:,.0f} €** | {prix_estime + mae:,.0f} € |

**Prix au m²** : {prix_m2:,.0f} €/m²

---

## 📋 Récapitulatif

| Caractéristique | Valeur |
|:----------------|:-------|
| Type de bien | {type_bien} |
| Commune | {commune} |
| Surface habitable | {float(surface_bati):.0f} m² |
| Nombre de pièces | {float(nb_pieces):.0f} |
| Surface terrain | {float(surface_terrain):.0f} m² |
| Période | {int(mois):02d}/{int(annee)} |
"""

        # -- MODULE 2 : Scoring (si prix affiché renseigné) --
        if float(prix_affiche) > 0:
            px_affiche = float(prix_affiche)

            # Prédiction des probabilités des 3 classes
            data_score = pipelines_score_final[type_bien]
            pipe_score = data_score['pipeline']
            proba      = pipe_score.predict_proba(bien)[0]
            # proba[0] = P(bonne affaire)
            # proba[1] = P(prix marché)
            # proba[2] = P(trop cher)

            # Ratio prix affiché / prix estimé
            ratio    = px_affiche / prix_estime if prix_estime > 0 else 1
            diff_pct = (ratio - 1) * 100  # en %
            diff_eur = px_affiche - prix_estime

            # Score de deal basé sur le ratio
            if ratio < SEUIL_BAS:
                score_label = '🟢 BONNE AFFAIRE'
                score_prob  = proba[0]
                explication = f'Ce bien est affiché {abs(diff_pct):.1f}% EN DESSOUS du prix estimé'
            elif ratio <= SEUIL_HAUT:
                score_label = '🟡 PRIX DANS LE MARCHÉ'
                score_prob  = proba[1]
                explication = f'Ce bien est affiché proche du prix estimé ({diff_pct:+.1f}%)'
            else:
                score_label = '🔴 PRIX ÉLEVÉ'
                score_prob  = proba[2]
                explication = f'Ce bien est affiché {diff_pct:.1f}% AU DESSUS du prix estimé'

            signe_diff = '+' if diff_eur >= 0 else ''

            bloc_scoring = f"""
---

## 🎯 Scoring de l'annonce

| Prix affiché | Prix estimé | Écart |
|:------------:|:-----------:|:-----:|
| {px_affiche:,.0f} € | {prix_estime:,.0f} € | {signe_diff}{diff_eur:,.0f} € ({diff_pct:+.1f}%) |

### {score_label}
_{explication}_

**Probabilité** : {score_prob*100:.1f}%

| Classe | Probabilité |
|:-------|:-----------:|
| 🟢 Bonne affaire | {proba[0]*100:.1f}% |
| 🟡 Prix marché   | {proba[1]*100:.1f}% |
| 🔴 Trop cher     | {proba[2]*100:.1f}% |

---
*⚠️ Le scoring compare le prix affiché au prix estimé par le modèle.
La fourchette d'estimation (±{mae:,}€) reflète l'incertitude,
notamment liée au quartier non modélisé.*
"""
            return bloc_estimation + bloc_scoring

        # Pas de prix affiché → juste l'estimation
        return bloc_estimation + f"""
---
*{nb_tr:,} transactions DVF 2021-2023 | Fourchette ±{mae:,}€
Pour obtenir un scoring, renseigne le champ 'Prix affiché'.*
"""

    except Exception as e:
        return f'❌ Erreur : {str(e)}'


# -- Interface Gradio ----------------------------------------
with gr.Blocks(
    title='🏠 Estimateur + Scoring Immobilier — Saône-et-Loire',
    theme=gr.themes.Soft()
) as demo_final:

    gr.Markdown("""
    # 🏠 Estimateur & Scoring Immobilier
    ## Saône-et-Loire — Données DVF officielles 2021-2023
    > Basé sur des transactions **réellement vendues** (DGFiP / data.gouv.fr)
    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown('### 📍 Localisation')
            type_bien = gr.Dropdown(
                choices=['Maison', 'Appartement', 'Local commercial'],
                value='Appartement', label='Type de bien'
            )
            commune = gr.Dropdown(
                choices=communes_71, value='Autun', label='Commune'
            )
            annee = gr.Slider(2021, 2024, value=2024, step=1, label='Année')
            mois  = gr.Slider(1, 12, value=6, step=1, label='Mois')

        with gr.Column():
            gr.Markdown('### 📐 Caractéristiques du bien')
            surface_bati = gr.Number(
                value=70, minimum=9, label='Surface habitable (m²)'
            )
            nb_pieces = gr.Slider(1, 10, value=3, step=1, label='Nombre de pièces')
            surface_terrain = gr.Number(
                value=0, minimum=0, label='Terrain (m²) — 0 si aucun'
            )

    gr.Markdown('---')
    gr.Markdown("""
    ### 🎯 Scoring d'une annonce *(optionnel)*
    Renseigne le prix affiché d'une annonce pour savoir si c'est une bonne affaire.
    Laisse à **0** si tu veux seulement l'estimation.
    """)

    prix_affiche = gr.Number(
        value=0, minimum=0,
        label='Prix affiché dans l\'annonce (€) — 0 = pas de scoring'
    )

    gr.Markdown('---')
    btn    = gr.Button('🔍 Estimer et scorer', variant='primary', size='lg')
    output = gr.Markdown('_Remplis le formulaire et clique sur Estimer_')

    btn.click(
        fn=predire_complet,
        inputs=[type_bien, commune, surface_bati, nb_pieces,
                surface_terrain, mois, annee, prix_affiche],
        outputs=output
    )

    gr.Markdown('---\n### 💡 Exemples')
    gr.Examples(
        examples=[
            # Sans scoring (prix affiché = 0)
            ['Appartement', 'Autun',            65, 3,   0, 6, 2024, 0],
            ['Maison',      'Chalon-sur-Saône', 100, 4, 300, 5, 2024, 0],
            # Avec scoring (prix affiché renseigné)
            ['Appartement', 'Autun',             65, 3,   0, 6, 2024, 75000],
            ['Maison',      'Autun',             85, 4, 500, 9, 2024, 200000],
            ['Appartement', 'Mâcon',             50, 2,   0, 3, 2024, 95000],
        ],
        inputs=[type_bien, commune, surface_bati, nb_pieces,
                surface_terrain, mois, annee, prix_affiche]
    )

    gr.Markdown("""
    ---
    ### ⚠️ Limites du modèle
    - **Quartiers non modélisés** : dans une même ville, le prix varie de ±30% selon le quartier.
    - **Données absentes** : DPE, étage, état du bien, exposition, charges.
    - **Scoring probabiliste** : une probabilité de 75% n'est pas une certitude — c'est un indicateur d'aide à la décision.
    """)


demo_final.launch(share=True, debug=True)
