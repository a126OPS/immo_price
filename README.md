# 🏠 Prédiction Prix Immobilier — Saône-et-Loire

## Description

Ce modèle prédit le **prix de vente d'un bien immobilier en Saône-et-Loire (71)** à partir de ses caractéristiques. Il a été développé à partir de données de transactions locales et constitue un outil d'aide à l'estimation pour ce département de Bourgogne-Franche-Comté.

## Utilisation

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Chargement du modèle
model_path = hf_hub_download(repo_id="a126OPS/prediction_immo_soane_et_loire", filename="model.joblib")
model = joblib.load(model_path)

# Exemple de prédiction
# [surface_m2, nb_pieces, code_postal, type_bien]
features = np.array([[85, 4, 71000, 1]])
predicted_price = model.predict(features)
print(f"Prix estimé : {predicted_price[0]:.0f} €")
```

## Données d'entraînement

- **Source :** Données de valeurs foncières (DVF) — open data gouvernemental
- **Zone géographique :** Département de la Saône-et-Loire (71)
- **Variables d'entrée :** surface habitable, nombre de pièces, localisation (commune / code postal), type de bien (maison / appartement)
- **Variable cible :** prix de vente en euros


## Limites

- Le modèle est spécifique à la Saône-et-Loire et ne doit pas être utilisé sur d'autres départements
- Les biens atypiques (châteaux, propriétés agricoles) sont moins bien estimés
- Les évolutions récentes du marché local ne sont pas forcément reflétées

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Démo interactive : [prediction_immo_soane_et_loirePS](https://huggingface.co/spaces/a126OPS/prediction_immo_soane_et_loirePS)

## Licence

[MIT](https://opensource.org/licenses/MIT)
