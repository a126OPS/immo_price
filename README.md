---
title: Estimateur Immobilier Saone-et-Loire
sdk: gradio
sdk_version: 4.44.0
python_version: 3.12
app_file: interface.py
fullWidth: true
---

# Estimateur Immobilier Saone-et-Loire

Application Gradio de prediction immobiliere pour les biens en Saone-et-Loire.

Le Space charge les pipelines de regression et de scoring sauvegardes dans ce depot.

## Organisation Hugging Face

- Repo modeles : `a126OPS/prediction_immo_soane_et_loire`
- Space API + Gradio : `a126OPS/prediction_immo_soane_et_loirePS`

Le chargement des modeles utilise par defaut le repo Hugging Face `a126OPS/prediction_immo_soane_et_loire`.
Si les fichiers sont presents localement, tu peux forcer l'usage local avec :

```bash
IMMO_PREFER_LOCAL_ARTIFACTS=1
```

Tu peux aussi changer le repo modele avec :

```bash
HF_MODEL_REPO_ID=a126OPS/prediction_immo_soane_et_loire
```

## API pour portfolio

Le projet expose maintenant deux modes d'integration :

- `interface.py` : interface Gradio pour tester le modele manuellement
- `api.py` : API REST + interface Gradio montee sur `/gradio`

### Lancer l'API localement

```bash
python api.py
```

Endpoints principaux :

- `GET /api/health`
- `POST /api/predict`
- `GET /gradio`

### API Gradio du Space

Si tu deploies sur le Space Gradio, le portfolio peut aussi appeler directement l'endpoint Gradio
`/predict_property` via `@gradio/client`, sans passer par `api.py`.

### Exemple de payload JSON

```json
{
  "type_bien": "Appartement",
  "commune": "Autun",
  "surface_bati": 65,
  "nb_pieces": 3,
  "surface_terrain": 0,
  "mois": 6,
  "annee": 2024,
  "prix_affiche": 75000
}
```

### Exemple JavaScript pour un portfolio

```js
const response = await fetch("http://127.0.0.1:7860/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    type_bien: "Appartement",
    commune: "Autun",
    surface_bati: 65,
    nb_pieces: 3,
    surface_terrain: 0,
    mois: 6,
    annee: 2024,
    prix_affiche: 75000
  })
});

const result = await response.json();
console.log(result);
```
