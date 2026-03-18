from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BASE_DIR = Path(__file__).resolve().parent
MODEL_TYPES = ["Maison", "Appartement", "Local commercial"]
SCORING_LABELS = {
    0: "Bonne affaire",
    1: "Prix marche",
    2: "Trop cher",
}
SEUIL_BAS = 0.85
SEUIL_HAUT = 1.15

COMMUNES_71 = [
    "Chalon-sur-Sa\u00f4ne",
    "M\u00e2con",
    "Le Creusot",
    "Autun",
    "Montceau-les-Mines",
    "Paray-le-Monial",
    "Charnay-l\u00e8s-M\u00e2con",
    "Digoin",
    "Louhans",
    "Tournus",
    "Saint-Vallier",
    "Bourbon-Lancy",
    "Chauffailles",
    "Gueugnon",
    "Ch\u00e2tenoy-le-Royal",
    "Saint-Germain-du-Bois",
    "Saint-Marcel",
    "Chagny",
    "Blanzy",
    "Saint-R\u00e9my",
]

DVF_TYPE_MAPPING = {
    "Local industriel. commercial ou assimil\u00e9": "Local commercial",
}


def normalize_text(value):
    if pd.isna(value):
        return value
    text = str(value)
    replacements = {
        "à": "a",
        "á": "a",
        "â": "a",
        "ä": "a",
        "ç": "c",
        "è": "e",
        "é": "e",
        "ê": "e",
        "ë": "e",
        "ì": "i",
        "í": "i",
        "î": "i",
        "ï": "i",
        "ñ": "n",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
        "ý": "y",
        "ÿ": "y",
        "À": "A",
        "Á": "A",
        "Â": "A",
        "Ä": "A",
        "Ç": "C",
        "È": "E",
        "É": "E",
        "Ê": "E",
        "Ë": "E",
        "Ì": "I",
        "Í": "I",
        "Î": "I",
        "Ï": "I",
        "Ñ": "N",
        "Ò": "O",
        "Ó": "O",
        "Ô": "O",
        "Ö": "O",
        "Ù": "U",
        "Ú": "U",
        "Û": "U",
        "Ü": "U",
        "Ý": "Y",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


class ImmobilierFE(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_commune_ = X["nom_commune"].value_counts().to_dict()

        if y is not None:
            temp = X.copy()
            temp["target"] = np.asarray(y).reshape(-1)

            self.med_commune_ = (
                temp.groupby("nom_commune")["target"].median().to_dict()
            )
            temp["_m2"] = temp["target"] / temp["surface_reelle_bati"].fillna(1)
            self.m2_commune_ = (
                temp.groupby("nom_commune")["_m2"].median().to_dict()
            )
            self.m2_q25_ = (
                temp.groupby("nom_commune")["_m2"].quantile(0.25).to_dict()
            )
            self.m2_q75_ = (
                temp.groupby("nom_commune")["_m2"].quantile(0.75).to_dict()
            )
        else:
            self.med_commune_ = {}
            self.m2_commune_ = {}
            self.m2_q25_ = {}
            self.m2_q75_ = {}

        return self

    def transform(self, X):
        X = X.copy()
        X["date_mutation"] = pd.to_datetime(X["date_mutation"])
        X["mois_vente"] = X["date_mutation"].dt.month
        X["trimestre_vente"] = X["date_mutation"].dt.quarter
        X["annee_mutation"] = X["date_mutation"].dt.year.astype(int)

        X["surface_terrain"] = X["surface_terrain"].fillna(0)
        surf = X["surface_reelle_bati"].fillna(1)

        X["ratio_terrain_bati"] = X["surface_terrain"] / (surf + 1)
        X["surface_par_piece"] = surf / (
            X["nombre_pieces_principales"].fillna(1) + 1
        )

        m2_global = (
            np.median(list(self.m2_commune_.values()))
            if self.m2_commune_
            else 1500
        )
        med_global = (
            np.median(list(self.med_commune_.values()))
            if self.med_commune_
            else 100_000
        )

        X["commune_freq"] = X["nom_commune"].map(self.freq_commune_).fillna(1)
        X["prix_m2_commune"] = X["nom_commune"].map(self.m2_commune_).fillna(m2_global)
        X["prix_m2_q25"] = X["nom_commune"].map(self.m2_q25_).fillna(m2_global * 0.7)
        X["prix_m2_q75"] = X["nom_commune"].map(self.m2_q75_).fillna(m2_global * 1.3)
        X["ecart_marche"] = X["prix_m2_q75"] - X["prix_m2_q25"]
        X["prix_med_commune"] = X["nom_commune"].map(self.med_commune_).fillna(med_global)

        X["prix_estime_brut"] = surf * X["prix_m2_commune"]
        X["valeur_terrain_estime"] = X["surface_terrain"] * (
            X["prix_m2_commune"] / 3
        )
        X["score_liquidite"] = np.log1p(X["commune_freq"])

        def cat_commune(commune):
            prix = self.med_commune_.get(commune, 0)
            if prix >= 130_000:
                return "premium"
            if prix >= 90_000:
                return "intermediaire"
            return "accessible"

        def cat_surface(surface):
            if pd.isna(surface):
                return "inconnue"
            if surface < 40:
                return "tres_petit"
            if surface < 70:
                return "petit"
            if surface < 100:
                return "moyen"
            if surface < 150:
                return "grand"
            return "tres_grand"

        X["cat_commune"] = X["nom_commune"].apply(cat_commune)
        if "surface_reelle_bati" in X.columns:
            X["cat_surface"] = X["surface_reelle_bati"].apply(cat_surface)

        return X.drop(
            columns=[
                column
                for column in ["date_mutation", "code_postal", "type_local"]
                if column in X.columns
            ]
        )


class TargetEncoderCommune(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TargetEncoderCommune expects a single column.")
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X

        arr = np.asarray(X)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError("TargetEncoderCommune expects a single column.")
            arr = arr[:, 0]

        return pd.Series(arr)

    def fit(self, X, y):
        commune = self._to_series(X)
        target = pd.Series(np.asarray(y).reshape(-1))
        self.global_mean_ = target.mean()

        stats = pd.DataFrame({"commune": commune, "target": target})
        stats = stats.groupby("commune")["target"].agg(["mean", "count"])
        n = stats["count"]
        stats["encoded"] = (
            (n * stats["mean"] + self.smoothing * self.global_mean_)
            / (n + self.smoothing)
        )
        self.encoding_map_ = stats["encoded"].to_dict()
        return self

    def transform(self, X):
        commune = self._to_series(X)
        return (
            commune.map(self.encoding_map_)
            .fillna(self.global_mean_)
            .to_numpy()
            .reshape(-1, 1)
        )


def slugify_type(type_bien):
    return type_bien.lower().replace(" ", "_")


def load_pipeline_artifact(path):
    import __main__
    import joblib

    # Backward compatibility for pipelines saved from notebook cells, where
    # custom transformers were pickled under the "__main__" module.
    legacy_classes = {
        "ImmobilierFE": ImmobilierFE,
        "TargetEncoderCommune": TargetEncoderCommune,
        "ScoringImmobilierFE": ImmobilierFE,
        "ScoringTargetEncoderCommune": TargetEncoderCommune,
    }
    for name, cls in legacy_classes.items():
        setattr(__main__, name, cls)

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact
    return {"pipeline": artifact}


def build_input_frame(
    type_bien,
    commune,
    surface_bati,
    nb_pieces,
    surface_terrain,
    mois,
    annee,
):
    return pd.DataFrame(
        [
            {
                "type_local": type_bien,
                "nom_commune": commune,
                "surface_reelle_bati": float(surface_bati),
                "nombre_pieces_principales": float(nb_pieces),
                "surface_terrain": float(surface_terrain),
                "longitude": 4.5,
                "latitude": 46.5,
                "date_mutation": f"{int(annee)}-{int(mois):02d}-01",
                "annee_mutation": str(int(annee)),
                "code_postal": "71000",
            }
        ]
    )


def artifact_path(base_dir, prefix, type_bien, suffix=".joblib"):
    return Path(base_dir) / f"{prefix}_{slugify_type(type_bien)}{suffix}"
