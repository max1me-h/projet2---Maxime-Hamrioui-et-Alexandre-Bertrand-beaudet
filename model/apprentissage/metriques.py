"""Métriques de classification binaire calculées avec NumPy, sans boucle Python.

Les tests vérifient que chaque fonction donne le même résultat que scikit-learn.
Convention : 1 = l'employé part, 0 = il reste.
"""

import numpy as np


def _preparer(y_vrai, y_predit) -> tuple[np.ndarray, np.ndarray]:
    """Convertit les deux séries en tableaux d'entiers 0/1.

    Lève ValueError si elles sont vides, de tailles différentes ou contiennent autre chose que 0 et 1.
    """
    vrai = np.asarray(y_vrai).astype(int).ravel()
    predit = np.asarray(y_predit).astype(int).ravel()
    if vrai.size == 0:
        raise ValueError("Aucune observation à évaluer.")
    if vrai.shape != predit.shape:
        raise ValueError(f"Tailles différentes : {vrai.size} valeurs réelles, {predit.size} prédites.")
    if not (np.isin(vrai, [0, 1]).all() and np.isin(predit, [0, 1]).all()):
        raise ValueError("Les étiquettes doivent valoir 0 ou 1.")
    return vrai, predit


def matrice_confusion(y_vrai, y_predit) -> np.ndarray:
    """Renvoie la matrice 2 × 2 [[VN, FP], [FN, VP]] (lignes = réel, colonnes = prédit)."""
    vrai, predit = _preparer(y_vrai, y_predit)
    # Chaque couple (réel, prédit) devient un indice 0..3, puis on compte les indices.
    return np.bincount(2 * vrai + predit, minlength=4).reshape(2, 2)


def exactitude(y_vrai, y_predit) -> float:
    """Proportion de prédictions correctes."""
    vrai, predit = _preparer(y_vrai, y_predit)
    return float(np.mean(vrai == predit))


def rappel(y_vrai, y_predit) -> float:
    """Parmi les employés réellement partis, proportion que le modèle a détectés (0 si aucun départ)."""
    matrice = matrice_confusion(y_vrai, y_predit)
    total = matrice[1].sum()
    return float(matrice[1, 1] / total) if total else 0.0


def precision_depart(y_vrai, y_predit) -> float:
    """Parmi les départs prédits, proportion de vrais départs (0 si aucun départ prédit)."""
    matrice = matrice_confusion(y_vrai, y_predit)
    total = matrice[:, 1].sum()
    return float(matrice[1, 1] / total) if total else 0.0


def score_f1(y_vrai, y_predit) -> float:
    """Moyenne harmonique de la précision et du rappel sur la classe « départ »."""
    p, r = precision_depart(y_vrai, y_predit), rappel(y_vrai, y_predit)
    return 2 * p * r / (p + r) if p + r else 0.0


def exactitude_equilibree(y_vrai, y_predit) -> float:
    """Moyenne des taux de bonnes réponses de chaque classe présente.

    Un modèle qui répond toujours la même chose obtient 0,5, peu importe le déséquilibre :
    c'est la métrique principale du projet.
    """
    matrice = matrice_confusion(y_vrai, y_predit)
    effectifs = matrice.sum(axis=1)
    presentes = effectifs > 0
    return float(np.mean(np.diag(matrice)[presentes] / effectifs[presentes]))


def resume_metriques(y_vrai, y_predit) -> dict[str, float]:
    """Renvoie toutes les métriques dans un dictionnaire (clés en français)."""
    return {
        "exactitude_equilibree": exactitude_equilibree(y_vrai, y_predit),
        "exactitude": exactitude(y_vrai, y_predit),
        "rappel_depart": rappel(y_vrai, y_predit),
        "precision_depart": precision_depart(y_vrai, y_predit),
        "f1_depart": score_f1(y_vrai, y_predit),
    }
