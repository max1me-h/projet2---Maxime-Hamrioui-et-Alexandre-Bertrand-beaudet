"""Prédire avec un formulaire incomplet et mesurer la précision qui en résulte.

Quand un champ n'est pas renseigné, on ne connaît pas sa valeur. Plutôt que de supposer
une valeur unique, on essaie les valeurs de K vrais employés de référence (tirés de
l'ensemble d'entraînement) et on fait la moyenne des risques obtenus. Le même calcul,
appliqué aux employés du jeu de test, donne la précision attendue pour un ensemble
donné de champs renseignés : c'est le pourcentage affiché dans l'interface.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from model.config import GRAINE, SEUIL_DECISION
from model.exceptions import DonneeInvalideError
from model.apprentissage.metriques import exactitude_equilibree
from model.apprentissage.modeles import Modele


def risques_partiels(modele: Modele, X: pd.DataFrame, champs_connus: Iterable[str],
                     reference: pd.DataFrame) -> np.ndarray:
    """Risque de départ de chaque ligne de X quand seuls champs_connus sont connus.

    Les colonnes inconnues prennent tour à tour les valeurs des K lignes de reference ;
    on renvoie la moyenne des K risques pour chaque ligne (tableau de taille len(X)).
    Lève DonneeInvalideError s'il manque une colonne connue dans X ou une colonne dans reference.
    """
    connus = set(champs_connus)
    colonnes_connues = [nom for nom in modele.colonnes if nom in connus]
    colonnes_inconnues = [nom for nom in modele.colonnes if nom not in connus]
    absentes = [nom for nom in colonnes_connues if nom not in X.columns]
    absentes += [nom for nom in colonnes_inconnues if nom not in reference.columns]
    if absentes:
        raise DonneeInvalideError(f"Colonnes absentes : {', '.join(absentes)}")
    if not colonnes_inconnues:
        return modele.probabilites_depart(X)
    n, k = len(X), len(reference)
    # Grille de n × k lignes : la ligne i de X est répétée k fois (np.repeat) et associée,
    # à chaque fois, à un employé de référence différent (np.tile). Aucune boucle Python.
    connues = X[colonnes_connues].iloc[np.repeat(np.arange(n), k)].reset_index(drop=True)
    inconnues = reference[colonnes_inconnues].iloc[np.tile(np.arange(k), n)].reset_index(drop=True)
    grille = pd.concat([connues, inconnues], axis=1)
    return modele.probabilites_depart(grille).reshape(n, k).mean(axis=1)


def precision_selon_champs(modele: Modele, X_test: pd.DataFrame, y_test, champs_connus: Iterable[str],
                           reference: pd.DataFrame) -> float:
    """Exactitude équilibrée mesurée sur le test quand seuls champs_connus sont renseignés."""
    risques = risques_partiels(modele, X_test, champs_connus, reference)
    return exactitude_equilibree(y_test, (risques >= SEUIL_DECISION).astype(int))


def effets_champs(modele: Modele, profil: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Effet de chaque champ renseigné sur le risque d'un employé (entre -1 et 1).

    profil : tableau d'une ligne contenant seulement les champs renseignés.
    Effet d'un champ = risque avec tous les champs - risque si ce champ était laissé vide.
    Positif : la valeur saisie augmente le risque. Lève DonneeInvalideError si profil
    n'a pas exactement une ligne.
    """
    if len(profil) != 1:
        raise DonneeInvalideError("Le profil doit contenir exactement une ligne.")
    champs = [nom for nom in profil.columns if nom in modele.colonnes]
    risque = risques_partiels(modele, profil, champs, reference)[0]
    return {nom: float(risque - risques_partiels(modele, profil,
                                                  [autre for autre in champs if autre != nom],
                                                  reference)[0])
            for nom in champs}


def courbe_precision(modele: Modele, X_test: pd.DataFrame, y_test, reference: pd.DataFrame,
                     ordre: list[str]) -> list[float]:
    """Précision attendue avec 0, 1, 2, ... champs renseignés, ajoutés dans l'ordre donné."""
    return [precision_selon_champs(modele, X_test, y_test, ordre[:nombre], reference)
            for nombre in range(len(ordre) + 1)]


def courbe_precision_aleatoire(modele: Modele, X_test: pd.DataFrame, y_test, reference: pd.DataFrame,
                               nb_tirages: int, graine: int = GRAINE) -> dict[str, list[float]]:
    """Précision moyenne (et 10e et 90e centiles) quand les champs renseignés sont pris au hasard.

    Lève ValueError si nb_tirages < 1.
    """
    if nb_tirages < 1:
        raise ValueError("Il faut au moins un tirage.")
    generateur = np.random.default_rng(graine)
    courbes = np.array([
        courbe_precision(modele, X_test, y_test, reference,
                         [str(nom) for nom in generateur.permutation(modele.colonnes)])
        for _ in range(nb_tirages)
    ])
    return {"moyenne": courbes.mean(axis=0).tolist(),
            "bas": np.percentile(courbes, 10, axis=0).tolist(),
            "haut": np.percentile(courbes, 90, axis=0).tolist()}
