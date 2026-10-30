"""Lecture, nettoyage et séparation du jeu de données ; lecture et écriture JSON."""

import json
import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (CHEMIN_CSV, COLONNE_CIBLE, COLONNES_RETIREES, DESCRIPTIONS, GRAINE,
                        PROPORTION_TEST, VALEUR_DEPART)
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.variables import Variable, VariableCategorielle, VariableNumerique, VariableOrdinale


def verifier_fichier(chemin: str, conseil: str = "") -> None:
    """Lève FichierIntrouvableError si le fichier n'existe pas ; ne renvoie rien sinon."""
    if not os.path.exists(chemin):
        raise FichierIntrouvableError(chemin, conseil)


def lire_par_blocs(chemin: str, taille_bloc: int = 500) -> Iterator[pd.DataFrame]:
    """Générateur : produit le CSV par blocs de taille_bloc lignes, sans tout lire d'un coup.

    Lève FichierIntrouvableError si le fichier est absent, ValueError si taille_bloc < 1.
    """
    if taille_bloc < 1:
        raise ValueError("La taille d'un bloc doit être au moins 1.")
    verifier_fichier(chemin)
    # utf-8-sig retire le caractère invisible (BOM) placé au début du fichier IBM.
    with open(chemin, encoding="utf-8-sig", newline="") as fichier:
        yield from pd.read_csv(fichier, chunksize=taille_bloc)


def compter_valeurs_aberrantes(valeurs: np.ndarray) -> int:
    """Compte (sans boucle) les valeurs hors de [Q1 - 1,5 × EIQ, Q3 + 1,5 × EIQ]."""
    valeurs = np.asarray(valeurs, dtype=float)
    if valeurs.size == 0:
        return 0
    q1, q3 = np.percentile(valeurs, [25, 75])
    ecart = q3 - q1
    return int(np.sum((valeurs < q1 - 1.5 * ecart) | (valeurs > q3 + 1.5 * ecart)))


def lire_csv(chemin: str, conseil: str = "") -> pd.DataFrame:
    """Lit un petit CSV produit par le projet (jeu de test, employés de référence).

    Lève FichierIntrouvableError si le fichier est absent, DonneeInvalideError s'il est vide.
    """
    verifier_fichier(chemin, conseil)
    with open(chemin, encoding="utf-8", newline="") as fichier:
        try:
            return pd.read_csv(fichier)
        except pd.errors.EmptyDataError:
            raise DonneeInvalideError(f"Le fichier {chemin} est vide.") from None


def ecrire_json(chemin: str, contenu: dict) -> None:
    """Écrit contenu dans un fichier JSON lisible (indenté, accents conservés)."""
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(contenu, fichier, ensure_ascii=False, indent=2)


def lire_json(chemin: str, conseil: str = "") -> dict:
    """Renvoie le contenu d'un fichier JSON.

    Lève FichierIntrouvableError si le fichier est absent, DonneeInvalideError s'il est corrompu.
    """
    verifier_fichier(chemin, conseil)
    with open(chemin, encoding="utf-8") as fichier:
        try:
            return json.load(fichier)
        except json.JSONDecodeError as erreur:
            raise DonneeInvalideError(f"Fichier JSON illisible ({chemin}) : {erreur}") from None


class JeuDeDonnees:
    """Jeu de données IBM : chargement, nettoyage et séparation entraînement/test."""

    def __init__(self, chemin: str = CHEMIN_CSV) -> None:
        """Reçoit le chemin du fichier CSV ; rien n'est lu avant l'appel à charger()."""
        self._chemin = chemin
        self._tableau: pd.DataFrame | None = None
        self._rapport: list[str] = []

    def charger(self, taille_bloc: int = 500) -> "JeuDeDonnees":
        """Lit le CSV par blocs et renvoie l'objet lui-même (pour enchaîner les appels).

        Lève FichierIntrouvableError si le fichier est absent, DonneeInvalideError s'il est vide.
        """
        try:
            blocs = list(lire_par_blocs(self._chemin, taille_bloc))
        except pd.errors.EmptyDataError:
            raise DonneeInvalideError(f"Le fichier {self._chemin} est vide.") from None
        if not blocs:
            raise DonneeInvalideError(f"Le fichier {self._chemin} ne contient aucune ligne.")
        self._tableau = pd.concat(blocs, ignore_index=True)
        self._rapport = [f"{len(self._tableau)} lignes et {self._tableau.shape[1]} colonnes lues."]
        return self

    def nettoyer(self) -> list[str]:
        """Nettoie les données chargées et renvoie la liste des corrections effectuées.

        Lève DonneeInvalideError si les données ne sont pas chargées ou s'il manque une colonne.
        """
        tableau = self._exiger_tableau()
        manquantes = [nom for nom in [COLONNE_CIBLE, *DESCRIPTIONS] if nom not in tableau.columns]
        if manquantes:
            raise DonneeInvalideError(f"Colonnes absentes du fichier : {', '.join(manquantes)}")
        rapport = []

        avant = len(tableau)
        tableau = tableau.drop_duplicates()
        rapport.append(f"Doublons exacts retirés : {avant - len(tableau)}.")

        retirees = [nom for nom in COLONNES_RETIREES if nom in tableau.columns]
        tableau = tableau.drop(columns=retirees)
        rapport.append(f"Colonnes retirées ({len(retirees)}) : {', '.join(retirees)}.")

        for nom in tableau.select_dtypes(exclude="number").columns:
            tableau[nom] = tableau[nom].str.strip()

        nb_non_numeriques = 0
        for nom, description in DESCRIPTIONS.items():
            if description["type"] != "categorielle":
                converti = pd.to_numeric(tableau[nom], errors="coerce")
                nb_non_numeriques += int((converti.isna() & tableau[nom].notna()).sum())
                tableau[nom] = converti
        rapport.append(f"Valeurs non numériques dans une colonne numérique : {nb_non_numeriques}.")

        cible_invalide = ~tableau[COLONNE_CIBLE].isin(["Yes", "No"])
        incompletes = tableau.isna().any(axis=1) | cible_invalide
        tableau = tableau[~incompletes].copy()
        rapport.append(f"Lignes incomplètes ou à cible invalide retirées : {int(incompletes.sum())}.")
        for nom, description in DESCRIPTIONS.items():
            if description["type"] != "categorielle":
                tableau[nom] = tableau[nom].astype(int)

        aberrantes = []
        for nom, description in DESCRIPTIONS.items():
            if description["type"] == "numerique":
                nombre = compter_valeurs_aberrantes(tableau[nom].to_numpy())
                if nombre:
                    aberrantes.append(f"{nom} ({nombre})")
        rapport.append("Valeurs aberrantes (règle 1,5 × EIQ) conservées car plausibles "
                       f"(hauts salaires, longues carrières) : {', '.join(aberrantes) or 'aucune'}.")

        self._tableau = tableau[[*DESCRIPTIONS, COLONNE_CIBLE]].reset_index(drop=True)
        self._rapport.extend(rapport)
        return rapport

    def separer(self, proportion_test: float = PROPORTION_TEST, graine: int = GRAINE
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Renvoie (X_entrainement, X_test, y_entrainement, y_test), stratifiés selon la cible.

        Lève ValueError si proportion_test n'est pas strictement entre 0 et 1.
        """
        if not 0 < proportion_test < 1:
            raise ValueError("La proportion de test doit être strictement entre 0 et 1.")
        return train_test_split(self.caracteristiques, self.cible, test_size=proportion_test,
                                random_state=graine, stratify=self.cible)

    @property
    def tableau(self) -> pd.DataFrame:
        """Copie du tableau complet (caractéristiques et cible)."""
        return self._exiger_tableau().copy()

    @property
    def caracteristiques(self) -> pd.DataFrame:
        """Copie des colonnes d'entrée du modèle, dans l'ordre de config.DESCRIPTIONS."""
        return self._exiger_tableau()[list(DESCRIPTIONS)].copy()

    @property
    def cible(self) -> pd.Series:
        """Cible binaire : 1 si l'employé a quitté l'entreprise, 0 sinon."""
        tableau = self._exiger_tableau()
        return (tableau[COLONNE_CIBLE] == VALEUR_DEPART).astype(int).rename("Depart")

    @property
    def rapport(self) -> list[str]:
        """Copie du rapport de chargement et de nettoyage."""
        return list(self._rapport)

    def taux_depart(self) -> float:
        """Proportion d'employés partis, entre 0 et 1."""
        return float(np.mean(self.cible.to_numpy()))

    def _exiger_tableau(self) -> pd.DataFrame:
        """Renvoie le tableau chargé ; lève DonneeInvalideError si charger() n'a pas été appelé."""
        if self._tableau is None:
            raise DonneeInvalideError("Données non chargées : appelez charger() d'abord.")
        return self._tableau

    def __len__(self) -> int:
        return 0 if self._tableau is None else len(self._tableau)

    def __repr__(self) -> str:
        return f"JeuDeDonnees(chemin={os.path.basename(self._chemin)!r}, lignes={len(self)})"


def creer_variables(X: pd.DataFrame) -> list[Variable]:
    """Construit les variables du formulaire à partir des données d'entraînement X.

    Bornes min/max pour les nombres, valeurs observées pour les listes ; la valeur typique
    est la médiane (nombres, échelles) ou la valeur la plus fréquente (catégories).
    Lève DonneeInvalideError si une colonne décrite dans la configuration est absente de X.
    """
    variables: list[Variable] = []
    for nom, description in DESCRIPTIONS.items():
        if nom not in X.columns:
            raise DonneeInvalideError(f"Colonne absente des données : {nom}", nom)
        colonne = X[nom]
        commun = (nom, description["libelle"], description["groupe"])
        if description["type"] == "numerique":
            variables.append(VariableNumerique(*commun, int(round(colonne.median())),
                                               int(colonne.min()), int(colonne.max())))
        elif description["type"] == "ordinale":
            valeurs = sorted(int(valeur) for valeur in colonne.unique())
            variables.append(VariableOrdinale(*commun, int(round(colonne.median())), valeurs,
                                              description["valeurs"]))
        else:
            observees = set(colonne.unique())
            connues = [valeur for valeur in description["valeurs"] if valeur in observees]
            valeurs = connues + sorted(str(valeur) for valeur in observees - set(connues))
            variables.append(VariableCategorielle(*commun, str(colonne.mode().iloc[0]), valeurs,
                                                  description["valeurs"]))
    return variables
