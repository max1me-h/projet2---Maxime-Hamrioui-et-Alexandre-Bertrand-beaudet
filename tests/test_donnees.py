"""Tests du chargement, du nettoyage et de la séparation des données."""

import numpy as np
import pandas as pd
import pytest

from src.config import CHEMIN_CSV, COLONNES_RETIREES, DESCRIPTIONS, PROPORTION_TEST
from src.donnees import (JeuDeDonnees, compter_valeurs_aberrantes, creer_variables, ecrire_json, lire_json,
                         lire_par_blocs)
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.variables import VariableCategorielle, VariableNumerique, VariableOrdinale


def test_chargement_du_vrai_fichier(jeu):
    assert len(jeu) == 1470
    assert jeu.taux_depart() == pytest.approx(237 / 1470)


def test_nettoyage_garde_les_colonnes_decrites(jeu):
    for nom in COLONNES_RETIREES:
        assert nom not in jeu.tableau.columns
    assert list(jeu.caracteristiques.columns) == list(DESCRIPTIONS)


def test_generateur_lit_par_blocs():
    tailles = [len(bloc) for bloc in lire_par_blocs(CHEMIN_CSV, 500)]
    assert tailles == [500, 500, 470]


def test_generateur_retire_le_caractere_invisible_du_debut():
    premier_bloc = next(lire_par_blocs(CHEMIN_CSV, 10))
    assert premier_bloc.columns[0] == "Age"


def test_taille_de_bloc_invalide():
    with pytest.raises(ValueError):
        next(lire_par_blocs(CHEMIN_CSV, 0))


def test_fichier_absent(tmp_path):
    with pytest.raises(FichierIntrouvableError, match="absent.csv"):
        JeuDeDonnees(str(tmp_path / "absent.csv")).charger()


def test_fichier_vide(tmp_path):
    chemin = tmp_path / "vide.csv"
    chemin.write_text("", encoding="utf-8")
    with pytest.raises(DonneeInvalideError, match="vide"):
        JeuDeDonnees(str(chemin)).charger()


def test_nettoyer_avant_charger():
    with pytest.raises(DonneeInvalideError, match="charger"):
        JeuDeDonnees(CHEMIN_CSV).nettoyer()


def test_colonne_manquante(tmp_path):
    brut = pd.read_csv(CHEMIN_CSV, encoding="utf-8-sig", nrows=5).drop(columns=["OverTime"])
    chemin = tmp_path / "incomplet.csv"
    brut.to_csv(chemin, index=False)
    with pytest.raises(DonneeInvalideError, match="OverTime"):
        JeuDeDonnees(str(chemin)).charger().nettoyer()


def test_nettoyage_retire_doublons_textes_et_cibles_invalides(tmp_path):
    brut = pd.read_csv(CHEMIN_CSV, encoding="utf-8-sig", nrows=6)
    brut = pd.concat([brut, brut.iloc[[0]]], ignore_index=True)  # un doublon exact
    brut["Age"] = brut["Age"].astype(object)
    brut.loc[1, "Age"] = "abc"                                   # une valeur non numérique
    brut.loc[2, "Attrition"] = "Maybe"                           # une cible invalide
    chemin = tmp_path / "sale.csv"
    brut.to_csv(chemin, index=False)
    jeu = JeuDeDonnees(str(chemin)).charger()
    rapport = jeu.nettoyer()
    assert len(jeu) == 4
    assert "Doublons exacts retirés : 1." in rapport
    assert "Valeurs non numériques dans une colonne numérique : 1." in rapport
    assert jeu.tableau["Age"].dtype.kind == "i"


def test_separation_stratifiee_et_reproductible(jeu):
    _, X_test_1, _, y_test_1 = jeu.separer()
    _, X_test_2, _, _ = jeu.separer()
    assert X_test_1.index.equals(X_test_2.index)
    assert len(X_test_1) == int(np.ceil(PROPORTION_TEST * 1470))
    assert y_test_1.mean() == pytest.approx(jeu.taux_depart(), abs=0.01)


@pytest.mark.parametrize("proportion", [0, 1, 1.5])
def test_separation_proportion_invalide(jeu, proportion):
    with pytest.raises(ValueError):
        jeu.separer(proportion)


def test_valeurs_aberrantes():
    assert compter_valeurs_aberrantes(np.array([1, 2, 3, 4, 5, 100])) == 1
    assert compter_valeurs_aberrantes(np.array([])) == 0


def test_creer_variables_depuis_l_entrainement(separation):
    X_entr = separation[0]
    variables = {variable.nom: variable for variable in creer_variables(X_entr)}
    assert len(variables) == len(DESCRIPTIONS)
    assert isinstance(variables["Age"], VariableNumerique)
    assert variables["Age"].minimum == X_entr["Age"].min()
    assert isinstance(variables["JobSatisfaction"], VariableOrdinale)
    assert isinstance(variables["OverTime"], VariableCategorielle)
    assert "Oui" in variables["OverTime"].choix()


def test_creer_variables_colonne_absente(separation):
    with pytest.raises(DonneeInvalideError, match="Age"):
        creer_variables(separation[0].drop(columns=["Age"]))


def test_json_aller_retour_et_fichier_corrompu(tmp_path):
    chemin = str(tmp_path / "sous_dossier" / "meta.json")
    ecrire_json(chemin, {"modèle": "logistique", "valeurs": [1, 2]})
    assert lire_json(chemin) == {"modèle": "logistique", "valeurs": [1, 2]}
    (tmp_path / "corrompu.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(DonneeInvalideError, match="illisible"):
        lire_json(str(tmp_path / "corrompu.json"))
