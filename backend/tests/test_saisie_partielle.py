"""Tests de la prédiction avec un formulaire incomplet et de la précision attendue."""

import numpy as np
import pandas as pd
import pytest

from backend.exceptions import DonneeInvalideError
from backend.saisie_partielle import (courbe_precision, courbe_precision_aleatoire, effets_champs,
                                  precision_selon_champs, risques_partiels)


def test_tous_les_champs_connus_donne_le_risque_du_modele(modele_logistique, separation, reference):
    X = separation[1].iloc[:20]
    assert risques_partiels(modele_logistique, X, modele_logistique.colonnes, reference) == pytest.approx(
        modele_logistique.probabilites_depart(X))


def test_aucun_champ_connu_donne_le_meme_risque_a_tous(modele_logistique, separation, reference):
    risques = risques_partiels(modele_logistique, separation[1].iloc[:10], [], reference)
    assert risques == pytest.approx(np.full(10, risques[0]))


def test_le_calcul_vectorise_egale_une_boucle_ecrite_a_la_main(modele_logistique, separation, reference):
    employe = separation[1].iloc[[0]]
    attendu = []
    for _, ligne in reference.iterrows():
        copie = ligne.copy()
        copie[["OverTime", "Age"]] = employe[["OverTime", "Age"]].iloc[0]
        attendu.append(modele_logistique.probabilites_depart(pd.DataFrame([copie]))[0])
    obtenu = risques_partiels(modele_logistique, employe, ["OverTime", "Age"], reference)[0]
    assert obtenu == pytest.approx(np.mean(attendu))


def test_precision_sans_champ_vaut_le_hasard(modele_logistique, separation, reference):
    _, X_test, _, y_test = separation
    assert precision_selon_champs(modele_logistique, X_test, y_test, [], reference) == pytest.approx(0.5)


def test_precision_avec_tous_les_champs_egale_l_evaluation(modele_logistique, separation, reference):
    _, X_test, _, y_test = separation
    precision = precision_selon_champs(modele_logistique, X_test, y_test, modele_logistique.colonnes, reference)
    assert precision == pytest.approx(modele_logistique.evaluer(X_test, y_test)["exactitude_equilibree"])


def test_un_seul_champ_utile_suffit_a_battre_le_hasard(modele_logistique, separation, reference):
    _, X_test, _, y_test = separation
    assert precision_selon_champs(modele_logistique, X_test, y_test, ["OverTime"], reference) > 0.6


def test_courbe_de_precision(modele_logistique, separation, reference):
    _, X_test, _, y_test = separation
    ordre = ["OverTime", "JobRole", "Age"]
    courbe = courbe_precision(modele_logistique, X_test, y_test, reference, ordre)
    assert len(courbe) == 4
    assert courbe[0] == pytest.approx(0.5)
    assert courbe[1] == pytest.approx(
        precision_selon_champs(modele_logistique, X_test, y_test, ["OverTime"], reference))


def test_courbe_aleatoire_reproductible(modele_logistique, separation, reference):
    _, X_test, _, y_test = separation
    premiere = courbe_precision_aleatoire(modele_logistique, X_test.iloc[:60], y_test.iloc[:60], reference, 2, 1)
    seconde = courbe_precision_aleatoire(modele_logistique, X_test.iloc[:60], y_test.iloc[:60], reference, 2, 1)
    assert premiere == seconde
    assert len(premiere["moyenne"]) == len(modele_logistique.colonnes) + 1
    assert all(bas <= haut for bas, haut in zip(premiere["bas"], premiere["haut"]))


def test_courbe_aleatoire_sans_tirage(modele_logistique, separation, reference):
    with pytest.raises(ValueError):
        courbe_precision_aleatoire(modele_logistique, separation[1], separation[3], reference, 0)


def test_les_heures_supplementaires_augmentent_le_risque(modele_logistique, reference):
    effet_oui = effets_champs(modele_logistique, pd.DataFrame([{"OverTime": "Yes"}]), reference)["OverTime"]
    effet_non = effets_champs(modele_logistique, pd.DataFrame([{"OverTime": "No"}]), reference)["OverTime"]
    assert effet_oui > 0 > effet_non


def test_effets_d_un_profil_de_plusieurs_lignes(modele_logistique, separation, reference):
    with pytest.raises(DonneeInvalideError, match="une ligne"):
        effets_champs(modele_logistique, separation[1].iloc[:2], reference)


def test_reference_incomplete(modele_logistique, separation, reference):
    with pytest.raises(DonneeInvalideError, match="Age"):
        risques_partiels(modele_logistique, separation[1].iloc[:2], ["OverTime"], reference.drop(columns=["Age"]))
