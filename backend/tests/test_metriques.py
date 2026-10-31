"""Tests des métriques NumPy, comparées à celles de scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from backend.metriques import (exactitude, exactitude_equilibree, matrice_confusion, precision_depart, rappel,
                           resume_metriques, score_f1)


@pytest.fixture
def etiquettes():
    """500 étiquettes réelles (20 % de départs) et des prédictions justes à environ 70 %."""
    generateur = np.random.default_rng(0)
    y_vrai = (generateur.random(500) < 0.2).astype(int)
    y_predit = np.where(generateur.random(500) < 0.7, y_vrai, 1 - y_vrai)
    return y_vrai, y_predit


def test_matrice_sur_un_petit_exemple_calcule_a_la_main():
    # Réel : 0 0 1 1 1 ; prédit : 0 1 1 0 1 -> VN=1, FP=1, FN=1, VP=2
    assert matrice_confusion([0, 0, 1, 1, 1], [0, 1, 1, 0, 1]).tolist() == [[1, 1], [1, 2]]


def test_concordance_avec_scikit_learn(etiquettes):
    y_vrai, y_predit = etiquettes
    assert np.array_equal(matrice_confusion(y_vrai, y_predit), confusion_matrix(y_vrai, y_predit))
    assert exactitude(y_vrai, y_predit) == pytest.approx(accuracy_score(y_vrai, y_predit))
    assert exactitude_equilibree(y_vrai, y_predit) == pytest.approx(balanced_accuracy_score(y_vrai, y_predit))
    assert rappel(y_vrai, y_predit) == pytest.approx(recall_score(y_vrai, y_predit))
    assert precision_depart(y_vrai, y_predit) == pytest.approx(precision_score(y_vrai, y_predit))
    assert score_f1(y_vrai, y_predit) == pytest.approx(f1_score(y_vrai, y_predit))


def test_un_modele_constant_obtient_un_demi():
    y_vrai = np.array([0] * 90 + [1] * 10)
    toujours_reste = np.zeros(100, dtype=int)
    assert exactitude(y_vrai, toujours_reste) == pytest.approx(0.9)
    assert exactitude_equilibree(y_vrai, toujours_reste) == pytest.approx(0.5)


def test_aucun_depart_predit_ne_divise_pas_par_zero():
    resultats = resume_metriques([0, 1, 1], [0, 0, 0])
    assert resultats["precision_depart"] == 0.0
    assert resultats["f1_depart"] == 0.0
    assert resultats["rappel_depart"] == 0.0


def test_une_seule_classe_presente():
    assert exactitude_equilibree([0, 0, 0], [0, 0, 1]) == pytest.approx(2 / 3)


def test_tailles_differentes_refusees():
    with pytest.raises(ValueError, match="Tailles"):
        exactitude([0, 1, 1], [0, 1])


def test_entree_vide_refusee():
    with pytest.raises(ValueError, match="Aucune"):
        exactitude([], [])


def test_etiquettes_autres_que_0_et_1_refusees():
    with pytest.raises(ValueError, match="0 ou 1"):
        matrice_confusion([0, 2], [0, 1])
