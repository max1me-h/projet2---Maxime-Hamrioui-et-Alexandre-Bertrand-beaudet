"""Tests de la logique de l'application (ce que l'interface appelle)."""

import json

import pytest

from interface.logique.application import Application, Prediction, pourcentage
from model.config import GROUPES
from model.exceptions import DonneeInvalideError, SaisieInvalideError
from interface.logique.historique import COLONNES_HISTORIQUE, Historique


def test_prediction_valide_et_enregistree(application):
    prediction = application.predire({"Age": "30", "OverTime": "Oui", "MaritalStatus": "Célibataire"})
    assert 0 <= prediction.probabilite <= 1
    assert prediction.nb_champs == 3
    assert prediction.valeurs_connues == {"Age": 30, "OverTime": "Yes", "MaritalStatus": "Single"}
    assert len(application.historique) == 1


def test_formulaire_vide(application):
    with pytest.raises(DonneeInvalideError, match="Aucun champ"):
        application.predire({"Age": "", "OverTime": "— Non renseigné —"})
    assert len(application.historique) == 0


def test_toutes_les_erreurs_sont_reunies(application):
    with pytest.raises(SaisieInvalideError) as information:
        application.predire({"Age": "abc", "MonthlyIncome": "999999", "OverTime": "Peut-être"})
    assert set(information.value.erreurs) == {"Age", "MonthlyIncome", "OverTime"}
    assert len(application.historique) == 0


@pytest.mark.parametrize("saisies, fautif", [
    ({"YearsAtCompany": "20", "TotalWorkingYears": "5"}, "YearsAtCompany"),
    ({"Age": "20", "TotalWorkingYears": "10"}, "TotalWorkingYears"),
    ({"YearsInCurrentRole": "8", "YearsAtCompany": "3"}, "YearsInCurrentRole"),
])
def test_regles_de_coherence(application, saisies, fautif):
    with pytest.raises(SaisieInvalideError) as information:
        application.lire_saisies(saisies)
    assert "Incohérence" in information.value.erreurs[fautif]


def test_champ_inconnu(application):
    with pytest.raises(SaisieInvalideError, match="inconnu"):
        application.lire_saisies({"Salaire": "3000"})


def test_precision_attendue_selon_les_champs(application):
    assert application.precision_attendue([]) == pytest.approx(0.5)
    tous = [variable.nom for variable in application.variables]
    assert application.precision_attendue(tous) > 0.65
    assert application.precision_attendue(["OverTime"]) == application.precision_attendue({"OverTime"})


def test_les_heures_supplementaires_augmentent_le_risque(application):
    oui = application.predire({"OverTime": "Oui"}, enregistrer=False)
    non = application.predire({"OverTime": "Non"}, enregistrer=False)
    assert oui.probabilite > non.probabilite
    assert oui.effets["OverTime"] > 0


def test_exemples_d_employes_reels(application):
    saisies, parti = application.exemple("risque")
    assert parti is True
    assert application.predire(saisies, enregistrer=False).depart_predit
    saisies, parti = application.exemple("stable")
    assert parti is False
    assert not application.predire(saisies, enregistrer=False).depart_predit
    saisies, _ = application.exemple("hasard")
    assert len(saisies) == 26
    with pytest.raises(ValueError):
        application.exemple("inconnu")


def test_champs_conseilles(application):
    conseilles = application.champs_conseilles(["OverTime"], nombre=2)
    assert [variable.nom for variable in conseilles] == ["JobRole", "TotalWorkingYears"]


def test_variables_par_groupe(application):
    groupes = application.variables_par_groupe()
    assert list(groupes) == GROUPES
    assert sum(len(liste) for liste in groupes.values()) == 26


def test_variables_incompatibles_avec_le_modele(modele_logistique, variables, separation, reference, tmp_path):
    _, X_test, _, y_test = separation
    with pytest.raises(DonneeInvalideError, match="ne correspondent pas"):
        Application(modele_logistique, variables[:-1], X_test, y_test, reference, {},
                    Historique(str(tmp_path / "h.csv")))


@pytest.mark.parametrize("probabilite, niveau", [(0.1, "faible"), (0.3, "modéré"), (0.55, "élevé"),
                                                 (0.9, "très élevé")])
def test_niveaux_de_risque(probabilite, niveau):
    assert Prediction(probabilite, {"Age": 30}, 26, 0.6).niveau == niveau


def test_prediction_hors_bornes():
    with pytest.raises(ValueError):
        Prediction(1.2, {}, 26, 0.6)


def test_ligne_d_historique():
    prediction = Prediction(0.684, {"Age": 30, "OverTime": "Yes"}, 26, 0.7)
    ligne = prediction.vers_ligne()
    assert list(ligne) == COLONNES_HISTORIQUE
    assert json.loads(ligne["saisies"]) == {"Age": 30, "OverTime": "Yes"}
    assert ligne["champs_renseignes"] == "2/26"
    assert "Risque de départ élevé (68 %)" in str(prediction)


@pytest.mark.parametrize("valeur, texte", [(0.684, "68 %"), (0.001, "< 1 %"), (0.999, "> 99 %"),
                                           (0, "0 %"), (1, "100 %")])
def test_pourcentage(valeur, texte):
    assert pourcentage(valeur) == texte
