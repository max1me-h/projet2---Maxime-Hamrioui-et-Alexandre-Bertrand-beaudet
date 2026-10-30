"""Tests des classes de variables : chaque type valide la saisie à sa façon."""

import json

import pytest

from src.exceptions import DonneeInvalideError
from src.variables import (NON_RENSEIGNE, VariableCategorielle, VariableNumerique, VariableOrdinale,
                           variable_depuis_dict)


@pytest.fixture
def age():
    return VariableNumerique("Age", "Âge (ans)", "Profil personnel", 36, 18, 60)


@pytest.fixture
def heures_sup():
    return VariableCategorielle("OverTime", "Heures supplémentaires", "Poste", "No", ["Yes", "No"],
                                {"Yes": "Oui", "No": "Non"})


@pytest.fixture
def satisfaction():
    return VariableOrdinale("JobSatisfaction", "Satisfaction", "Satisfaction et équilibre", 3,
                            [1, 2, 3, 4], {1: "Faible", 4: "Très élevée"})


def test_numerique_accepte_un_entier(age):
    valeur = age.valider(" 42 ")
    assert valeur == 42
    assert isinstance(valeur, int)


def test_numerique_accepte_espaces_et_virgule():
    salaire = VariableNumerique("MonthlyIncome", "Salaire", "Rémunération", 5000, 1000, 20000, entier=False)
    assert salaire.valider("5 000,5") == pytest.approx(5000.5)


def test_numerique_bornes_incluses(age):
    assert age.valider("18") == 18
    assert age.valider("60") == 60


@pytest.mark.parametrize("saisie", ["abc", "12a", "nan", "inf", ""])
def test_numerique_refuse_ce_qui_n_est_pas_un_nombre(age, saisie):
    with pytest.raises(DonneeInvalideError, match="Âge"):
        age.valider(saisie)


@pytest.mark.parametrize("saisie", ["17", "61", "-5"])
def test_numerique_refuse_hors_intervalle(age, saisie):
    with pytest.raises(DonneeInvalideError, match="entre 18 et 60"):
        age.valider(saisie)


def test_numerique_refuse_un_decimal_si_entier_attendu(age):
    with pytest.raises(DonneeInvalideError, match="entier"):
        age.valider("36,5")


def test_bornes_incoherentes_refusees():
    with pytest.raises(ValueError):
        VariableNumerique("Age", "Âge", "Profil", 30, 60, 18)


def test_erreur_indique_la_variable_fautive(age):
    with pytest.raises(DonneeInvalideError) as information:
        age.valider("vingt")
    assert information.value.variable == "Age"


def test_champ_vide_signifie_non_renseigne(age, heures_sup):
    assert age.lire("   ") is None
    assert age.lire(None) is None
    assert heures_sup.lire(NON_RENSEIGNE) is None


def test_categorielle_accepte_libelle_et_valeur_brute(heures_sup):
    assert heures_sup.valider("Oui") == "Yes"
    assert heures_sup.valider("No") == "No"


def test_categorielle_refuse_une_valeur_inconnue(heures_sup):
    with pytest.raises(DonneeInvalideError, match="choix possibles"):
        heures_sup.valider("Peut-être")


def test_ordinale_renvoie_un_entier_et_formate_son_libelle(satisfaction):
    assert satisfaction.valider("4 — Très élevée") == 4
    assert satisfaction.valider("2") == 2
    assert satisfaction.formater(1) == "1 — Faible"
    assert satisfaction.formater(2) == "2"


def test_polymorphisme_de_valider(age, heures_sup, satisfaction):
    cas = [(age, "30", 30), (heures_sup, "Non", "No"), (satisfaction, "3", 3)]
    for variable, saisie, attendu in cas:
        assert variable.valider(saisie) == attendu


def test_aller_retour_json(age, heures_sup, satisfaction):
    for variable in (age, heures_sup, satisfaction):
        copie = variable_depuis_dict(json.loads(json.dumps(variable.vers_dict())))
        assert copie == variable
        assert type(copie) is type(variable)


def test_description_json_invalide():
    with pytest.raises(DonneeInvalideError, match="inconnu"):
        variable_depuis_dict({"type": "date", "nom": "x", "libelle": "x", "groupe": "g", "valeur_typique": 1})
    with pytest.raises(DonneeInvalideError, match="incomplète"):
        variable_depuis_dict({"type": "numerique", "nom": "x"})


def test_str_et_repr(age):
    assert str(age) == "Âge (ans) (de 18 à 60)"
    assert repr(age) == "VariableNumerique(nom='Age')"
