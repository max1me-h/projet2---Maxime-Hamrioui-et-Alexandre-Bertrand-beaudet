"""Tests de l'historique des prédictions (fichier CSV)."""

import pytest

from model.exceptions import DonneeInvalideError
from interface.logique.historique import COLONNES_HISTORIQUE, Historique


def ligne(date: str, risque: str = "0.500") -> dict[str, str]:
    return {"date": date, "risque_depart": risque, "conclusion": "départ probable",
            "champs_renseignes": "3/26", "precision_attendue": "0.680", "saisies": '{"Age": 30}'}


def test_fichier_absent_donne_une_liste_vide(tmp_path):
    historique = Historique(str(tmp_path / "historique.csv"))
    assert historique.lire() == []
    assert len(historique) == 0


def test_ajouter_puis_relire(tmp_path):
    chemin = tmp_path / "historique.csv"
    historique = Historique(str(chemin))
    historique.ajouter(ligne("2026-10-30T10:00:00"))
    historique.ajouter(ligne("2026-10-30T11:00:00", "0.900"))
    assert len(historique) == 2
    assert set(historique.lire()[0]) == set(COLONNES_HISTORIQUE)
    assert len(chemin.read_text(encoding="utf-8").splitlines()) == 3  # l'en-tête n'est écrit qu'une fois


def test_la_plus_recente_en_premier(tmp_path):
    historique = Historique(str(tmp_path / "historique.csv"))
    historique.ajouter(ligne("2026-10-30T09:00:00"))
    historique.ajouter(ligne("2026-10-31T08:00:00"))
    historique.ajouter(ligne("2026-10-30T12:00:00"))
    assert [entree["date"][:10] for entree in historique.lire()] == ["2026-10-31", "2026-10-30", "2026-10-30"]


def test_ligne_incomplete_refusee(tmp_path):
    historique = Historique(str(tmp_path / "historique.csv"))
    with pytest.raises(DonneeInvalideError, match="saisies"):
        historique.ajouter({"date": "2026-10-30"})


def test_cree_le_dossier_au_besoin(tmp_path):
    historique = Historique(str(tmp_path / "nouveau" / "historique.csv"))
    historique.ajouter(ligne("2026-10-30T10:00:00"))
    assert len(historique) == 1


def test_effacer(tmp_path):
    historique = Historique(str(tmp_path / "historique.csv"))
    historique.ajouter(ligne("2026-10-30T10:00:00"))
    historique.effacer()
    assert historique.lire() == []
    historique.effacer()  # effacer un historique absent ne plante pas


def test_fichier_au_mauvais_format(tmp_path):
    chemin = tmp_path / "historique.csv"
    chemin.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DonneeInvalideError, match="colonnes"):
        Historique(str(chemin)).lire()
