"""Fixtures des tests de l'interface : celles du modèle, plus une Application complète."""

import pytest

from frontend.application import Application
from frontend.historique import Historique
from backend.tests.fixtures import jeu, modele_logistique, reference, separation, variables

__all__ = ["jeu", "separation", "variables", "modele_logistique", "reference", "application"]


@pytest.fixture
def application(tmp_path, modele_logistique, variables, separation, reference):
    """Application complète dont l'historique est écrit dans un dossier temporaire."""
    _, X_test, _, y_test = separation
    metadonnees = {"courbe_precision": {"ordre": ["OverTime", "JobRole", "TotalWorkingYears"]}}
    return Application(modele_logistique, variables, X_test, y_test, reference, metadonnees,
                       Historique(str(tmp_path / "historique.csv")))
