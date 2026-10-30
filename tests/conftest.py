"""Objets partagés par les tests : le vrai jeu de données est chargé et le modèle entraîné une seule fois."""

import pytest

from src.application import Application
from src.config import CHEMIN_CSV
from src.donnees import JeuDeDonnees, creer_variables
from src.entrainement import colonnes_par_encodage
from src.historique import Historique
from src.modele import ModeleLogistique


@pytest.fixture(scope="session")
def jeu():
    """Jeu de données IBM chargé et nettoyé."""
    jeu = JeuDeDonnees(CHEMIN_CSV).charger()
    jeu.nettoyer()
    return jeu


@pytest.fixture(scope="session")
def separation(jeu):
    """(X_entrainement, X_test, y_entrainement, y_test) avec la graine du projet."""
    return jeu.separer()


@pytest.fixture(scope="session")
def variables(separation):
    """Les 26 variables du formulaire, construites sur l'entraînement."""
    return creer_variables(separation[0])


@pytest.fixture(scope="session")
def modele_logistique(separation, variables):
    """Régression logistique entraînée sur l'ensemble d'entraînement."""
    numeriques, categorielles = colonnes_par_encodage(variables)
    X_entr, _, y_entr, _ = separation
    return ModeleLogistique(numeriques, categorielles).entrainer(X_entr, y_entr)


@pytest.fixture(scope="session")
def reference(separation):
    """40 employés de référence tirés de l'entraînement."""
    return separation[0].sample(n=40, random_state=0)


@pytest.fixture
def application(tmp_path, modele_logistique, variables, separation, reference):
    """Application complète dont l'historique est écrit dans un dossier temporaire."""
    _, X_test, _, y_test = separation
    metadonnees = {"courbe_precision": {"ordre": ["OverTime", "JobRole", "TotalWorkingYears"]}}
    return Application(modele_logistique, variables, X_test, y_test, reference, metadonnees,
                       Historique(str(tmp_path / "historique.csv")))
