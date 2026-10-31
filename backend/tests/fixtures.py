"""Fixtures partagées par backend/tests et frontend/tests (importées par leur conftest.py).

Le vrai jeu de données est chargé et le modèle entraîné une seule fois par session.
"""

import pytest

from backend.entrainement import colonnes_par_encodage
from backend.modeles import ModeleLogistique
from backend.config import CHEMIN_CSV
from backend.jeu_de_donnees import JeuDeDonnees, creer_variables


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
