"""Fixtures partagées par model/tests et interface/tests (importées par leur conftest.py).

Le vrai jeu de données est chargé et le modèle entraîné une seule fois par session.
"""

import pytest

from model.apprentissage.entrainement import colonnes_par_encodage
from model.apprentissage.modeles import ModeleLogistique
from model.config import CHEMIN_CSV
from model.preparation.donnees import JeuDeDonnees, creer_variables


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
