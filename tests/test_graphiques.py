"""Tests rapides des graphiques : chaque fonction renvoie une figure titrée, même sans données."""

import numpy as np
from matplotlib.figure import Figure

from src.graphiques import (figure_comparaison_cibles, figure_effets, figure_importances,
                            figure_matrice_confusion, figure_position_risque, figure_precision_champs,
                            figure_satisfaction, figure_taux_depart)


def _titre(figure: Figure) -> str:
    return figure.axes[0].get_title(loc="left")


def test_graphiques_d_exploration(jeu):
    assert isinstance(figure_taux_depart(jeu.tableau), Figure)
    assert "satisfaits" in _titre(figure_satisfaction(jeu.tableau))


def test_graphiques_du_modele():
    assert _titre(figure_matrice_confusion([[254, 55], [22, 37]])).startswith("Matrice")
    assert _titre(figure_importances({"OverTime": 0.07, "Age": 0.02})) == "Variables les plus utiles au modèle"
    comparaison = {"Satisfaction": {"Référence": 0.5, "Logistique": 0.44},
                   "Départ": {"Référence": 0.5, "Logistique": 0.72}}
    assert len(figure_comparaison_cibles(comparaison).axes[0].patches) == 4


def test_graphiques_de_l_interface():
    courbe = {"nb_champs": [0, 1, 2], "aleatoire_bas": [0.5, 0.5, 0.6], "aleatoire_haut": [0.5, 0.7, 0.7],
              "aleatoire_moyenne": [0.5, 0.6, 0.65], "ordre_importance": [0.5, 0.69, 0.7]}
    assert _titre(figure_precision_champs(courbe, 1, 0.69)).startswith("Plus vous renseignez")
    assert len(figure_effets({"OverTime": 0.2, "Age": -0.05}, {"OverTime": "Oui"}).axes[0].patches) == 2
    vide = figure_effets({}, message_vide="Rien à afficher")
    assert not vide.axes[0].axison
    scores, departs = np.array([0.1, 0.8, 0.4]), np.array([0, 1, 0])
    assert isinstance(figure_position_risque(scores, departs, 0.6), Figure)
