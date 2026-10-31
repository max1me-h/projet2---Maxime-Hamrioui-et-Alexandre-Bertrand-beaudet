"""Tests rapides des graphiques de l'interface (thème sombre par défaut)."""

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from interface.affichage.graphiques import figure_effets, figure_position_risque, figure_precision_champs
from model.visualisation.style import couleurs


def test_courbe_de_precision_avec_la_saisie():
    courbe = {"nb_champs": [0, 1, 2], "aleatoire_bas": [0.5, 0.5, 0.6], "aleatoire_haut": [0.5, 0.7, 0.7],
              "aleatoire_moyenne": [0.5, 0.6, 0.65], "ordre_importance": [0.5, 0.69, 0.7]}
    figure = figure_precision_champs(courbe, 1, 0.69)
    assert figure.axes[0].get_title(loc="left").startswith("Plus vous renseignez")
    assert to_hex(figure.get_facecolor()) == couleurs("sombre")["surface"]


def test_effets_et_message_sans_donnees():
    assert len(figure_effets({"OverTime": 0.2, "Age": -0.05}, {"OverTime": "Oui"}).axes[0].patches) == 2
    vide = figure_effets({}, message_vide="Rien à afficher")
    assert not vide.axes[0].axison


def test_position_du_risque():
    scores, departs = np.array([0.1, 0.8, 0.4]), np.array([0, 1, 0])
    assert isinstance(figure_position_risque(scores, departs, 0.6), Figure)
    assert isinstance(figure_position_risque(scores, departs), Figure)
