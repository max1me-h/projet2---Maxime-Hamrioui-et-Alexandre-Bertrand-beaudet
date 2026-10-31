"""Graphiques d'exploration et d'évaluation du modèle.

Chaque fonction renvoie un objet Figure : l'entraînement l'enregistre avec savefig (thème
clair) et l'onglet « Le modèle » de l'interface l'affiche (thème sombre).
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from backend.config import COLONNE_CIBLE, DESCRIPTIONS, SATISFACTION, VALEUR_DEPART
from backend.style import (couleurs, degrade, habiller, legende, libelle, libelle_court,
                                       nouvelle_figure, texte_valeur)


def figure_taux_depart(tableau: pd.DataFrame,
                       colonnes: tuple[str, ...] = ("OverTime", "MaritalStatus", "JobLevel", "BusinessTravel"),
                       theme: str = "clair") -> Figure:
    """Exploration : taux de départ selon quatre caractéristiques, comparé au taux global."""
    palette = couleurs(theme)
    figure = nouvelle_figure(theme, 10, 6.5)
    depart = tableau[COLONNE_CIBLE] == VALEUR_DEPART
    taux_global = depart.mean()
    for position, nom in enumerate(colonnes, start=1):
        axe = figure.add_subplot(2, 2, position)
        taux = depart.groupby(tableau[nom]).mean()
        if DESCRIPTIONS[nom]["type"] == "categorielle":
            taux = taux.sort_values(ascending=False)
        etiquettes = [texte_valeur(nom, valeur) for valeur in taux.index]
        barres = axe.bar(etiquettes, taux.to_numpy() * 100, color=palette["series"][0], width=0.6)
        axe.bar_label(barres, fmt="%.0f %%", fontsize=8, color=palette["encre_secondaire"], padding=2)
        axe.axhline(taux_global * 100, color=palette["series"][1], linestyle="--", linewidth=1.5,
                    label=f"Taux global ({taux_global * 100:.0f} %)")
        habiller(axe, theme, libelle(nom), "", "Employés partis (%)")
        legende(axe, theme)
    figure.suptitle("Qui quitte l'entreprise ? Taux de départ par groupe (1470 employés)",
                    fontsize=12, color=palette["encre"])
    return figure


def figure_satisfaction(tableau: pd.DataFrame, theme: str = "clair") -> Figure:
    """Exploration : taux de départ selon le niveau de chacun des indicateurs de satisfaction."""
    palette = couleurs(theme)
    figure = nouvelle_figure(theme, 7, 4.5)
    axe = figure.add_subplot()
    depart = tableau[COLONNE_CIBLE] == VALEUR_DEPART
    noms = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    for couleur, nom in zip(palette["series"], noms):
        taux = depart.groupby(tableau[nom]).mean() * 100
        axe.plot(taux.index, taux.to_numpy(), marker="o", markersize=8, linewidth=2, color=couleur,
                 label=libelle(nom), markeredgecolor=palette["surface"], markeredgewidth=2)
    axe.set_xticks([1, 2, 3, 4], [f"{niveau}\n{texte}" for niveau, texte in SATISFACTION.items()])
    habiller(axe, theme, "Les employés insatisfaits partent plus souvent",
             "Niveau déclaré (1 = le plus bas)", "Employés partis (%)")
    legende(axe, theme)
    return figure


def figure_comparaison_cibles(resultats: dict[str, dict[str, float]], theme: str = "clair") -> Figure:
    """Pourquoi la cible « départ » plutôt que « satisfaction » : exactitude équilibrée sur le test.

    resultats : {nom de la cible : {nom du modèle : exactitude équilibrée}}.
    """
    palette = couleurs(theme)
    figure = nouvelle_figure(theme, 7, 4.2)
    axe = figure.add_subplot()
    cibles = list(resultats)
    modeles = list(next(iter(resultats.values())))
    largeur = 0.8 / len(modeles)
    positions = np.arange(len(cibles))
    for rang, (couleur, modele) in enumerate(zip(palette["series"], modeles)):
        valeurs = [resultats[cible][modele] * 100 for cible in cibles]
        barres = axe.bar(positions + (rang - (len(modeles) - 1) / 2) * largeur, valeurs,
                         width=largeur * 0.92, color=couleur, label=modele)
        axe.bar_label(barres, fmt="%.0f %%", fontsize=8, color=palette["encre_secondaire"], padding=2)
    axe.axhline(50, color=palette["encre_discrete"], linestyle=":", linewidth=1.2)
    axe.set_xticks(positions, cibles)
    axe.set_ylim(0, 100)
    habiller(axe, theme, "La satisfaction n'est pas prévisible, le départ l'est",
             "Cible à prédire", "Exactitude équilibrée sur le test (%)")
    legende(axe, theme, loc="upper left")
    return figure


def figure_matrice_confusion(matrice, titre: str = "Matrice de confusion (jeu de test)",
                             theme: str = "clair") -> Figure:
    """Matrice 2 × 2 : nombre d'employés par couple (réel, prédit), avec le pourcentage par ligne."""
    palette = couleurs(theme)
    matrice = np.asarray(matrice)
    figure = nouvelle_figure(theme, 5.2, 4.2)
    axe = figure.add_subplot()
    axe.imshow(matrice, cmap=degrade(theme), vmin=0)
    totaux = matrice.sum(axis=1, keepdims=True)
    parts = np.divide(matrice, totaux, out=np.zeros(matrice.shape), where=totaux > 0)
    seuil_couleur = matrice.max() / 2
    for (ligne, colonne), valeur in np.ndenumerate(matrice):
        couleur = "white" if valeur > seuil_couleur else palette["encre"]
        axe.text(colonne, ligne, f"{valeur}\n({parts[ligne, colonne] * 100:.0f} % de la ligne)",
                 ha="center", va="center", fontsize=10, color=couleur)
    etiquettes = ["Reste", "Part"]
    axe.set_xticks([0, 1], etiquettes)
    axe.set_yticks([0, 1], etiquettes)
    habiller(axe, theme, titre, "Prédiction du modèle", "Réalité")
    axe.grid(False)
    return figure


def figure_importances(importances: dict[str, float], nb: int = 12, theme: str = "clair") -> Figure:
    """Barres horizontales : les nb variables dont le mélange fait le plus baisser la précision."""
    figure = nouvelle_figure(theme, 6.4, 4.6)
    axe = figure.add_subplot()
    meilleures = sorted(importances.items(), key=lambda paire: paire[1], reverse=True)[:nb]
    meilleures.reverse()
    axe.barh([libelle_court(nom) for nom, _ in meilleures], [valeur * 100 for _, valeur in meilleures],
             color=couleurs(theme)["series"][0], height=0.6)
    habiller(axe, theme, "Variables les plus utiles au modèle",
             "Perte de précision si la variable est mélangée (points de %)", "")
    axe.grid(axis="y", visible=False)
    return figure
