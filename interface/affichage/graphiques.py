"""Graphiques propres à l'interface : précision selon les champs, effets, position du risque.

Thème sombre par défaut, pour se fondre dans la fenêtre.
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from model.visualisation.style import couleurs, habiller, legende, libelle_court, nouvelle_figure


def figure_precision_champs(courbe: dict, nb_champs: int | None = None, precision: float | None = None,
                            theme: str = "sombre") -> Figure:
    """Précision attendue selon le nombre de champs renseignés, avec la position de l'utilisateur.

    courbe : dictionnaire « courbe_precision » des métadonnées.
    """
    palette = couleurs(theme)
    bleu, orange = palette["series"][0], palette["series"][1]
    figure = nouvelle_figure(theme)
    axe = figure.add_subplot()
    x = np.asarray(courbe["nb_champs"])
    axe.fill_between(x, np.asarray(courbe["aleatoire_bas"]) * 100, np.asarray(courbe["aleatoire_haut"]) * 100,
                     color=orange, alpha=0.18, linewidth=0, label="Champs au hasard (80 % des cas)")
    axe.plot(x, np.asarray(courbe["aleatoire_moyenne"]) * 100, color=orange, linewidth=2,
             label="Champs au hasard (moyenne)")
    axe.plot(x, np.asarray(courbe["ordre_importance"]) * 100, color=bleu, linewidth=2,
             label="Champs les plus utiles d'abord")
    axe.axhline(50, color=palette["encre_discrete"], linestyle=":", linewidth=1.2)
    axe.text(x[-1], 51, "Hasard (50 %)", ha="right", va="bottom", fontsize=8, color=palette["encre_discrete"])
    if nb_champs is not None and precision is not None:
        axe.plot([nb_champs], [precision * 100], marker="o", markersize=10, color=palette["encre"],
                 markeredgecolor=palette["surface"], markeredgewidth=2, linestyle="none", label="Votre saisie")
        axe.annotate(f"Votre saisie : {precision * 100:.0f} %", (nb_champs, precision * 100),
                     xytext=(8, -16), textcoords="offset points", fontsize=9, color=palette["encre"])
    axe.set_xlim(0, x[-1])
    axe.set_ylim(40, 85)
    habiller(axe, theme, "Plus vous renseignez de champs, plus la prédiction est fiable",
             "Nombre de champs renseignés", "Précision attendue (%)")
    legende(axe, theme, loc="lower right")
    return figure


def figure_effets(effets: dict[str, float], valeurs_affichees: dict[str, str] | None = None, nb: int = 10,
                  message_vide: str = "Aucun champ renseigné", theme: str = "sombre") -> Figure:
    """Barres horizontales : de combien chaque valeur saisie change le risque.

    effets : {nom : effet entre -1 et 1} ; valeurs_affichees : {nom : valeur saisie, en texte}.
    """
    palette = couleurs(theme)
    hausse, baisse = palette["rouge"], palette["series"][0]
    valeurs_affichees = valeurs_affichees or {}
    figure = nouvelle_figure(theme)
    axe = figure.add_subplot()
    principaux = sorted(effets.items(), key=lambda paire: abs(paire[1]), reverse=True)[:nb]
    principaux.reverse()
    habiller(axe, theme, "Ce qui influence ce résultat", "Contribution au risque (points de %)", "")
    if not principaux:
        axe.text(0.5, 0.5, message_vide, ha="center", va="center", transform=axe.transAxes,
                 color=palette["encre_secondaire"])
        axe.set_axis_off()
        return figure
    valeurs = [effet * 100 for _, effet in principaux]
    etiquettes = [f"{libelle_court(nom)} : {valeurs_affichees[nom]}" if nom in valeurs_affichees
                  else libelle_court(nom) for nom, _ in principaux]
    axe.barh(etiquettes, valeurs, height=0.6, color=[hausse if valeur > 0 else baisse for valeur in valeurs])
    axe.axvline(0, color=palette["axe"], linewidth=1)
    legende(axe, theme, loc="lower right", handles=[Patch(color=hausse, label="Augmente le risque"),
                                                    Patch(color=baisse, label="Diminue le risque")])
    axe.grid(axis="y", visible=False)
    return figure


def figure_position_risque(scores: np.ndarray, departs: np.ndarray, probabilite: float | None = None,
                           theme: str = "sombre") -> Figure:
    """Distribution des risques prédits sur le test (restés / partis) et position de l'employé saisi."""
    palette = couleurs(theme)
    figure = nouvelle_figure(theme)
    axe = figure.add_subplot()
    intervalles = np.linspace(0, 1, 21)
    groupes = ((0, palette["series"][0], "Employés restés"), (1, palette["series"][1], "Employés partis"))
    for valeur, couleur, nom in groupes:
        groupe = scores[departs == valeur]
        poids = np.full(len(groupe), 100 / max(len(groupe), 1))
        axe.hist(groupe * 100, bins=intervalles * 100, weights=poids, histtype="stepfilled",
                 color=couleur, alpha=0.25)
        axe.hist(groupe * 100, bins=intervalles * 100, weights=poids, histtype="step",
                 color=couleur, linewidth=2, label=f"{nom} ({len(groupe)})")
    axe.axvline(50, color=palette["encre_discrete"], linestyle=":", linewidth=1.2,
                label="Seuil de décision (50 %)")
    if probabilite is not None:
        axe.axvline(probabilite * 100, color=palette["encre"], linewidth=2,
                    label=f"Employé saisi ({probabilite * 100:.0f} %)")
    habiller(axe, theme, "Où se situe cet employé parmi les employés du test",
             "Risque de départ estimé (%)", "Part du groupe (%)")
    axe.set_xlim(0, 100)
    legende(axe, theme, loc="upper center")
    return figure
