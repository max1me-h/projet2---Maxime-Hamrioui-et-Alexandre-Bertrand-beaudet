"""Graphiques Matplotlib du projet.

Chaque fonction renvoie un objet Figure (sans passer par pyplot) : l'interface Tkinter
l'intègre dans sa fenêtre et le script d'entraînement l'enregistre avec savefig.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.config import COLONNE_CIBLE, DESCRIPTIONS, SATISFACTION, VALEUR_DEPART

SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"
BLEU = "#2a78d6"
ORANGE = "#eb6834"
ROUGE = "#e34948"
SERIES = [BLEU, ORANGE, "#1baf7a", "#6250d6"]
BLEUS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def _nouvelle_figure(largeur: float = 6.4, hauteur: float = 4.2) -> Figure:
    """Renvoie une figure vide aux couleurs du projet."""
    return Figure(figsize=(largeur, hauteur), dpi=100, layout="constrained", facecolor=SURFACE)


def _habiller(axe, titre: str, etiquette_x: str, etiquette_y: str) -> None:
    """Applique titre, étiquettes d'axes et style discret (grille fine, pas de cadre)."""
    axe.set_facecolor(SURFACE)
    axe.set_title(titre, fontsize=11, color=ENCRE, loc="left")
    axe.set_xlabel(etiquette_x, fontsize=9, color=ENCRE_SECONDAIRE)
    axe.set_ylabel(etiquette_y, fontsize=9, color=ENCRE_SECONDAIRE)
    axe.tick_params(colors=ENCRE_SECONDAIRE, labelsize=8)
    axe.grid(True, color=GRILLE, linewidth=0.8)
    axe.set_axisbelow(True)
    for cote in ("top", "right"):
        axe.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axe.spines[cote].set_color(AXE)


def _libelle(nom: str) -> str:
    """Libellé français d'une colonne (le nom brut si elle n'est pas décrite)."""
    return DESCRIPTIONS.get(nom, {}).get("libelle", nom)


def _libelle_court(nom: str) -> str:
    """Libellé sans la précision entre parenthèses (« Âge (ans) » -> « Âge »), pour les axes."""
    return _libelle(nom).split(" (")[0]


def _texte_valeur(nom: str, valeur: object) -> str:
    """Traduction d'une valeur du jeu de données (« Yes » -> « Oui »)."""
    return str(DESCRIPTIONS.get(nom, {}).get("valeurs", {}).get(valeur, valeur))


def figure_taux_depart(tableau: pd.DataFrame,
                       colonnes: tuple[str, ...] = ("OverTime", "MaritalStatus",
                                                    "JobLevel", "BusinessTravel")) -> Figure:
    """Exploration : taux de départ selon quatre caractéristiques, comparé au taux global."""
    figure = _nouvelle_figure(10, 6.5)
    depart = tableau[COLONNE_CIBLE] == VALEUR_DEPART
    taux_global = depart.mean()
    for position, nom in enumerate(colonnes, start=1):
        axe = figure.add_subplot(2, 2, position)
        taux = depart.groupby(tableau[nom]).mean()
        if DESCRIPTIONS[nom]["type"] == "categorielle":
            taux = taux.sort_values(ascending=False)
        etiquettes = [_texte_valeur(nom, valeur) for valeur in taux.index]
        barres = axe.bar(etiquettes, taux.to_numpy() * 100, color=BLEU, width=0.6)
        axe.bar_label(barres, fmt="%.0f %%", fontsize=8, color=ENCRE_SECONDAIRE, padding=2)
        axe.axhline(taux_global * 100, color=ORANGE, linestyle="--", linewidth=1.5,
                    label=f"Taux global ({taux_global * 100:.0f} %)")
        _habiller(axe, _libelle(nom), "", "Employés partis (%)")
        axe.legend(fontsize=8, frameon=False)
    figure.suptitle("Qui quitte l'entreprise ? Taux de départ par groupe (1470 employés)",
                    fontsize=12, color=ENCRE)
    return figure


def figure_satisfaction(tableau: pd.DataFrame) -> Figure:
    """Exploration : taux de départ selon le niveau de chacun des indicateurs de satisfaction."""
    figure = _nouvelle_figure(7, 4.5)
    axe = figure.add_subplot()
    depart = tableau[COLONNE_CIBLE] == VALEUR_DEPART
    noms = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    for couleur, nom in zip(SERIES, noms):
        taux = depart.groupby(tableau[nom]).mean() * 100
        axe.plot(taux.index, taux.to_numpy(), marker="o", markersize=8, linewidth=2,
                 color=couleur, label=_libelle(nom), markeredgecolor=SURFACE, markeredgewidth=2)
    axe.set_xticks([1, 2, 3, 4], [f"{niveau}\n{texte}" for niveau, texte in SATISFACTION.items()])
    _habiller(axe, "Les employés insatisfaits partent plus souvent",
              "Niveau déclaré (1 = le plus bas)", "Employés partis (%)")
    axe.legend(fontsize=8, frameon=False)
    return figure


def figure_comparaison_cibles(resultats: dict[str, dict[str, float]]) -> Figure:
    """Pourquoi la cible « départ » plutôt que « satisfaction » : exactitude équilibrée sur le test.

    resultats : {nom de la cible : {nom du modèle : exactitude équilibrée}}.
    """
    figure = _nouvelle_figure(7, 4.2)
    axe = figure.add_subplot()
    cibles = list(resultats)
    modeles = list(next(iter(resultats.values())))
    largeur = 0.8 / len(modeles)
    positions = np.arange(len(cibles))
    for rang, (couleur, modele) in enumerate(zip(SERIES, modeles)):
        valeurs = [resultats[cible][modele] * 100 for cible in cibles]
        barres = axe.bar(positions + (rang - (len(modeles) - 1) / 2) * largeur, valeurs,
                         width=largeur * 0.92, color=couleur, label=modele)
        axe.bar_label(barres, fmt="%.0f %%", fontsize=8, color=ENCRE_SECONDAIRE, padding=2)
    axe.axhline(50, color=ENCRE_DISCRETE, linestyle=":", linewidth=1.2)
    axe.set_xticks(positions, cibles)
    axe.set_ylim(0, 100)
    _habiller(axe, "La satisfaction n'est pas prévisible, le départ l'est",
              "Cible à prédire", "Exactitude équilibrée sur le test (%)")
    axe.legend(fontsize=8, frameon=False, loc="upper left")
    return figure


def figure_matrice_confusion(matrice, titre: str = "Matrice de confusion (jeu de test)") -> Figure:
    """Matrice 2 × 2 : nombre d'employés par couple (réel, prédit), avec le pourcentage par ligne."""
    matrice = np.asarray(matrice)
    figure = _nouvelle_figure(5.2, 4.2)
    axe = figure.add_subplot()
    axe.imshow(matrice, cmap="Blues", vmin=0)
    totaux = matrice.sum(axis=1, keepdims=True)
    parts = np.divide(matrice, totaux, out=np.zeros(matrice.shape), where=totaux > 0)
    seuil_couleur = matrice.max() / 2
    for (ligne, colonne), valeur in np.ndenumerate(matrice):
        couleur = "white" if valeur > seuil_couleur else ENCRE
        axe.text(colonne, ligne, f"{valeur}\n({parts[ligne, colonne] * 100:.0f} % de la ligne)",
                 ha="center", va="center", fontsize=10, color=couleur)
    etiquettes = ["Reste", "Part"]
    axe.set_xticks([0, 1], etiquettes)
    axe.set_yticks([0, 1], etiquettes)
    _habiller(axe, titre, "Prédiction du modèle", "Réalité")
    axe.grid(False)
    return figure


def figure_importances(importances: dict[str, float], nb: int = 12) -> Figure:
    """Barres horizontales : les nb variables dont le mélange fait le plus baisser la précision."""
    figure = _nouvelle_figure(6.4, 4.6)
    axe = figure.add_subplot()
    meilleures = sorted(importances.items(), key=lambda paire: paire[1], reverse=True)[:nb]
    meilleures.reverse()
    axe.barh([_libelle_court(nom) for nom, _ in meilleures], [valeur * 100 for _, valeur in meilleures],
             color=BLEU, height=0.6)
    _habiller(axe, "Variables les plus utiles au modèle",
              "Perte de précision si la variable est mélangée (points de %)", "")
    axe.grid(axis="y", visible=False)
    return figure


def figure_precision_champs(courbe: dict, nb_champs: int | None = None,
                            precision: float | None = None) -> Figure:
    """Précision attendue selon le nombre de champs renseignés, avec la position de l'utilisateur.

    courbe : dictionnaire « courbe_precision » des métadonnées.
    """
    figure = _nouvelle_figure(6.4, 4.2)
    axe = figure.add_subplot()
    x = np.asarray(courbe["nb_champs"])
    axe.fill_between(x, np.asarray(courbe["aleatoire_bas"]) * 100, np.asarray(courbe["aleatoire_haut"]) * 100,
                     color=ORANGE, alpha=0.15, linewidth=0, label="Champs au hasard (80 % des cas)")
    axe.plot(x, np.asarray(courbe["aleatoire_moyenne"]) * 100, color=ORANGE, linewidth=2,
             label="Champs au hasard (moyenne)")
    axe.plot(x, np.asarray(courbe["ordre_importance"]) * 100, color=BLEU, linewidth=2,
             label="Champs les plus utiles d'abord")
    axe.axhline(50, color=ENCRE_DISCRETE, linestyle=":", linewidth=1.2)
    axe.text(x[-1], 51, "Hasard (50 %)", ha="right", va="bottom", fontsize=8, color=ENCRE_DISCRETE)
    if nb_champs is not None and precision is not None:
        axe.plot([nb_champs], [precision * 100], marker="o", markersize=10, color=ENCRE,
                 markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label="Votre saisie")
        axe.annotate(f"Votre saisie : {precision * 100:.0f} %", (nb_champs, precision * 100),
                     xytext=(8, -16), textcoords="offset points", fontsize=9, color=ENCRE)
    axe.set_xlim(0, x[-1])
    axe.set_ylim(40, 85)
    _habiller(axe, "Plus vous renseignez de champs, plus la prédiction est fiable",
              "Nombre de champs renseignés", "Précision attendue (%)")
    axe.legend(fontsize=8, frameon=False, loc="lower right")
    return figure


def figure_effets(effets: dict[str, float], valeurs_affichees: dict[str, str] | None = None,
                  nb: int = 10, message_vide: str = "Aucun champ renseigné") -> Figure:
    """Barres horizontales : de combien chaque valeur saisie change le risque.

    effets : {nom : effet entre -1 et 1} ; valeurs_affichees : {nom : valeur saisie, en texte}.
    """
    valeurs_affichees = valeurs_affichees or {}
    figure = _nouvelle_figure(6.4, 4.2)
    axe = figure.add_subplot()
    principaux = sorted(effets.items(), key=lambda paire: abs(paire[1]), reverse=True)[:nb]
    principaux.reverse()
    _habiller(axe, "Ce qui influence ce résultat", "Contribution au risque (points de %)", "")
    if not principaux:
        axe.text(0.5, 0.5, message_vide, ha="center", va="center",
                 transform=axe.transAxes, color=ENCRE_SECONDAIRE)
        axe.set_axis_off()
        return figure
    valeurs = [effet * 100 for _, effet in principaux]
    etiquettes = [f"{_libelle_court(nom)} : {valeurs_affichees[nom]}" if nom in valeurs_affichees
                  else _libelle_court(nom) for nom, _ in principaux]
    axe.barh(etiquettes, valeurs, height=0.6, color=[ROUGE if valeur > 0 else BLEU for valeur in valeurs])
    axe.axvline(0, color=AXE, linewidth=1)
    axe.legend(handles=[Patch(color=ROUGE, label="Augmente le risque"),
                        Patch(color=BLEU, label="Diminue le risque")],
               fontsize=8, frameon=False, loc="lower right")
    axe.grid(axis="y", visible=False)
    return figure


def figure_position_risque(scores: np.ndarray, departs: np.ndarray,
                           probabilite: float | None = None) -> Figure:
    """Distribution des risques prédits sur le test (restés / partis) et position de l'employé saisi, s'il y en a un."""
    figure = _nouvelle_figure(6.4, 4.2)
    axe = figure.add_subplot()
    intervalles = np.linspace(0, 1, 21)
    for valeur, couleur, nom in ((0, BLEU, "Employés restés"), (1, ORANGE, "Employés partis")):
        groupe = scores[departs == valeur]
        poids = np.full(len(groupe), 100 / max(len(groupe), 1))
        axe.hist(groupe * 100, bins=intervalles * 100, weights=poids, histtype="stepfilled",
                 color=couleur, alpha=0.25)
        axe.hist(groupe * 100, bins=intervalles * 100, weights=poids, histtype="step",
                 color=couleur, linewidth=2, label=f"{nom} ({len(groupe)})")
    axe.axvline(50, color=ENCRE_DISCRETE, linestyle=":", linewidth=1.2, label="Seuil de décision (50 %)")
    if probabilite is not None:
        axe.axvline(probabilite * 100, color=ENCRE, linewidth=2,
                    label=f"Employé saisi ({probabilite * 100:.0f} %)")
    _habiller(axe, "Où se situe cet employé parmi les employés du test",
              "Risque de départ estimé (%)", "Part du groupe (%)")
    axe.set_xlim(0, 100)
    axe.legend(fontsize=8, frameon=False, loc="upper center")
    return figure
