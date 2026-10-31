"""Couleurs et habillage communs à tous les graphiques.

Thème « clair » pour les figures enregistrées (README, présentation) et thème « sombre »
pour l'interface graphique. Les figures sont des objets Figure (sans pyplot).
"""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from backend.config import DESCRIPTIONS

THEMES = {
    "clair": {
        "surface": "#fcfcfb", "encre": "#0b0b0b", "encre_secondaire": "#52514e",
        "encre_discrete": "#898781", "grille": "#e1e0d9", "axe": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"], "rouge": "#e34948",
        "sequentiel": ["#eef4fc", "#184f95"],
    },
    "sombre": {
        "surface": "#161b22", "encre": "#e6edf3", "encre_secondaire": "#9da7b3",
        "encre_discrete": "#6e7681", "grille": "#262c36", "axe": "#3a424e",
        "series": ["#3987e5", "#d95926", "#199e70", "#9085e9"], "rouge": "#e66767",
        "sequentiel": ["#1b2433", "#2f6fc4"],
    },
}


def couleurs(theme: str) -> dict:
    """Renvoie la palette du thème ; lève ValueError si le thème est inconnu."""
    if theme not in THEMES:
        raise ValueError(f"Thème inconnu : {theme!r} (choix : {', '.join(THEMES)})")
    return THEMES[theme]


def nouvelle_figure(theme: str, largeur: float = 6.4, hauteur: float = 4.2) -> Figure:
    """Renvoie une figure vide aux couleurs du thème."""
    return Figure(figsize=(largeur, hauteur), dpi=100, layout="constrained",
                  facecolor=couleurs(theme)["surface"])


def habiller(axe, theme: str, titre: str, etiquette_x: str, etiquette_y: str) -> None:
    """Applique titre, étiquettes d'axes et style discret (grille fine, pas de cadre)."""
    palette = couleurs(theme)
    axe.set_facecolor(palette["surface"])
    axe.set_title(titre, fontsize=11, color=palette["encre"], loc="left")
    axe.set_xlabel(etiquette_x, fontsize=9, color=palette["encre_secondaire"])
    axe.set_ylabel(etiquette_y, fontsize=9, color=palette["encre_secondaire"])
    axe.tick_params(colors=palette["encre_secondaire"], labelsize=8)
    axe.grid(True, color=palette["grille"], linewidth=0.8)
    axe.set_axisbelow(True)
    for cote in ("top", "right"):
        axe.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axe.spines[cote].set_color(palette["axe"])


def legende(axe, theme: str, **options) -> None:
    """Légende sans cadre, au texte lisible dans les deux thèmes."""
    axe.legend(fontsize=8, frameon=False, labelcolor=couleurs(theme)["encre_secondaire"], **options)


def degrade(theme: str) -> LinearSegmentedColormap:
    """Dégradé d'une seule teinte (bleu), du plus faible au plus fort."""
    return LinearSegmentedColormap.from_list(f"bleus_{theme}", couleurs(theme)["sequentiel"])


def libelle(nom: str) -> str:
    """Libellé français d'une colonne (le nom brut si elle n'est pas décrite)."""
    return DESCRIPTIONS.get(nom, {}).get("libelle", nom)


def libelle_court(nom: str) -> str:
    """Libellé sans la précision entre parenthèses (« Âge (ans) » -> « Âge »), pour les axes."""
    return libelle(nom).split(" (")[0]


def texte_valeur(nom: str, valeur: object) -> str:
    """Traduction d'une valeur du jeu de données (« Yes » -> « Oui »)."""
    return str(DESCRIPTIONS.get(nom, {}).get("valeurs", {}).get(valeur, valeur))
