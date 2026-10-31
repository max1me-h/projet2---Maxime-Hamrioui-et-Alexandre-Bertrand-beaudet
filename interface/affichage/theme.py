"""Palette sombre et polices de l'interface.

Les couleurs des cartes (CARTE) sont les mêmes que le fond des graphiques sombres
(model/visualisation/style.py) : les figures se fondent dans la fenêtre.
"""

from tkinter import ttk

import customtkinter as ctk

POLICE = "Segoe UI"

FOND = "#0d1117"
BARRE_LATERALE = "#010409"
CARTE = "#161b22"
CARTE_RELIEF = "#1c2128"
SAISIE = "#0d1117"
BORDURE = "#30363d"
SURVOL = "#262c36"
TEXTE = "#e6edf3"
TEXTE_SECONDAIRE = "#9da7b3"
TEXTE_DISCRET = "#6e7681"
ACCENT = "#2f81f7"
ACCENT_SURVOL = "#1f6feb"
ACCENT_DOUX = "#13233a"
SUCCES = "#3fb950"
ERREUR = "#f85149"
ERREUR_FOND = "#2d1416"
AVERTISSEMENT = "#e3b341"
AVERTISSEMENT_FOND = "#272011"

# Niveau de risque -> (couleur du texte et de la barre, fond du badge).
NIVEAUX = {
    "faible": ("#3fb950", "#12261a"),
    "modéré": ("#d29922", "#2b2310"),
    "élevé": ("#f0883e", "#2d1d10"),
    "très élevé": ("#f85149", "#2d1416"),
}


def police(taille: int = 13, gras: bool = False) -> ctk.CTkFont:
    """Police de l'interface (taille en pixels avant mise à l'échelle de l'écran)."""
    return ctk.CTkFont(family=POLICE, size=taille, weight="bold" if gras else "normal")


def configurer_tableaux(racine: ctk.CTk, echelle: float) -> None:
    """Style sombre des tableaux ttk.Treeview (CustomTkinter n'a pas de tableau)."""
    style = ttk.Style(racine)
    style.theme_use("clam")
    style.configure("Sombre.Treeview", background=CARTE, fieldbackground=CARTE, foreground=TEXTE,
                    bordercolor=CARTE, lightcolor=CARTE, darkcolor=CARTE, borderwidth=0,
                    rowheight=int(34 * echelle), font=(POLICE, 10))
    style.configure("Sombre.Treeview.Heading", background=CARTE_RELIEF, foreground=TEXTE_SECONDAIRE,
                    bordercolor=CARTE_RELIEF, lightcolor=CARTE_RELIEF, darkcolor=CARTE_RELIEF,
                    relief="flat", font=(POLICE, 10, "bold"), padding=(8, 8))
    style.map("Sombre.Treeview", background=[("selected", ACCENT_DOUX)], foreground=[("selected", TEXTE)])
    style.map("Sombre.Treeview.Heading", background=[("active", SURVOL)])
    style.layout("Sombre.Treeview", [("Sombre.Treeview.treearea", {"sticky": "nswe"})])
