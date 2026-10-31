"""Composants réutilisables de l'interface : cartes, tuiles, boutons, champs, zone de graphique."""

from collections.abc import Callable

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from interface.affichage.theme import (ACCENT, ACCENT_SURVOL, BORDURE, CARTE, ERREUR, ERREUR_FOND, SAISIE,
                                       SURVOL, TEXTE, TEXTE_DISCRET, TEXTE_SECONDAIRE, police)
from model.exceptions import DonneeInvalideError
from model.preparation.variables import NON_RENSEIGNE, Variable, VariableCategorielle, VariableNumerique


class Carte(ctk.CTkFrame):
    """Cadre arrondi, un peu plus clair que le fond de la fenêtre."""

    def __init__(self, parent: ctk.CTkBaseClass, **options) -> None:
        """options remplace au besoin le style par défaut (ex. border_color)."""
        style = {"fg_color": CARTE, "corner_radius": 14, "border_width": 1, "border_color": BORDURE}
        super().__init__(parent, **{**style, **options})


class Tuile(Carte):
    """Indicateur clé : petit titre, grande valeur, détail."""

    def __init__(self, parent: ctk.CTkBaseClass, titre: str, valeur: str, detail: str,
                 couleur: str = TEXTE) -> None:
        super().__init__(parent)
        etiquette_titre(self, titre).pack(anchor="w", padx=18, pady=(16, 0))
        ctk.CTkLabel(self, text=valeur, font=police(30, True), text_color=couleur).pack(anchor="w", padx=18)
        ctk.CTkLabel(self, text=detail, font=police(12), text_color=TEXTE_SECONDAIRE).pack(
            anchor="w", padx=18, pady=(0, 16))


def etiquette_titre(parent: ctk.CTkBaseClass, texte: str) -> ctk.CTkLabel:
    """Petit titre de section en majuscules discrètes."""
    return ctk.CTkLabel(parent, text=texte.upper(), font=police(11, True), text_color=TEXTE_DISCRET)


def texte_long(parent: ctk.CTkBaseClass, texte: str = "", taille: int = 13,
               couleur: str = TEXTE_SECONDAIRE, conteneur: ctk.CTkBaseClass | None = None,
               marge: int = 40) -> ctk.CTkLabel:
    """Étiquette multiligne dont le retour à la ligne suit la largeur de conteneur."""
    etiquette = ctk.CTkLabel(parent, text=texte, font=police(taille), text_color=couleur,
                             justify="left", anchor="w", wraplength=400)

    def ajuster(evenement) -> None:
        echelle = ctk.ScalingTracker.get_widget_scaling(etiquette)
        etiquette.configure(wraplength=max(120, evenement.width / echelle - marge))

    (conteneur or parent).bind("<Configure>", ajuster, add="+")
    return etiquette


def bouton_principal(parent: ctk.CTkBaseClass, texte: str, commande: Callable[[], None],
                     **options) -> ctk.CTkButton:
    """Bouton d'action principale (bleu) ; options remplace au besoin le style par défaut."""
    style = {"fg_color": ACCENT, "hover_color": ACCENT_SURVOL, "text_color": "white",
             "font": police(14, True), "height": 40, "corner_radius": 10}
    return ctk.CTkButton(parent, text=texte, command=commande, **{**style, **options})


def bouton_secondaire(parent: ctk.CTkBaseClass, texte: str, commande: Callable[[], None],
                      couleur: str = TEXTE, **options) -> ctk.CTkButton:
    """Bouton discret, contour seulement ; options remplace au besoin le style par défaut."""
    style = {"fg_color": "transparent", "hover_color": SURVOL, "border_width": 1, "border_color": BORDURE,
             "text_color": couleur, "font": police(13), "height": 36, "corner_radius": 10}
    return ctk.CTkButton(parent, text=texte, command=commande, **{**style, **options})


def selecteur(parent: ctk.CTkBaseClass, choix: list[str], commande: Callable[[str], None]) -> ctk.CTkSegmentedButton:
    """Boutons segmentés (un seul choix actif), utilisés pour choisir un graphique."""
    bouton = ctk.CTkSegmentedButton(parent, values=choix, command=commande, font=police(12), height=32,
                                    corner_radius=8, fg_color=SAISIE, selected_color=ACCENT,
                                    selected_hover_color=ACCENT_SURVOL, unselected_color=SAISIE,
                                    unselected_hover_color=SURVOL, text_color=TEXTE)
    bouton.set(choix[0])
    return bouton


class Champ:
    """Ligne du formulaire : libellé et aide à gauche, saisie à droite, erreur en dessous.

    Une liste déroulante pour les catégories, une zone de texte pour les nombres. Occupe
    deux lignes de la grille du parent (la seconde pour le message d'erreur).
    """

    def __init__(self, parent: ctk.CTkBaseClass, variable: Variable, ligne: int,
                 au_changement: Callable[[], None]) -> None:
        """Crée les widgets aux lignes ligne et ligne + 1 ; au_changement est appelé à chaque modification."""
        self._variable = variable
        self._au_changement = au_changement
        textes = ctk.CTkFrame(parent, fg_color="transparent")
        textes.grid(row=ligne, column=0, sticky="w", padx=(18, 12), pady=(6, 0))
        ctk.CTkLabel(textes, text=variable.libelle, font=police(13), text_color=TEXTE, height=20,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(textes, text=self._aide(), font=police(11), text_color=TEXTE_DISCRET, height=16,
                     anchor="w").pack(anchor="w")

        self._liste = isinstance(variable, VariableCategorielle)
        if self._liste:
            self._widget = ctk.CTkOptionMenu(
                parent, values=[NON_RENSEIGNE, *variable.choix()], command=lambda _: self._modifie(),
                width=215, height=34, corner_radius=8, fg_color=SAISIE, button_color=SAISIE,
                button_hover_color=SURVOL, text_color=TEXTE_DISCRET, dropdown_fg_color=CARTE,
                dropdown_hover_color=SURVOL, dropdown_text_color=TEXTE, font=police(13),
                dropdown_font=police(13), anchor="w", dynamic_resizing=False)
            self._widget.set(NON_RENSEIGNE)
        else:
            self._widget = ctk.CTkEntry(parent, width=215, height=34, corner_radius=8, border_width=1,
                                        fg_color=SAISIE, border_color=BORDURE, text_color=TEXTE,
                                        placeholder_text=f"ex. {variable.formater(variable.valeur_typique)}",
                                        placeholder_text_color=TEXTE_DISCRET, font=police(13))
            self._widget.bind("<KeyRelease>", lambda _: self._modifie())
            self._widget.bind("<FocusOut>", lambda _: self.verifier())
        self._widget.grid(row=ligne, column=1, sticky="e", padx=(0, 18), pady=(6, 0))
        self._message = ctk.CTkLabel(parent, text="", font=police(11), text_color=ERREUR, height=16)
        self._message.grid(row=ligne + 1, column=0, columnspan=2, sticky="e", padx=18)
        self._message.grid_remove()

    @property
    def variable(self) -> Variable:
        """Variable saisie dans ce champ."""
        return self._variable

    @property
    def widget(self) -> ctk.CTkBaseClass:
        """Widget de saisie (pour placer le curseur ou faire défiler jusqu'à lui)."""
        return self._widget

    def _aide(self) -> str:
        """Intervalle permis (nombres) et valeur typique."""
        variable = self._variable
        typique = variable.formater(variable.valeur_typique)
        if isinstance(variable, VariableNumerique):
            return (f"Entre {variable.formater(variable.minimum)} et {variable.formater(variable.maximum)}"
                    f" · typique {typique}")
        return f"Typique : {typique}"

    def _modifie(self) -> None:
        """Efface l'erreur affichée, grise la liste si elle est vide, prévient la page."""
        self.marquer_erreur(None)
        if self._liste:
            self._widget.configure(text_color=TEXTE_DISCRET if self.texte() == NON_RENSEIGNE else TEXTE)
        self._au_changement()

    def texte(self) -> str:
        """Texte actuellement saisi ou choisi (vide si rien n'est saisi)."""
        return self._widget.get()

    def definir(self, texte: str) -> None:
        """Remplace le contenu du champ."""
        if self._liste:
            self._widget.set(texte)
        else:
            self._widget.delete(0, "end")
            self._widget.insert(0, texte)
        self._modifie()

    def effacer(self) -> None:
        """Remet le champ à « non renseigné »."""
        self.definir(NON_RENSEIGNE if self._liste else "")
        if not self._liste:
            self._widget.delete(0, "end")  # réaffiche l'exemple grisé

    def verifier(self) -> bool:
        """Valide la saisie avec la variable ; affiche l'erreur sous le champ. Renvoie True si valide."""
        try:
            self._variable.lire(self.texte())
        except DonneeInvalideError as erreur:
            self.marquer_erreur(str(erreur))
            return False
        self.marquer_erreur(None)
        return True

    def marquer_erreur(self, message: str | None) -> None:
        """Affiche message en rouge sous le champ, ou retire l'erreur si message est None."""
        if message is None:
            self._message.grid_remove()
            if self._liste:
                self._widget.configure(fg_color=SAISIE, button_color=SAISIE)
            else:
                self._widget.configure(border_color=BORDURE)
            return
        # « Âge (ans) : la valeur doit... » -> « La valeur doit... » (le libellé est déjà à gauche).
        detail = message.split(" : ", 1)[-1] if message.startswith(self._variable.libelle) else message
        self._message.configure(text=detail[:1].upper() + detail[1:])
        self._message.grid()
        if self._liste:
            self._widget.configure(fg_color=ERREUR_FOND, button_color=ERREUR_FOND)
        else:
            self._widget.configure(border_color=ERREUR)

    def donner_focus(self) -> None:
        """Place le curseur dans le champ."""
        self._widget.focus_set()


class ZoneGraphique(ctk.CTkFrame):
    """Cadre qui affiche une figure Matplotlib et peut la remplacer par une autre."""

    def __init__(self, parent: ctk.CTkBaseClass) -> None:
        super().__init__(parent, fg_color=CARTE, corner_radius=0)
        self._canevas: FigureCanvasTkAgg | None = None

    def afficher(self, figure: Figure) -> None:
        """Remplace le graphique affiché par figure."""
        if self._canevas is not None:
            self._canevas.get_tk_widget().destroy()
        self._canevas = FigureCanvasTkAgg(figure, master=self)
        widget = self._canevas.get_tk_widget()
        widget.configure(background=CARTE, highlightthickness=0)
        self._canevas.draw()
        widget.pack(fill="both", expand=True)
