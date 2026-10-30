"""Interface graphique Tkinter.

Elle ne fait aucun calcul : elle lit les saisies, appelle la classe Application et
affiche ce qu'elle renvoie (texte, graphiques Matplotlib, historique).
"""

import ctypes
import json
import os
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.application import Application, Prediction, pourcentage
from src.entrainement import entrainer_et_sauvegarder
from src.exceptions import DonneeInvalideError, ErreurProjet, FichierIntrouvableError, SaisieInvalideError
from src.graphiques import (figure_comparaison_cibles, figure_effets, figure_importances,
                            figure_matrice_confusion, figure_position_risque, figure_precision_champs)
from src.variables import NON_RENSEIGNE, Variable, VariableCategorielle

POLICE = "Segoe UI"
FOND = "#f0efec"
SURFACE = "#ffffff"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ACCENT = "#2a78d6"
ACCENT_FONCE = "#1c5cab"
ERREUR = "#b02a2a"
ERREUR_FOND = "#fdecec"
AVERTISSEMENT = "#6b4700"
AVERTISSEMENT_FOND = "#fff4d6"
COULEURS_NIVEAU = {"faible": "#006300", "modéré": "#8a5a00", "élevé": "#c0461a", "très élevé": ERREUR}

TEXTE_AVERTISSEMENT = (
    "⚠  Exercice pédagogique (INF1007). Cet outil ne doit jamais servir à prendre une décision "
    "concernant un employé réel : les données sont fictives et le modèle se trompe environ une fois sur quatre.")

TEXTE_A_PROPOS = """Que fait cette application ?
Elle estime le risque qu'un employé quitte son entreprise à partir de 26 caractéristiques de sa vie professionnelle et personnelle : âge, état civil, poste, salaire, heures supplémentaires, ancienneté, satisfaction, équilibre travail-vie personnelle, etc.

Avertissement
Ceci est un exercice pédagogique, pas une aide à la décision. Les données sont fictives (créées par des scientifiques des données d'IBM) et le modèle se trompe souvent. Plusieurs variables, comme l'âge et l'état civil, sont des motifs de discrimination interdits par la Charte des droits et libertés de la personne du Québec (article 10) : utiliser un tel outil pour embaucher, promouvoir ou congédier serait injuste et illégal. L'accident du Therac-25 rappelle qu'une confiance aveugle dans un logiciel peut avoir des conséquences graves.

Comment lire le « risque de départ » ?
Pendant l'entraînement, le modèle accorde autant de poids aux départs (16 % des employés) qu'aux maintiens, pour ne pas se contenter de répondre toujours « reste ». Le pourcentage affiché est donc un indice de risque plutôt qu'une fréquence réelle : au-delà de 50 %, le modèle conclut à un départ probable.

Comment est calculée la « précision attendue » ?
Quand un champ est vide, on ne connaît pas sa valeur : le modèle essaie alors les valeurs de 40 vrais employés de référence et fait la moyenne des risques obtenus. Le même calcul est appliqué aux employés du jeu de test (jamais vus à l'entraînement) en ne gardant que les champs que vous avez remplis. La précision attendue est l'exactitude équilibrée obtenue : la moyenne entre le pourcentage de départs bien détectés et le pourcentage de maintiens bien détectés. 50 % correspond au hasard. Le jeu de test ne contient que 59 départs : ce pourcentage varie d'environ ± 3 points selon l'échantillon.

Pourquoi ne pas prédire directement la satisfaction ?
Nous l'avons essayé : la satisfaction au travail n'a presque aucun lien avec les autres variables de ce jeu de données (corrélations inférieures à 0,11). Le meilleur modèle faisait moins bien que la référence triviale (voir l'onglet « Le modèle »). Le départ, lui, est l'expression mesurable du contentement et il est prévisible.

Source des données
« IBM HR Analytics Employee Attrition & Performance », Kaggle (pavansubhasht), licence Open Database License (base) et Database Contents License (contenu). 1470 employés fictifs, 35 colonnes.

Auteurs
Maxime Hamrioui et Alexandre Bertrand-Beaudet — INF1007, automne 2026."""


class Champ:
    """Ligne du formulaire : libellé, widget adapté au type de la variable et aide."""

    def __init__(self, parent: ttk.Frame, variable: Variable, ligne: int,
                 au_changement: Callable[[], None]) -> None:
        """Crée les widgets à la ligne donnée ; au_changement est appelé à chaque modification."""
        self._variable = variable
        self._texte = tk.StringVar()
        if isinstance(variable, VariableCategorielle):
            self._vide = NON_RENSEIGNE
            self._widget = ttk.Combobox(parent, textvariable=self._texte, state="readonly", width=24,
                                        values=[NON_RENSEIGNE, *variable.choix()])
            self._styles = ("TCombobox", "Erreur.TCombobox")
        else:
            self._vide = ""
            self._widget = ttk.Entry(parent, textvariable=self._texte, width=26)
            self._styles = ("TEntry", "Erreur.TEntry")
        self._texte.set(self._vide)
        ttk.Label(parent, text=variable.libelle, style="Carte.TLabel").grid(
            row=ligne, column=0, sticky="w", padx=(12, 8), pady=3)
        self._widget.grid(row=ligne, column=1, sticky="ew", pady=3)
        ttk.Label(parent, text=self._aide(), style="Aide.TLabel").grid(
            row=ligne, column=2, sticky="w", padx=(8, 12))
        self._texte.trace_add("write", lambda *_: self._apres_modification(au_changement))

    def _aide(self) -> str:
        """Texte d'aide : intervalle permis (nombres) et valeur typique."""
        typique = self._variable.formater(self._variable.valeur_typique)
        if isinstance(self._variable, VariableCategorielle):
            return f"typique : {typique}"
        return f"{self._variable.aide()} · typique {typique}"

    def _apres_modification(self, au_changement: Callable[[], None]) -> None:
        """Retire la marque d'erreur dès que l'utilisateur corrige le champ."""
        self.marquer_erreur(False)
        au_changement()

    def texte(self) -> str:
        """Texte actuellement saisi ou choisi."""
        return self._texte.get()

    def definir(self, texte: str) -> None:
        """Remplace le contenu du champ."""
        self._texte.set(texte)

    def effacer(self) -> None:
        """Vide le champ (« non renseigné »)."""
        self._texte.set(self._vide)

    def marquer_erreur(self, en_erreur: bool) -> None:
        """Colore le champ en rouge (en_erreur) ou lui rend son style normal."""
        self._widget.configure(style=self._styles[1] if en_erreur else self._styles[0])

    def donner_focus(self) -> None:
        """Place le curseur dans le champ."""
        self._widget.focus_set()


class ZoneGraphique(ttk.Frame):
    """Cadre qui affiche une figure Matplotlib et peut la remplacer par une autre."""

    def __init__(self, parent: tk.Misc) -> None:
        """Crée un cadre vide."""
        super().__init__(parent, style="Carte.TFrame")
        self._canevas: FigureCanvasTkAgg | None = None

    def afficher(self, figure: Figure) -> None:
        """Remplace le graphique affiché par figure."""
        if self._canevas is not None:
            self._canevas.get_tk_widget().destroy()
        self._canevas = FigureCanvasTkAgg(figure, master=self)
        self._canevas.draw()
        self._canevas.get_tk_widget().pack(fill="both", expand=True)


class FenetrePrincipale:
    """Fenêtre principale : contient l'Application et les champs du formulaire (composition)."""

    def __init__(self, racine: tk.Tk, application: Application) -> None:
        """Construit tous les onglets à partir des métadonnées de l'application."""
        self._racine = racine
        self._application = application
        self._champs: dict[str, Champ] = {}
        self._prediction: Prediction | None = None
        self._saisies_predites: dict[str, str] = {}
        self._minuterie: str | None = None
        self._lignes_historique: dict[str, dict[str, str]] = {}
        # Facteur d'agrandissement de l'écran (1 à 96 ppp, 1,5 à 144 ppp...).
        self._echelle = float(racine.tk.call("tk", "scaling")) / (96 / 72)
        self._configurer_fenetre()
        self._configurer_styles()
        self._construire()
        self._maj_indicateur()
        self._rafraichir_historique()

    # ----- Mise en place -------------------------------------------------------------

    def _px(self, valeur: float) -> int:
        """Convertit une taille pensée pour un écran standard en pixels réels."""
        return int(valeur * self._echelle)

    def _configurer_fenetre(self) -> None:
        """Titre, taille, raccourcis et défilement à la molette."""
        self._racine.title("Risque de départ d'un employé — INF1007 (exercice pédagogique)")
        largeur = min(self._px(1360), self._racine.winfo_screenwidth() - 40)
        hauteur = min(self._px(880), self._racine.winfo_screenheight() - 80)
        self._racine.geometry(f"{largeur}x{hauteur}")
        self._racine.minsize(min(self._px(1000), largeur), min(self._px(620), hauteur))
        self._racine.configure(background=FOND)
        self._racine.protocol("WM_DELETE_WINDOW", self._fermer)
        self._racine.bind("<Return>", self._predire)
        # Une liste déroulante ne doit pas changer de valeur quand on fait défiler le formulaire.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._racine.unbind_class("TCombobox", sequence)
            self._racine.bind_all(sequence, self._defiler)

    def _configurer_styles(self) -> None:
        """Thème « clam » (le seul qui permet de colorer les champs) et styles du projet."""
        style = ttk.Style(self._racine)
        style.theme_use("clam")
        style.configure(".", background=FOND, foreground=ENCRE, font=(POLICE, 10))
        style.configure("TFrame", background=FOND)
        style.configure("Carte.TFrame", background=SURFACE)
        style.configure("TLabel", background=FOND, foreground=ENCRE)
        style.configure("Carte.TLabel", background=SURFACE)
        style.configure("Titre.TLabel", font=(POLICE, 13, "bold"))
        style.configure("CarteTitre.TLabel", background=SURFACE, font=(POLICE, 12, "bold"))
        style.configure("Sous.TLabel", foreground=ENCRE_SECONDAIRE, font=(POLICE, 9))
        style.configure("Aide.TLabel", background=SURFACE, foreground=ENCRE_SECONDAIRE, font=(POLICE, 9))
        style.configure("Section.TLabel", background=SURFACE, foreground=ACCENT_FONCE,
                        font=(POLICE, 11, "bold"))
        style.configure("TButton", padding=(10, 4))
        style.configure("Accent.TButton", background=ACCENT, foreground="white", font=(POLICE, 10, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_FONCE), ("pressed", ACCENT_FONCE)])
        for nom, fond in (("TCombobox", SURFACE), ("Erreur.TCombobox", ERREUR_FOND)):
            style.map(nom, fieldbackground=[("readonly", fond)], selectbackground=[("readonly", fond)],
                      selectforeground=[("readonly", ENCRE)])
        style.configure("Erreur.TCombobox", bordercolor=ERREUR, lightcolor=ERREUR, darkcolor=ERREUR)
        style.configure("Erreur.TEntry", fieldbackground=ERREUR_FOND, bordercolor=ERREUR,
                        lightcolor=ERREUR, darkcolor=ERREUR)
        style.configure("Carte.TRadiobutton", background=SURFACE)
        style.map("Carte.TRadiobutton", background=[("active", SURFACE)])
        style.configure("Champs.Horizontal.TProgressbar", background=ACCENT, troughcolor="#e1e0d9",
                        bordercolor="#e1e0d9", lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("TNotebook", background=FOND, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 6), font=(POLICE, 10))
        style.map("TNotebook.Tab", background=[("selected", SURFACE)])
        style.configure("Treeview", rowheight=self._px(24), font=(POLICE, 9),
                        background=SURFACE, fieldbackground=SURFACE)
        style.configure("Treeview.Heading", font=(POLICE, 9, "bold"))

    def _construire(self) -> None:
        """Bandeau d'avertissement puis les quatre onglets."""
        bandeau = tk.Label(self._racine, text=TEXTE_AVERTISSEMENT, bg=AVERTISSEMENT_FOND, fg=AVERTISSEMENT,
                           font=(POLICE, 10, "bold"), anchor="w", justify="left", padx=12, pady=6)
        bandeau.pack(fill="x")
        bandeau.bind("<Configure>", lambda evenement: bandeau.configure(wraplength=evenement.width - 24))
        onglets = ttk.Notebook(self._racine)
        onglets.pack(fill="both", expand=True, padx=8, pady=8)
        onglets.add(self._onglet_prediction(onglets), text="Prédiction")
        onglets.add(self._onglet_modele(onglets), text="Le modèle")
        onglets.add(self._onglet_historique(onglets), text="Historique")
        onglets.add(self._onglet_a_propos(onglets), text="À propos")

    # ----- Onglet « Prédiction » -----------------------------------------------------

    def _onglet_prediction(self, parent: ttk.Notebook) -> ttk.Frame:
        """Formulaire à gauche, résultat et graphique à droite."""
        onglet = ttk.Frame(parent, padding=8)
        onglet.columnconfigure(0, weight=6, uniform="colonnes")
        onglet.columnconfigure(1, weight=5, uniform="colonnes")
        onglet.rowconfigure(0, weight=1)
        gauche = ttk.Frame(onglet)
        gauche.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        droite = ttk.Frame(onglet, style="Carte.TFrame", padding=12)
        droite.grid(row=0, column=1, sticky="nsew")
        self._construire_formulaire(gauche)
        self._construire_resultat(droite)
        return onglet

    def _construire_formulaire(self, parent: ttk.Frame) -> None:
        """Formulaire défilant construit à partir des métadonnées, puis les boutons."""
        ttk.Label(parent, text="Décrivez l'employé", style="Titre.TLabel").pack(anchor="w")
        ttk.Label(parent, style="Sous.TLabel", justify="left", wraplength=self._px(640),
                  text="Remplissez autant de champs que vous le souhaitez : un champ vide est permis, "
                       "mais la précision attendue (à droite) baisse en conséquence.").pack(anchor="w", pady=(0, 6))

        boutons = ttk.Frame(parent)
        boutons.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(boutons, text="Prédire", style="Accent.TButton", command=self._predire).pack(side="left")
        ttk.Button(boutons, text="Tout effacer", command=self._effacer).pack(side="left", padx=6)
        ttk.Label(boutons, text="Exemples réels :", style="Sous.TLabel").pack(side="left", padx=(10, 4))
        for texte, genre in (("Profil à risque", "risque"), ("Profil stable", "stable"), ("Au hasard", "hasard")):
            ttk.Button(boutons, text=texte, command=lambda g=genre: self._charger_exemple(g)).pack(side="left", padx=2)

        self._message = tk.Label(parent, text="", bg=FOND, fg=ENCRE_SECONDAIRE, justify="left", anchor="w",
                                 font=(POLICE, 9), wraplength=self._px(640))
        self._message.pack(side="bottom", fill="x", pady=(6, 0))

        cadre = ttk.Frame(parent, style="Carte.TFrame")
        cadre.pack(fill="both", expand=True)
        self._canevas_formulaire = tk.Canvas(cadre, background=SURFACE, highlightthickness=0)
        barre = ttk.Scrollbar(cadre, orient="vertical", command=self._canevas_formulaire.yview)
        interieur = ttk.Frame(self._canevas_formulaire, style="Carte.TFrame", padding=(0, 0, 0, 12))
        fenetre = self._canevas_formulaire.create_window((0, 0), window=interieur, anchor="nw")
        interieur.bind("<Configure>", lambda _: self._canevas_formulaire.configure(
            scrollregion=self._canevas_formulaire.bbox("all")))
        self._canevas_formulaire.bind("<Configure>", lambda evenement: self._canevas_formulaire.itemconfigure(
            fenetre, width=evenement.width))
        self._canevas_formulaire.configure(yscrollcommand=barre.set)
        barre.pack(side="right", fill="y")
        self._canevas_formulaire.pack(side="left", fill="both", expand=True)

        interieur.columnconfigure(1, weight=1)
        ligne = 0
        for groupe, variables in self._application.variables_par_groupe().items():
            ttk.Label(interieur, text=groupe, style="Section.TLabel").grid(
                row=ligne, column=0, columnspan=3, sticky="w", padx=12, pady=(14 if ligne else 8, 2))
            ligne += 1
            for variable in variables:
                self._champs[variable.nom] = Champ(interieur, variable, ligne, self._planifier_maj)
                ligne += 1

    def _construire_resultat(self, parent: ttk.Frame) -> None:
        """Risque, indicateur de précision selon les champs remplis et graphique."""
        largeur_texte = self._px(560)
        ttk.Label(parent, text="Risque de départ estimé", style="CarteTitre.TLabel").pack(anchor="w")
        ligne = ttk.Frame(parent, style="Carte.TFrame")
        ligne.pack(fill="x")
        self._etiquette_risque = tk.Label(ligne, text="—", font=(POLICE, 30, "bold"), bg=SURFACE,
                                          fg=ENCRE_SECONDAIRE)
        self._etiquette_risque.pack(side="left")
        self._etiquette_niveau = tk.Label(ligne, text="", font=(POLICE, 13, "bold"), bg=SURFACE,
                                          fg=ENCRE_SECONDAIRE, justify="left")
        self._etiquette_niveau.pack(side="left", padx=12)
        self._etiquette_phrase = tk.Label(parent, text="Remplissez le formulaire, puis cliquez sur « Prédire ».",
                                          bg=SURFACE, fg=ENCRE, justify="left", anchor="w",
                                          wraplength=largeur_texte, font=(POLICE, 10))
        self._etiquette_phrase.pack(fill="x", pady=(2, 0))
        self._etiquette_perimee = tk.Label(parent, text="", bg=SURFACE, fg=AVERTISSEMENT, justify="left",
                                           anchor="w", wraplength=largeur_texte, font=(POLICE, 9, "italic"))
        self._etiquette_perimee.pack(fill="x")

        ttk.Separator(parent).pack(fill="x", pady=8)
        indicateur = ttk.Frame(parent, style="Carte.TFrame")
        indicateur.pack(fill="x")
        indicateur.columnconfigure(1, weight=1)
        self._etiquette_champs = ttk.Label(indicateur, style="Carte.TLabel")
        self._etiquette_champs.grid(row=0, column=0, sticky="w")
        self._barre_champs = ttk.Progressbar(indicateur, maximum=len(self._champs),
                                             style="Champs.Horizontal.TProgressbar")
        self._barre_champs.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self._etiquette_precision = tk.Label(indicateur, font=(POLICE, 12, "bold"), bg=SURFACE, fg=ACCENT_FONCE)
        self._etiquette_precision.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._etiquette_conseil = ttk.Label(indicateur, style="Aide.TLabel", wraplength=largeur_texte,
                                            justify="left")
        self._etiquette_conseil.grid(row=2, column=0, columnspan=2, sticky="w")
        meta = self._application.metadonnees
        maximum = meta["courbe_precision"]["ordre_importance"][-1]
        ttk.Label(parent, style="Aide.TLabel", wraplength=largeur_texte, justify="left",
                  text=f"Précision attendue : pourcentage de bonnes réponses du modèle (moyenne entre "
                       f"employés partis et restés), mesuré sur {meta['lignes']['test']} employés jamais vus "
                       f"à l'entraînement, avec les mêmes champs remplis que vous. 50 % = hasard ; "
                       f"tous les champs remplis : {pourcentage(maximum)}.").pack(fill="x", pady=(4, 0))

        ttk.Separator(parent).pack(fill="x", pady=8)
        choix = ttk.Frame(parent, style="Carte.TFrame")
        choix.pack(fill="x")
        self._choix_graphique = tk.StringVar(value="precision")
        for texte, valeur in (("Précision selon les champs", "precision"),
                              ("Ce qui influence le résultat", "effets"),
                              ("Position parmi les employés", "position")):
            ttk.Radiobutton(choix, text=texte, value=valeur, variable=self._choix_graphique,
                            command=self._afficher_graphique, style="Carte.TRadiobutton").pack(side="left", padx=(0, 10))
        self._zone_graphique = ZoneGraphique(parent)
        self._zone_graphique.pack(fill="both", expand=True, pady=(6, 0))

    # ----- Actions de l'onglet « Prédiction » ----------------------------------------

    def _saisies(self) -> dict[str, str]:
        """Texte de chaque champ du formulaire."""
        return {nom: champ.texte() for nom, champ in self._champs.items()}

    def _planifier_maj(self) -> None:
        """Met à jour l'indicateur 250 ms après la dernière frappe (évite de recalculer à chaque touche)."""
        if self._minuterie is not None:
            self._racine.after_cancel(self._minuterie)
        self._minuterie = self._racine.after(250, self._maj_indicateur)

    def _maj_indicateur(self) -> None:
        """Nombre de champs remplis, précision attendue, conseils et graphique de précision."""
        self._minuterie = None
        saisies = self._saisies()
        champs = self._application.champs_renseignes(saisies)
        precision = self._application.precision_attendue(champs)
        self._etiquette_champs.configure(text=f"Champs renseignés : {len(champs)} sur {len(self._champs)}")
        self._barre_champs.configure(value=len(champs))
        self._etiquette_precision.configure(text=f"Précision attendue : {pourcentage(precision)}")
        conseilles = self._application.champs_conseilles(champs)
        self._etiquette_conseil.configure(
            text="Champs les plus utiles à remplir ensuite : " + ", ".join(v.libelle for v in conseilles)
            if conseilles else "Tous les champs sont remplis.")
        perime = self._prediction is not None and saisies != self._saisies_predites
        self._etiquette_perimee.configure(
            text="Le formulaire a changé depuis cette prédiction : cliquez sur « Prédire » pour la mettre à jour."
            if perime else "")
        if self._choix_graphique.get() == "precision":
            self._afficher_graphique()

    def _afficher_graphique(self) -> None:
        """Dessine le graphique choisi par les boutons radio."""
        choix = self._choix_graphique.get()
        if choix == "precision":
            champs = self._application.champs_renseignes(self._saisies())
            figure = figure_precision_champs(self._application.metadonnees["courbe_precision"],
                                             len(champs), self._application.precision_attendue(champs))
        elif choix == "effets":
            if self._prediction is None:
                figure = figure_effets({}, message_vide="Cliquez sur « Prédire » pour voir ce qui influence le résultat")
            else:
                valeurs = {nom: self._application.variable(nom).formater(valeur)
                           for nom, valeur in self._prediction.valeurs_connues.items()}
                figure = figure_effets(self._prediction.effets, valeurs)
        else:
            scores, departs = self._application.scores_test()
            figure = figure_position_risque(scores, departs,
                                            self._prediction.probabilite if self._prediction else None)
        self._zone_graphique.afficher(figure)

    def _predire(self, _evenement: tk.Event | None = None) -> None:
        """Valide le formulaire, affiche la prédiction et l'ajoute à l'historique."""
        for champ in self._champs.values():
            champ.marquer_erreur(False)
        saisies = self._saisies()
        try:
            prediction = self._application.predire(saisies, enregistrer=False)
        except SaisieInvalideError as erreur:
            fautifs = [nom for nom in self._champs if nom in erreur.erreurs]
            for nom in fautifs:
                self._champs[nom].marquer_erreur(True)
            if fautifs:
                self._champs[fautifs[0]].donner_focus()
            self._afficher_message(f"{len(erreur.erreurs)} saisie(s) à corriger (en rouge) :\n"
                                   + "\n".join(f"• {message}" for message in erreur.erreurs.values()),
                                   erreur=True)
            return
        except DonneeInvalideError as erreur:
            self._afficher_message(str(erreur), erreur=True)
            return
        self._prediction, self._saisies_predites = prediction, saisies
        self._afficher_prediction(prediction)
        try:
            self._application.enregistrer(prediction)
        except OSError as erreur:
            self._afficher_message(f"Prédiction affichée, mais l'historique n'a pas pu être enregistré "
                                   f"(le fichier est peut-être ouvert dans un autre programme) : {erreur}",
                                   erreur=True)
        else:
            self._afficher_message(f"Prédiction ajoutée à l'historique à {prediction.date:%H:%M:%S}.")
            self._rafraichir_historique()

    def _afficher_prediction(self, prediction: Prediction) -> None:
        """Affiche le risque, le niveau et une phrase compréhensible par un non-informaticien."""
        couleur = COULEURS_NIVEAU[prediction.niveau]
        self._etiquette_risque.configure(text=pourcentage(prediction.probabilite), fg=couleur)
        self._etiquette_niveau.configure(text=f"Risque {prediction.niveau}\n{prediction.conclusion().capitalize()}",
                                         fg=couleur)
        vides = prediction.nb_variables - prediction.nb_champs
        phrase = (f"Selon le modèle, un employé avec ce profil présente un risque {prediction.niveau} de "
                  f"quitter l'entreprise. Avec {prediction.nb_champs} champ(s) rempli(s) sur "
                  f"{prediction.nb_variables}, le modèle a raison dans environ "
                  f"{pourcentage(prediction.precision_attendue)} des cas (50 % = hasard).")
        if vides:
            phrase += f" Les {vides} champ(s) vide(s) ont été estimés à partir d'employés de référence."
        self._etiquette_phrase.configure(text=phrase)
        self._rafraichir_affichage()

    def _rafraichir_affichage(self) -> None:
        """Met à jour l'indicateur et le graphique choisi, sans dessiner deux fois le même graphique."""
        self._maj_indicateur()
        if self._choix_graphique.get() != "precision":
            self._afficher_graphique()

    def _afficher_message(self, texte: str, erreur: bool = False) -> None:
        """Message sous le formulaire : rouge pour une erreur, gris sinon."""
        self._message.configure(text=texte, fg=ERREUR if erreur else ENCRE_SECONDAIRE)

    def _effacer(self) -> None:
        """Vide le formulaire et le résultat affiché."""
        for champ in self._champs.values():
            champ.effacer()
        self._prediction, self._saisies_predites = None, {}
        self._etiquette_risque.configure(text="—", fg=ENCRE_SECONDAIRE)
        self._etiquette_niveau.configure(text="")
        self._etiquette_phrase.configure(text="Remplissez le formulaire, puis cliquez sur « Prédire ».")
        self._afficher_message("")
        self._rafraichir_affichage()

    def _charger_exemple(self, genre: str) -> None:
        """Remplit le formulaire avec un vrai employé du jeu de test."""
        saisies, parti = self._application.exemple(genre)
        for nom, texte in saisies.items():
            self._champs[nom].definir(texte)
        realite = "a réellement quitté l'entreprise" if parti else "est réellement resté dans l'entreprise"
        self._afficher_message(f"Exemple chargé : un véritable employé du jeu de test, qui {realite}. "
                               f"Cliquez sur « Prédire » pour comparer avec le modèle.")

    def _defiler(self, evenement: tk.Event) -> None:
        """Fait défiler le formulaire à la molette, seulement quand la souris est dessus."""
        try:
            widget = self._racine.winfo_containing(evenement.x_root, evenement.y_root)
        except KeyError:  # souris au-dessus de la liste ouverte d'une liste déroulante
            return
        if widget is None or not str(widget).startswith(str(self._canevas_formulaire)):
            return
        if evenement.num == 4 or evenement.delta > 0:
            self._canevas_formulaire.yview_scroll(-1, "units")
        elif evenement.num == 5 or evenement.delta < 0:
            self._canevas_formulaire.yview_scroll(1, "units")

    # ----- Onglet « Le modèle » ------------------------------------------------------

    def _onglet_modele(self, parent: ttk.Notebook) -> ttk.Frame:
        """Résultats de l'évaluation : tableau des métriques, explications et graphiques."""
        meta = self._application.metadonnees
        onglet = ttk.Frame(parent, padding=8)
        onglet.columnconfigure(0, weight=5, uniform="colonnes")
        onglet.columnconfigure(1, weight=6, uniform="colonnes")
        onglet.rowconfigure(0, weight=1)
        gauche = ttk.Frame(onglet, style="Carte.TFrame", padding=12)
        gauche.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        droite = ttk.Frame(onglet, style="Carte.TFrame", padding=12)
        droite.grid(row=0, column=1, sticky="nsew")
        largeur_texte = self._px(540)

        lignes = meta["lignes"]
        ttk.Label(gauche, text=f"Modèle retenu : {meta['modele']}", style="CarteTitre.TLabel").pack(anchor="w")
        ttk.Label(gauche, style="Carte.TLabel", wraplength=largeur_texte, justify="left",
                  text=f"Choisi par validation croisée sur {lignes['entrainement']} employés d'entraînement, "
                       f"puis évalué une seule fois sur {lignes['test']} employés de test. Entraîné le "
                       f"{meta['date_entrainement'].replace('T', ' à ')} en "
                       f"{meta['duree_entrainement_s']:.2f} s. Cible : l'employé a-t-il quitté l'entreprise ? "
                       f"({pourcentage(meta['taux_depart'])} des employés sont partis.)").pack(anchor="w", pady=(2, 8))

        colonnes = {"modele": ("Modèle", 180), "equilibree": ("Ex. équilibrée", 92),
                    "validation": ("Valid. croisée", 88), "exactitude": ("Exactitude", 72),
                    "rappel": ("Rappel", 56), "auc": ("AUC", 46)}
        tableau = ttk.Treeview(gauche, columns=list(colonnes), show="headings", height=3, selectmode="none")
        for cle, (titre, largeur) in colonnes.items():
            tableau.heading(cle, text=titre)
            tableau.column(cle, width=self._px(largeur), anchor="w" if cle == "modele" else "center")
        for nom, resultats in meta["metriques_test"].items():
            tableau.insert("", "end", values=(
                nom, pourcentage(resultats["exactitude_equilibree"]), pourcentage(meta["validation_croisee"][nom]),
                pourcentage(resultats["exactitude"]), pourcentage(resultats["rappel_depart"]),
                f"{resultats.get('auc', 0.5):.2f}"))
        tableau.pack(fill="x")

        explications = [
            "Exactitude équilibrée (métrique principale) : moyenne entre le pourcentage de départs bien "
            "détectés et le pourcentage de maintiens bien détectés. Répondre toujours la même chose donne 50 %.",
            f"Exactitude : pourcentage de bonnes réponses. Trompeuse ici : répondre toujours « reste » donne "
            f"déjà {pourcentage(meta['metriques_test']['Référence (classe majoritaire)']['exactitude'])}.",
            "Rappel des départs : part des départs réels que le modèle a détectés. "
            "AUC : capacité à classer un employé parti plus haut qu'un employé resté (0,5 = hasard).",
        ]
        satisfaction = meta["comparaison_cibles"]["Satisfaction (≥ 3 sur 4)"]
        explications.append(
            "Pourquoi pas la satisfaction ? Prédire si un employé est satisfait (niveau 3 ou 4 sur 4) donne "
            + " et ".join(f"{pourcentage(valeur)} ({nom.lower()})" for nom, valeur in satisfaction.items())
            + " : aucun modèle ne fait mieux que le hasard, la satisfaction n'est pas prévisible avec ces données.")
        explications.append("Limites : 59 départs seulement dans le test (± 3 points d'incertitude) ; "
                            "données fictives ; le modèle se trompe pour environ un employé sur quatre.")
        for texte in explications:
            ttk.Label(gauche, text=texte, style="Aide.TLabel", wraplength=largeur_texte,
                      justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(gauche, text="Nettoyage des données", style="Section.TLabel").pack(anchor="w", pady=(12, 0))
        for etape in meta["nettoyage"]:
            ttk.Label(gauche, text=f"• {etape}", style="Aide.TLabel", wraplength=largeur_texte,
                      justify="left").pack(anchor="w")

        self._choix_graphique_modele = tk.StringVar(value="matrice")
        choix = ttk.Frame(droite, style="Carte.TFrame")
        choix.pack(fill="x")
        for texte, valeur in (("Matrice de confusion", "matrice"), ("Variables les plus utiles", "importances"),
                              ("Satisfaction ou départ ?", "cibles")):
            ttk.Radiobutton(choix, text=texte, value=valeur, variable=self._choix_graphique_modele,
                            command=self._afficher_graphique_modele,
                            style="Carte.TRadiobutton").pack(side="left", padx=(0, 10))
        self._zone_graphique_modele = ZoneGraphique(droite)
        self._zone_graphique_modele.pack(fill="both", expand=True, pady=(6, 0))
        self._afficher_graphique_modele()
        return onglet

    def _afficher_graphique_modele(self) -> None:
        """Dessine le graphique choisi dans l'onglet « Le modèle »."""
        meta = self._application.metadonnees
        choix = self._choix_graphique_modele.get()
        if choix == "matrice":
            figure = figure_matrice_confusion(meta["matrice_confusion"])
        elif choix == "importances":
            figure = figure_importances(meta["importances"])
        else:
            figure = figure_comparaison_cibles(meta["comparaison_cibles"])
        self._zone_graphique_modele.afficher(figure)

    # ----- Onglet « Historique » -----------------------------------------------------

    def _onglet_historique(self, parent: ttk.Notebook) -> ttk.Frame:
        """Tableau des prédictions passées et détail de la ligne sélectionnée."""
        onglet = ttk.Frame(parent, padding=12)
        haut = ttk.Frame(onglet)
        haut.pack(fill="x")
        # Les boutons sont placés en premier pour garder leur place si le texte est long.
        ttk.Button(haut, text="Effacer l'historique", command=self._effacer_historique).pack(side="right")
        ttk.Button(haut, text="Rafraîchir", command=self._rafraichir_historique).pack(side="right", padx=6)
        self._etiquette_historique = ttk.Label(haut, text="")
        self._etiquette_historique.pack(side="left")

        cadre = ttk.Frame(onglet)
        cadre.pack(fill="both", expand=True, pady=8)
        colonnes = {"date": ("Date", 170), "risque": ("Risque de départ", 130), "conclusion": ("Conclusion", 150),
                    "champs": ("Champs remplis", 120), "precision": ("Précision attendue", 140)}
        self._tableau_historique = ttk.Treeview(cadre, columns=list(colonnes), show="headings", selectmode="browse")
        for cle, (titre, largeur) in colonnes.items():
            self._tableau_historique.heading(cle, text=titre)
            self._tableau_historique.column(cle, width=self._px(largeur), anchor="w" if cle == "date" else "center")
        barre = ttk.Scrollbar(cadre, orient="vertical", command=self._tableau_historique.yview)
        self._tableau_historique.configure(yscrollcommand=barre.set)
        barre.pack(side="right", fill="y")
        self._tableau_historique.pack(side="left", fill="both", expand=True)
        self._tableau_historique.bind("<<TreeviewSelect>>", self._afficher_detail)

        ttk.Label(onglet, text="Saisies de la prédiction sélectionnée :").pack(anchor="w")
        self._detail_historique = tk.Text(onglet, height=8, wrap="word", font=(POLICE, 9), bg=SURFACE,
                                          relief="flat", padx=8, pady=6, state="disabled")
        self._detail_historique.pack(fill="x")
        return onglet

    def _rafraichir_historique(self) -> None:
        """Relit le fichier d'historique et remplit le tableau."""
        try:
            lignes = self._application.historique.lire()
        except (DonneeInvalideError, OSError) as erreur:
            self._etiquette_historique.configure(text=f"Historique illisible : {erreur}")
            return
        self._tableau_historique.delete(*self._tableau_historique.get_children())
        self._lignes_historique.clear()
        for ligne in lignes:
            identifiant = self._tableau_historique.insert("", "end", values=(
                ligne["date"].replace("T", " "), self._format_pourcentage(ligne["risque_depart"]),
                ligne["conclusion"], ligne["champs_renseignes"],
                self._format_pourcentage(ligne["precision_attendue"])))
            self._lignes_historique[identifiant] = ligne
        chemin = self._application.historique.chemin
        fichier = os.path.join(os.path.basename(os.path.dirname(chemin)), os.path.basename(chemin))
        self._etiquette_historique.configure(text=f"{len(lignes)} prédiction(s) enregistrée(s) dans {fichier}")
        self._ecrire_detail("")

    @staticmethod
    def _format_pourcentage(texte: str) -> str:
        """« 0.684 » -> « 68 % » ; le texte tel quel s'il n'est pas un nombre."""
        try:
            return pourcentage(float(texte))
        except ValueError:
            return texte

    def _afficher_detail(self, _evenement: tk.Event | None = None) -> None:
        """Affiche les saisies de la prédiction sélectionnée, avec les libellés en français."""
        selection = self._tableau_historique.selection()
        if not selection:
            return
        texte_saisies = self._lignes_historique[selection[0]]["saisies"]
        try:
            saisies = json.loads(texte_saisies)
            lignes = [f"{self._application.variable(nom).libelle} : "
                      f"{self._application.variable(nom).formater(valeur)}" for nom, valeur in saisies.items()]
        except (json.JSONDecodeError, AttributeError, DonneeInvalideError):
            lignes = [texte_saisies]
        self._ecrire_detail("\n".join(lignes))

    def _ecrire_detail(self, texte: str) -> None:
        """Remplace le contenu de la zone de détail (en lecture seule)."""
        self._detail_historique.configure(state="normal")
        self._detail_historique.delete("1.0", "end")
        self._detail_historique.insert("1.0", texte)
        self._detail_historique.configure(state="disabled")

    def _effacer_historique(self) -> None:
        """Supprime le fichier d'historique après confirmation."""
        if not messagebox.askyesno("Effacer l'historique",
                                   "Supprimer définitivement toutes les prédictions enregistrées ?",
                                   parent=self._racine):
            return
        try:
            self._application.historique.effacer()
        except OSError as erreur:
            messagebox.showerror("Effacer l'historique", f"Impossible de supprimer le fichier : {erreur}",
                                 parent=self._racine)
        self._rafraichir_historique()

    # ----- Onglet « À propos » -------------------------------------------------------

    def _onglet_a_propos(self, parent: ttk.Notebook) -> ttk.Frame:
        """Texte explicatif en lecture seule ; les titres de section sont en gras."""
        onglet = ttk.Frame(parent, padding=12)
        texte = tk.Text(onglet, wrap="word", font=(POLICE, 10), bg=SURFACE, relief="flat",
                        padx=16, pady=12, spacing3=4)
        texte.tag_configure("titre", font=(POLICE, 11, "bold"), foreground=ACCENT_FONCE, spacing1=8)
        for paragraphe in TEXTE_A_PROPOS.split("\n\n"):
            titre, _, corps = paragraphe.partition("\n")
            texte.insert("end", titre + "\n", "titre")
            texte.insert("end", corps + "\n")
        texte.configure(state="disabled")
        texte.pack(fill="both", expand=True)
        return onglet

    def _fermer(self) -> None:
        """Ferme proprement la fenêtre (et arrête la boucle Tkinter)."""
        self._racine.quit()
        self._racine.destroy()


def _activer_haute_resolution() -> None:
    """Sous Windows, évite une fenêtre floue sur un écran à haute résolution (sans effet ailleurs)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def _charger_application(racine: tk.Tk) -> Application | None:
    """Charge le modèle sauvegardé ; propose de l'entraîner s'il manque. Renvoie None en cas d'abandon."""
    try:
        return Application.depuis_fichiers()
    except FichierIntrouvableError:
        if not messagebox.askyesno("Modèle introuvable", "Le modèle n'a pas encore été entraîné.\n\n"
                                   "Voulez-vous l'entraîner maintenant ? (environ 30 secondes)", parent=racine):
            return None
    except DonneeInvalideError as erreur:
        messagebox.showerror("Fichiers du modèle invalides",
                             f"{erreur}\n\nRelancez l'entraînement : python -m src.main entrainer", parent=racine)
        return None
    try:
        entrainer_et_sauvegarder()
        return Application.depuis_fichiers()
    except ErreurProjet as erreur:
        messagebox.showerror("Entraînement impossible", str(erreur), parent=racine)
        return None


def lancer_interface() -> None:
    """Ouvre la fenêtre principale et attend que l'utilisateur la ferme."""
    _activer_haute_resolution()
    racine = tk.Tk()
    racine.withdraw()
    application = _charger_application(racine)
    if application is None:
        racine.destroy()
        return
    FenetrePrincipale(racine, application)
    racine.deiconify()
    racine.mainloop()
