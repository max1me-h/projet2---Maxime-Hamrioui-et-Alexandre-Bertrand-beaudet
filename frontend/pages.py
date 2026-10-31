"""Les quatre pages de la fenêtre : Prédiction, Le modèle, Historique, À propos.

Les pages n'effectuent aucun calcul : elles lisent les saisies, appellent l'Application
et affichent ce qu'elle renvoie.
"""

import json
from collections.abc import Callable
from tkinter import messagebox, ttk

import customtkinter as ctk

from frontend.composants import (Carte, Champ, Tuile, ZoneGraphique, bouton_principal,
                                            bouton_secondaire, etiquette_titre, selecteur, texte_long)
from frontend.graphiques import figure_effets, figure_position_risque, figure_precision_champs
from frontend.theme import (ACCENT, AVERTISSEMENT, BORDURE, CARTE, CARTE_RELIEF, ERREUR, NIVEAUX,
                                       SUCCES, SURVOL, TEXTE, TEXTE_DISCRET, TEXTE_SECONDAIRE, police)
from frontend.application import Application, Prediction, pourcentage
from backend.exceptions import DonneeInvalideError, SaisieInvalideError
from backend.graphiques import (figure_comparaison_cibles, figure_importances,
                                            figure_matrice_confusion)

THEME = "sombre"


def entete(page: ctk.CTkFrame, titre: str, sous_titre: str) -> ctk.CTkFrame:
    """Titre de page et phrase d'introduction."""
    cadre = ctk.CTkFrame(page, fg_color="transparent")
    ctk.CTkLabel(cadre, text=titre, font=police(26, True), text_color=TEXTE).pack(anchor="w")
    texte_long(cadre, sous_titre, 13, conteneur=page, marge=20).pack(anchor="w", fill="x")
    return cadre


class PagePrediction(ctk.CTkFrame):
    """Formulaire à gauche ; risque, précision attendue et graphique à droite."""

    GRAPHIQUES = {"Précision selon les champs": "precision", "Facteurs": "effets",
                  "Position": "position"}

    def __init__(self, parent: ctk.CTkBaseClass, application: Application,
                 apres_enregistrement: Callable[[], None]) -> None:
        """apres_enregistrement est appelé après chaque prédiction ajoutée à l'historique."""
        super().__init__(parent, fg_color="transparent")
        self._application = application
        self._apres_enregistrement = apres_enregistrement
        self._champs: dict[str, Champ] = {}
        self._prediction: Prediction | None = None
        self._saisies_predites: dict[str, str] = {}
        self._minuterie: str | None = None
        self._graphique = "precision"
        self.columnconfigure(0, weight=11, uniform="colonnes")
        self.columnconfigure(1, weight=10, uniform="colonnes")
        self.rowconfigure(0, weight=1)
        self._construire_formulaire()
        self._construire_resultats()
        self.maj_indicateur()

    # ----- Construction ---------------------------------------------------------------

    def _construire_formulaire(self) -> None:
        """Carte de gauche : titre, exemples, champs défilants, boutons."""
        carte = Carte(self)
        carte.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        carte.columnconfigure(0, weight=1)
        carte.rowconfigure(2, weight=1)

        haut = ctk.CTkFrame(carte, fg_color="transparent")
        haut.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 2))
        ctk.CTkLabel(haut, text="Décrivez l'employé", font=police(22, True), text_color=TEXTE).pack(anchor="w")
        texte_long(haut, "Remplissez autant de champs que vous voulez : un champ vide est permis, "
                         "mais la précision attendue baisse.", 13, conteneur=carte, marge=60).pack(anchor="w", fill="x")
        exemples = ctk.CTkFrame(carte, fg_color="transparent")
        exemples.grid(row=1, column=0, sticky="w", padx=22, pady=(8, 10))
        ctk.CTkLabel(exemples, text="Exemples réels", font=police(12), text_color=TEXTE_DISCRET).pack(side="left",
                                                                                                    padx=(0, 10))
        for texte, genre in (("À risque", "risque"), ("Stable", "stable"), ("Au hasard", "hasard")):
            bouton_secondaire(exemples, texte, lambda g=genre: self._charger_exemple(g), width=92, height=30,
                              font=police(12)).pack(side="left", padx=3)

        self._zone_champs = ctk.CTkScrollableFrame(carte, fg_color=CARTE, corner_radius=0,
                                                   scrollbar_button_color=BORDURE,
                                                   scrollbar_button_hover_color=SURVOL)
        self._zone_champs.grid(row=2, column=0, sticky="nsew", padx=(4, 6))
        self._zone_champs.columnconfigure(0, weight=1)
        ligne = 0
        for groupe, variables in self._application.variables_par_groupe().items():
            ctk.CTkLabel(self._zone_champs, text=groupe.upper(), font=police(12, True), text_color=ACCENT).grid(
                row=ligne, column=0, columnspan=2, sticky="w", padx=18, pady=(22 if ligne else 6, 0))
            ligne += 1
            for variable in variables:
                self._champs[variable.nom] = Champ(self._zone_champs, variable, ligne, self._planifier_maj)
                ligne += 2

        ctk.CTkFrame(carte, height=1, fg_color=BORDURE).grid(row=3, column=0, sticky="ew", padx=1)
        pied = ctk.CTkFrame(carte, fg_color="transparent")
        pied.grid(row=4, column=0, sticky="ew", padx=22, pady=14)
        pied.columnconfigure(0, weight=1)
        self._message = texte_long(pied, "", 12, conteneur=pied, marge=320)
        self._message.grid(row=0, column=0, sticky="w")
        bouton_secondaire(pied, "Tout effacer", self._effacer, width=120).grid(row=0, column=1, padx=10)
        bouton_principal(pied, "Prédire  →", self.predire, width=150).grid(row=0, column=2)

    def _construire_resultats(self) -> None:
        """Colonne de droite : carte du risque, carte de la précision, carte du graphique."""
        colonne = ctk.CTkFrame(self, fg_color="transparent")
        colonne.grid(row=0, column=1, sticky="nsew")
        colonne.columnconfigure(0, weight=1)

        # Une seule carte : le risque à gauche, la précision attendue à droite, les phrases dessous.
        carte = Carte(colonne)
        carte.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        carte.columnconfigure((0, 1), weight=1, uniform="indicateurs")

        risque = ctk.CTkFrame(carte, fg_color="transparent")
        risque.grid(row=0, column=0, sticky="ew", padx=(20, 12), pady=(14, 0))
        etiquette_titre(risque, "Risque de départ").pack(anchor="w")
        self._risque = ctk.CTkLabel(risque, text="—", font=police(38, True), text_color=TEXTE_DISCRET, height=46)
        self._risque.pack(anchor="w")
        self._badge = ctk.CTkLabel(risque, text="", font=police(12, True), corner_radius=10, height=26,
                                   fg_color="transparent")
        self._badge.pack(anchor="w", pady=(0, 6))
        self._barre_risque = ctk.CTkProgressBar(risque, height=8, corner_radius=4, fg_color=SURVOL,
                                                progress_color=TEXTE_DISCRET)
        self._barre_risque.set(0)
        self._barre_risque.pack(fill="x")

        precision = ctk.CTkFrame(carte, fg_color="transparent")
        precision.grid(row=0, column=1, sticky="ew", padx=(12, 20), pady=(14, 0))
        etiquette_titre(precision, "Précision attendue").pack(anchor="w")
        self._precision = ctk.CTkLabel(precision, text="50 %", font=police(38, True), text_color=ACCENT,
                                       height=46)
        self._precision.pack(anchor="w")
        self._nb_champs = ctk.CTkLabel(precision, text="", font=police(12), text_color=TEXTE_SECONDAIRE,
                                       height=26)
        self._nb_champs.pack(anchor="w", pady=(0, 6))
        self._barre_champs = ctk.CTkProgressBar(precision, height=8, corner_radius=4, fg_color=SURVOL,
                                                progress_color=ACCENT)
        self._barre_champs.pack(fill="x")

        textes = ctk.CTkFrame(carte, fg_color="transparent")
        textes.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(12, 14))
        self._phrase = texte_long(textes, "Remplissez le formulaire, puis cliquez sur « Prédire ».", 13,
                                  conteneur=carte)
        self._phrase.pack(anchor="w", fill="x")
        # Affiché seulement quand le formulaire a changé depuis la prédiction (voir maj_indicateur).
        self._perime = texte_long(textes, "Le formulaire a changé depuis cette prédiction : cliquez sur "
                                          "« Prédire » pour la mettre à jour.", 12, AVERTISSEMENT, conteneur=carte)
        self._conseil = texte_long(textes, "", 12, conteneur=carte)
        self._conseil.pack(anchor="w", fill="x", pady=(4, 0))
        meta = self._application.metadonnees
        texte_long(textes, f"Précision mesurée sur {meta['lignes']['test']} employés jamais vus, avec les mêmes "
                           f"champs remplis que vous (50 % = hasard ; tous les champs : "
                           f"{pourcentage(meta['courbe_precision']['ordre_importance'][-1])}).",
                   11, TEXTE_DISCRET, conteneur=carte).pack(anchor="w", fill="x", pady=(2, 0))

        carte = Carte(colonne)
        carte.grid(row=1, column=0, sticky="nsew")
        colonne.rowconfigure(1, weight=1, minsize=300)
        selecteur(carte, list(self.GRAPHIQUES), self._choisir_graphique).pack(anchor="w", padx=16, pady=(14, 4))
        self._zone_graphique = ZoneGraphique(carte)
        self._zone_graphique.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ----- Réactions aux saisies ------------------------------------------------------

    def _saisies(self) -> dict[str, str]:
        """Texte de chaque champ du formulaire."""
        return {nom: champ.texte() for nom, champ in self._champs.items()}

    def _planifier_maj(self) -> None:
        """Met à jour l'indicateur 250 ms après la dernière frappe (évite de recalculer à chaque touche)."""
        if self._minuterie is not None:
            self.after_cancel(self._minuterie)
        self._minuterie = self.after(250, self.maj_indicateur)

    def maj_indicateur(self) -> None:
        """Champs remplis, précision attendue, conseils et graphique de précision."""
        self._minuterie = None
        saisies = self._saisies()
        champs = self._application.champs_renseignes(saisies)
        total = len(self._champs)
        self._precision.configure(text=pourcentage(self._application.precision_attendue(champs)))
        self._nb_champs.configure(text=f"{len(champs)} / {total} champs remplis")
        self._barre_champs.set(len(champs) / total)
        conseilles = self._application.champs_conseilles(champs)
        self._conseil.configure(
            text="Pour gagner en précision, remplissez : " + ", ".join(v.libelle for v in conseilles)
            if conseilles else "Tous les champs sont remplis.")
        if self._prediction is not None and saisies != self._saisies_predites:
            self._perime.pack(anchor="w", fill="x", after=self._phrase)
        else:
            self._perime.pack_forget()
        if self._graphique == "precision":
            self._afficher_graphique()

    def _choisir_graphique(self, choix: str) -> None:
        """Appelé par le sélecteur de graphique."""
        self._graphique = self.GRAPHIQUES[choix]
        self._afficher_graphique()

    def _afficher_graphique(self) -> None:
        """Dessine le graphique choisi."""
        if self._graphique == "precision":
            champs = self._application.champs_renseignes(self._saisies())
            figure = figure_precision_champs(self._application.metadonnees["courbe_precision"], len(champs),
                                             self._application.precision_attendue(champs), theme=THEME)
        elif self._graphique == "effets":
            if self._prediction is None:
                figure = figure_effets({}, message_vide="Cliquez sur « Prédire » pour voir ce qui influence "
                                                        "le résultat", theme=THEME)
            else:
                valeurs = {nom: self._application.variable(nom).formater(valeur)
                           for nom, valeur in self._prediction.valeurs_connues.items()}
                figure = figure_effets(self._prediction.effets, valeurs, theme=THEME)
        else:
            scores, departs = self._application.scores_test()
            figure = figure_position_risque(scores, departs,
                                            self._prediction.probabilite if self._prediction else None, theme=THEME)
        self._zone_graphique.afficher(figure)

    # ----- Actions --------------------------------------------------------------------

    def predire(self) -> None:
        """Valide le formulaire, affiche la prédiction et l'ajoute à l'historique."""
        for champ in self._champs.values():
            champ.marquer_erreur(None)
        saisies = self._saisies()
        try:
            prediction = self._application.predire(saisies, enregistrer=False)
        except SaisieInvalideError as erreur:
            fautifs = [nom for nom in self._champs if nom in erreur.erreurs]
            for nom in fautifs:
                self._champs[nom].marquer_erreur(erreur.erreurs[nom])
            if fautifs:
                self._defiler_vers(self._champs[fautifs[0]])
            self._afficher_message(f"{len(erreur.erreurs)} saisie(s) à corriger : voir les champs en rouge.",
                                   ERREUR)
            return
        except DonneeInvalideError as erreur:
            self._afficher_message(str(erreur), ERREUR)
            return
        self._prediction, self._saisies_predites = prediction, saisies
        self._afficher_prediction(prediction)
        try:
            self._application.enregistrer(prediction)
        except OSError as erreur:
            self._afficher_message(f"Prédiction affichée, mais l'historique n'a pas pu être enregistré "
                                   f"(fichier ouvert ailleurs ?) : {erreur}", ERREUR)
        else:
            self._afficher_message(f"Prédiction enregistrée dans l'historique à {prediction.date:%H:%M:%S}.",
                                   SUCCES)
            self._apres_enregistrement()

    def _afficher_prediction(self, prediction: Prediction) -> None:
        """Risque, badge coloré, barre et phrase compréhensible par un non-informaticien."""
        couleur, fond = NIVEAUX[prediction.niveau]
        self._risque.configure(text=pourcentage(prediction.probabilite), text_color=couleur)
        self._badge.configure(text=f"   Risque {prediction.niveau} · {prediction.conclusion()}   ",
                              fg_color=fond, text_color=couleur)
        self._barre_risque.configure(progress_color=couleur)
        self._barre_risque.set(prediction.probabilite)
        vides = prediction.nb_variables - prediction.nb_champs
        phrase = (f"Un employé avec ce profil présente un risque {prediction.niveau} de quitter l'entreprise. "
                  f"Avec {prediction.nb_champs} champ(s) rempli(s) sur {prediction.nb_variables}, le modèle a "
                  f"raison dans environ {pourcentage(prediction.precision_attendue)} des cas.")
        if vides:
            phrase += f" Les {vides} champ(s) vide(s) sont estimés à partir d'employés de référence."
        self._phrase.configure(text=phrase)
        self.maj_indicateur()
        if self._graphique != "precision":
            self._afficher_graphique()

    def _afficher_message(self, texte: str, couleur: str = TEXTE_SECONDAIRE) -> None:
        """Message à gauche des boutons (erreur, confirmation...)."""
        self._message.configure(text=texte, text_color=couleur)

    def _effacer(self) -> None:
        """Vide le formulaire et le résultat."""
        for champ in self._champs.values():
            champ.effacer()
        self._prediction, self._saisies_predites = None, {}
        self._risque.configure(text="—", text_color=TEXTE_DISCRET)
        self._badge.configure(text="", fg_color="transparent")
        self._barre_risque.configure(progress_color=TEXTE_DISCRET)
        self._barre_risque.set(0)
        self._phrase.configure(text="Remplissez le formulaire, puis cliquez sur « Prédire ».")
        self._afficher_message("")
        self.maj_indicateur()
        if self._graphique != "precision":
            self._afficher_graphique()

    def _charger_exemple(self, genre: str) -> None:
        """Remplit le formulaire avec un vrai employé du jeu de test."""
        saisies, parti = self._application.exemple(genre)
        for nom, texte in saisies.items():
            self._champs[nom].definir(texte)
        realite = "a réellement quitté l'entreprise" if parti else "est réellement resté"
        self._afficher_message(f"Exemple chargé : un véritable employé du jeu de test, qui {realite}. "
                               f"Cliquez sur « Prédire » pour comparer.")

    def _defiler_vers(self, champ: Champ) -> None:
        """Fait défiler le formulaire jusqu'au champ et y place le curseur."""
        self.update_idletasks()
        hauteur = self._zone_champs.winfo_height()
        if hauteur > 0:
            position = max(0.0, (champ.widget.winfo_y() - 80) / hauteur)
            self._zone_champs._parent_canvas.yview_moveto(position)  # canevas interne du cadre défilant
        champ.donner_focus()


class PageModele(ctk.CTkFrame):
    """Indicateurs clés, comparaison des modèles, explications et graphiques d'évaluation."""

    GRAPHIQUES = {"Matrice de confusion": "matrice", "Variables utiles": "importances",
                  "Satisfaction ?": "cibles"}

    def __init__(self, parent: ctk.CTkBaseClass, application: Application) -> None:
        super().__init__(parent, fg_color="transparent")
        self._meta = application.metadonnees
        meta = self._meta
        retenu = meta["metriques_test"][meta["modele"]]
        reference = meta["metriques_test"]["Référence (classe majoritaire)"]
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        entete(self, f"Le modèle : {meta['modele'].lower()}",
               f"Choisi par validation croisée sur {meta['lignes']['entrainement']} employés d'entraînement, "
               f"puis évalué une seule fois sur {meta['lignes']['test']} employés de test. Entraîné le "
               f"{meta['date_entrainement'].replace('T', ' à ')} en {meta['duree_entrainement_s']:.2f} s."
               ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        tuiles = ctk.CTkFrame(self, fg_color="transparent")
        tuiles.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        departs_test = sum(meta["matrice_confusion"][1])
        donnees_tuiles = [
            ("Exactitude équilibrée", pourcentage(retenu["exactitude_equilibree"]),
             f"Référence : {pourcentage(reference['exactitude_equilibree'])}", ACCENT),
            ("Validation croisée", pourcentage(meta["validation_croisee"][meta["modele"]]),
             "5 plis, sur l'entraînement", TEXTE),
            ("AUC", f"{retenu.get('auc', 0.5):.2f}".replace(".", ","), "0,50 = hasard", TEXTE),
            ("Départs détectés", pourcentage(retenu["rappel_depart"]), f"sur {departs_test} départs du test", TEXTE),
        ]
        for colonne, (titre, valeur, detail, couleur) in enumerate(donnees_tuiles):
            tuiles.columnconfigure(colonne, weight=1, uniform="tuiles")
            Tuile(tuiles, titre, valeur, detail, couleur).grid(row=0, column=colonne, sticky="ew",
                                                               padx=(0 if colonne == 0 else 6, 0))

        bas = ctk.CTkFrame(self, fg_color="transparent")
        bas.grid(row=2, column=0, sticky="nsew")
        bas.columnconfigure(0, weight=5, uniform="bas")
        bas.columnconfigure(1, weight=6, uniform="bas")
        bas.rowconfigure(0, weight=1)
        self._construire_details(bas)

        carte = Carte(bas)
        carte.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        selecteur(carte, list(self.GRAPHIQUES), self._choisir_graphique).pack(anchor="w", padx=16, pady=(14, 4))
        self._zone_graphique = ZoneGraphique(carte)
        self._zone_graphique.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._choisir_graphique(next(iter(self.GRAPHIQUES)))

    def _construire_details(self, parent: ctk.CTkFrame) -> None:
        """Carte de gauche : tableau comparatif, définitions, limites, nettoyage."""
        meta = self._meta
        carte = Carte(parent)
        carte.grid(row=0, column=0, sticky="nsew")
        zone = ctk.CTkScrollableFrame(carte, fg_color=CARTE, scrollbar_button_color=BORDURE,
                                      scrollbar_button_hover_color=SURVOL)
        zone.pack(fill="both", expand=True, padx=6, pady=6)
        etiquette_titre(zone, "Comparaison des modèles (jeu de test)").pack(anchor="w", padx=12, pady=(10, 6))
        echelle = ctk.ScalingTracker.get_widget_scaling(self)
        colonnes = {"modele": ("Modèle", 170), "equilibree": ("Ex. équil.", 72), "validation": ("Valid. cr.", 72),
                    "exactitude": ("Exactitude", 72), "auc": ("AUC", 48)}
        tableau = ttk.Treeview(zone, columns=list(colonnes), show="headings", height=3, selectmode="none",
                               style="Sombre.Treeview")
        for cle, (titre, largeur) in colonnes.items():
            tableau.heading(cle, text=titre)
            tableau.column(cle, width=int(largeur * echelle), anchor="w" if cle == "modele" else "center")
        tableau.tag_configure("retenu", foreground=ACCENT)
        for nom, resultats in meta["metriques_test"].items():
            tableau.insert("", "end", tags=("retenu",) if nom == meta["modele"] else (), values=(
                nom, pourcentage(resultats["exactitude_equilibree"]), pourcentage(meta["validation_croisee"][nom]),
                pourcentage(resultats["exactitude"]), f"{resultats.get('auc', 0.5):.2f}".replace(".", ",")))
        tableau.pack(fill="x", padx=12)

        satisfaction = meta["comparaison_cibles"]["Satisfaction (≥ 3 sur 4)"]
        sections = [
            ("Exactitude équilibrée", "Moyenne entre le pourcentage de départs bien détectés et celui des maintiens "
             "bien détectés. Répondre toujours la même chose donne 50 %. L'exactitude simple est trompeuse ici : "
             f"répondre toujours « reste » donne déjà "
             f"{pourcentage(meta['metriques_test']['Référence (classe majoritaire)']['exactitude'])}."),
            ("Pourquoi pas la satisfaction ?", "Prédire si un employé est satisfait (3 ou 4 sur 4) donne "
             + " et ".join(f"{pourcentage(valeur)} ({nom.lower()})" for nom, valeur in satisfaction.items())
             + " : aucun modèle ne fait mieux que le hasard. Le départ, lui, est prévisible."),
            ("Limites", "Seulement 59 départs dans le jeu de test (environ ± 3 points d'incertitude), données "
             "fictives, modèle linéaire : il se trompe pour environ un employé sur quatre."),
            ("Nettoyage des données", "\n".join(f"• {etape}" for etape in meta["nettoyage"])),
        ]
        for titre, texte in sections:
            etiquette_titre(zone, titre).pack(anchor="w", padx=12, pady=(16, 2))
            texte_long(zone, texte, 12, conteneur=carte, marge=70).pack(anchor="w", fill="x", padx=12)

    def _choisir_graphique(self, choix: str) -> None:
        """Dessine le graphique d'évaluation choisi."""
        meta = self._meta
        graphique = self.GRAPHIQUES[choix]
        if graphique == "matrice":
            figure = figure_matrice_confusion(meta["matrice_confusion"], theme=THEME)
        elif graphique == "importances":
            figure = figure_importances(meta["importances"], theme=THEME)
        else:
            figure = figure_comparaison_cibles(meta["comparaison_cibles"], theme=THEME)
        self._zone_graphique.afficher(figure)


class PageHistorique(ctk.CTkFrame):
    """Tableau des prédictions passées et détail de la ligne sélectionnée."""

    def __init__(self, parent: ctk.CTkBaseClass, application: Application) -> None:
        super().__init__(parent, fg_color="transparent")
        self._application = application
        self._lignes: dict[str, dict[str, str]] = {}
        self.columnconfigure(0, weight=3, uniform="colonnes")
        self.columnconfigure(1, weight=2, uniform="colonnes")
        self.rowconfigure(1, weight=1)

        haut = ctk.CTkFrame(self, fg_color="transparent")
        haut.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        haut.columnconfigure(0, weight=1)
        ctk.CTkLabel(haut, text="Historique", font=police(26, True), text_color=TEXTE).grid(row=0, column=0,
                                                                                         sticky="w")
        self._compte = ctk.CTkLabel(haut, text="", font=police(13), text_color=TEXTE_SECONDAIRE)
        self._compte.grid(row=1, column=0, sticky="w")
        bouton_secondaire(haut, "Rafraîchir", self.rafraichir, width=120).grid(row=0, column=1, rowspan=2, padx=8)
        bouton_secondaire(haut, "Effacer l'historique", self._effacer, couleur=ERREUR, width=170).grid(
            row=0, column=2, rowspan=2)

        carte = Carte(self)
        carte.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        echelle = ctk.ScalingTracker.get_widget_scaling(self)
        colonnes = {"date": ("Date", 150), "risque": ("Risque", 70), "conclusion": ("Conclusion", 130),
                    "champs": ("Champs", 64), "precision": ("Précision", 76)}
        self._tableau = ttk.Treeview(carte, columns=list(colonnes), show="headings", selectmode="browse",
                                     style="Sombre.Treeview")
        for cle, (titre, largeur) in colonnes.items():
            self._tableau.heading(cle, text=titre)
            self._tableau.column(cle, width=int(largeur * echelle), anchor="w" if cle == "date" else "center")
        self._tableau.tag_configure("impaire", background=CARTE_RELIEF)
        self._tableau.tag_configure("depart", foreground=NIVEAUX["élevé"][0])
        self._tableau.tag_configure("maintien", foreground=NIVEAUX["faible"][0])
        barre = ctk.CTkScrollbar(carte, command=self._tableau.yview, button_color=BORDURE,
                                 button_hover_color=SURVOL)
        self._tableau.configure(yscrollcommand=barre.set)
        barre.pack(side="right", fill="y", padx=(0, 6), pady=10)
        self._tableau.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        self._tableau.bind("<<TreeviewSelect>>", self._afficher_detail)

        detail = Carte(self)
        detail.grid(row=1, column=1, sticky="nsew")
        etiquette_titre(detail, "Saisies de la prédiction sélectionnée").pack(anchor="w", padx=18, pady=(16, 6))
        self._detail = ctk.CTkTextbox(detail, fg_color=CARTE, text_color=TEXTE, font=police(13), wrap="word",
                                      activate_scrollbars=True, scrollbar_button_color=BORDURE)
        self._detail.pack(fill="both", expand=True, padx=10, pady=(0, 12))
        self.rafraichir()

    def rafraichir(self) -> None:
        """Relit le fichier d'historique et remplit le tableau."""
        try:
            lignes = self._application.historique.lire()
        except (DonneeInvalideError, OSError) as erreur:
            self._compte.configure(text=f"Historique illisible : {erreur}", text_color=ERREUR)
            return
        self._tableau.delete(*self._tableau.get_children())
        self._lignes.clear()
        for rang, ligne in enumerate(lignes):
            etiquettes = ["depart" if ligne["conclusion"] == "départ probable" else "maintien"]
            if rang % 2:
                etiquettes.append("impaire")
            identifiant = self._tableau.insert("", "end", tags=etiquettes, values=(
                ligne["date"].replace("T", "  "), self._format_pourcentage(ligne["risque_depart"]),
                ligne["conclusion"], ligne["champs_renseignes"], self._format_pourcentage(ligne["precision_attendue"])))
            self._lignes[identifiant] = ligne
        self._compte.configure(text=f"{len(lignes)} prédiction(s) enregistrée(s) · sélectionnez une ligne pour "
                                    f"revoir ses saisies" if lignes else "Aucune prédiction pour l'instant.",
                               text_color=TEXTE_SECONDAIRE)
        self._ecrire_detail("")

    @staticmethod
    def _format_pourcentage(texte: str) -> str:
        """« 0.684 » -> « 68 % » ; le texte tel quel s'il n'est pas un nombre."""
        try:
            return pourcentage(float(texte))
        except ValueError:
            return texte

    def _afficher_detail(self, _evenement=None) -> None:
        """Affiche les saisies de la ligne choisie, avec les libellés en français."""
        selection = self._tableau.selection()
        if not selection:
            return
        texte_saisies = self._lignes[selection[0]]["saisies"]
        try:
            saisies = json.loads(texte_saisies)
            lignes = [f"{self._application.variable(nom).libelle} : "
                      f"{self._application.variable(nom).formater(valeur)}" for nom, valeur in saisies.items()]
        except (json.JSONDecodeError, AttributeError, DonneeInvalideError):
            lignes = [texte_saisies]
        self._ecrire_detail("\n".join(lignes))

    def _ecrire_detail(self, texte: str) -> None:
        """Remplace le contenu de la zone de détail (en lecture seule)."""
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", texte)
        self._detail.configure(state="disabled")

    def _effacer(self) -> None:
        """Supprime le fichier d'historique après confirmation."""
        if not messagebox.askyesno("Effacer l'historique",
                                   "Supprimer définitivement toutes les prédictions enregistrées ?", parent=self):
            return
        try:
            self._application.historique.effacer()
        except OSError as erreur:
            messagebox.showerror("Effacer l'historique", f"Impossible de supprimer le fichier : {erreur}",
                                 parent=self)
        self.rafraichir()


class PageAPropos(ctk.CTkScrollableFrame):
    """Explications pour l'utilisateur, une carte par question."""

    SECTIONS = [
        ("Que fait cette application ?",
         "Elle estime le risque qu'un employé quitte son entreprise à partir de 26 caractéristiques de sa vie "
         "professionnelle et personnelle : âge, état civil, poste, salaire, heures supplémentaires, ancienneté, "
         "satisfaction, équilibre travail-vie personnelle, etc."),
        ("Avertissement",
         "Ceci est un exercice pédagogique, pas une aide à la décision. Les données sont fictives (créées par des "
         "scientifiques des données d'IBM) et le modèle se trompe souvent. Plusieurs variables, comme l'âge et "
         "l'état civil, sont des motifs de discrimination interdits par la Charte des droits et libertés de la "
         "personne du Québec (article 10) : utiliser un tel outil pour embaucher, promouvoir ou congédier serait "
         "injuste et illégal. L'accident du Therac-25 rappelle qu'une confiance aveugle dans un logiciel peut "
         "avoir des conséquences graves."),
        ("Comment lire le « risque de départ » ?",
         "Pendant l'entraînement, le modèle accorde autant de poids aux départs (16 % des employés) qu'aux "
         "maintiens, pour ne pas se contenter de répondre toujours « reste ». Le pourcentage affiché est donc "
         "un indice de risque plutôt qu'une fréquence réelle : au-delà de 50 %, le modèle conclut à un départ "
         "probable."),
        ("Comment est calculée la « précision attendue » ?",
         "Quand un champ est vide, on ne connaît pas sa valeur : le modèle essaie les valeurs de 40 vrais "
         "employés de référence et fait la moyenne des risques obtenus. Le même calcul est appliqué aux employés "
         "du jeu de test (jamais vus à l'entraînement) en ne gardant que les champs que vous avez remplis. La "
         "précision attendue est l'exactitude équilibrée obtenue : la moyenne entre le pourcentage de départs "
         "bien détectés et celui des maintiens bien détectés. 50 % correspond au hasard. Le jeu de test ne "
         "contient que 59 départs : ce pourcentage varie d'environ ± 3 points selon l'échantillon."),
        ("Pourquoi ne pas prédire directement la satisfaction ?",
         "Nous l'avons essayé : la satisfaction au travail n'a presque aucun lien avec les autres variables de "
         "ce jeu de données (corrélations inférieures à 0,11). Le meilleur modèle faisait moins bien que la "
         "référence triviale (voir la page « Le modèle »). Le départ, lui, est l'expression mesurable du "
         "contentement, et il est prévisible."),
        ("Source des données",
         "« IBM HR Analytics Employee Attrition & Performance », Kaggle (pavansubhasht), licence Open Database "
         "License (base) et Database Contents License (contenu). 1470 employés fictifs, 35 colonnes."),
        ("Auteurs", "Maxime Hamrioui et Alexandre Bertrand-Beaudet — INF1007, automne 2026."),
    ]

    def __init__(self, parent: ctk.CTkBaseClass) -> None:
        super().__init__(parent, fg_color="transparent", scrollbar_button_color=BORDURE,
                         scrollbar_button_hover_color=SURVOL)
        entete(self, "À propos", "Ce que fait l'application, ses limites et la façon de lire ses résultats.").pack(
            anchor="w", fill="x", padx=4, pady=(0, 14))
        for titre, texte in self.SECTIONS:
            avertissement = titre == "Avertissement"
            carte = Carte(self, **({"border_color": AVERTISSEMENT} if avertissement else {}))
            carte.pack(fill="x", padx=4, pady=(0, 10))
            ctk.CTkLabel(carte, text=titre, font=police(16, True),
                         text_color=AVERTISSEMENT if avertissement else TEXTE).pack(anchor="w", padx=20, pady=(14, 2))
            texte_long(carte, texte, 13, conteneur=carte, marge=50).pack(anchor="w", fill="x", padx=20,
                                                                         pady=(0, 16))
