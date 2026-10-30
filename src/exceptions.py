"""Exceptions personnalisées du projet.

Toutes héritent de ErreurProjet : l'interface peut ainsi intercepter les erreurs
prévues sans jamais utiliser un except nu.
"""


class ErreurProjet(Exception):
    """Classe parente de toutes les erreurs prévues par le projet."""


class DonneeInvalideError(ErreurProjet):
    """Une donnée (saisie ou fichier) ne respecte pas le format attendu."""

    def __init__(self, message: str, variable: str | None = None) -> None:
        """Reçoit le message à afficher et, si connu, le nom de la variable fautive."""
        super().__init__(message)
        self.variable = variable


class SaisieInvalideError(DonneeInvalideError):
    """Un formulaire contient une ou plusieurs saisies invalides."""

    def __init__(self, erreurs: dict[str, str]) -> None:
        """Reçoit un dictionnaire {nom de variable : message d'erreur}."""
        self.erreurs = dict(erreurs)
        lignes = "\n".join(f"• {message}" for message in self.erreurs.values())
        super().__init__(f"Saisie invalide :\n{lignes}")


class FichierIntrouvableError(ErreurProjet):
    """Un fichier nécessaire (données, modèle, métadonnées) est absent."""

    def __init__(self, chemin: str, conseil: str = "") -> None:
        """Reçoit le chemin manquant et un conseil facultatif pour corriger le problème."""
        message = f"Fichier introuvable : {chemin}"
        if conseil:
            message += f"\n{conseil}"
        super().__init__(message)
        self.chemin = chemin


class ModeleNonEntraineError(ErreurProjet):
    """On demande une prédiction à un modèle qui n'a pas encore été entraîné."""
