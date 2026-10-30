"""Point d'entrée du projet.

    python -m src.main             ouvre l'interface graphique
    python -m src.main entrainer   entraîne le modèle et l'enregistre dans modeles/

« python src/main.py » (ou le bouton ▶ de VS Code sur ce fichier) fonctionne aussi.
"""

import argparse
import os
import sys

# Lancé comme simple fichier, Python cherche les modules dans src/ au lieu de la racine du
# dépôt : on remplace ce dossier par la racine pour que « import src.… » fonctionne.
if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.entrainement import entrainer_et_sauvegarder, resume_entrainement  # noqa: E402
from src.exceptions import ErreurProjet  # noqa: E402


def main(arguments: list[str] | None = None) -> int:
    """Analyse la ligne de commande, exécute la commande demandée et renvoie le code de sortie."""
    analyseur = argparse.ArgumentParser(
        description="Estimation du risque de départ d'un employé (INF1007, projet 2).")
    analyseur.add_argument("commande", nargs="?", choices=["interface", "entrainer"],
                           default="interface", help="interface (par défaut) ou entrainer")
    options = analyseur.parse_args(arguments)

    if options.commande == "entrainer":
        print("Entraînement en cours (environ 30 secondes)...")
        try:
            metadonnees = entrainer_et_sauvegarder()
        except ErreurProjet as erreur:
            print(f"Erreur : {erreur}", file=sys.stderr)
            return 1
        print(resume_entrainement(metadonnees))
        return 0

    # Import tardif : l'entraînement fonctionne même sans Tkinter (ex. serveur de CI).
    from src.interface import lancer_interface
    lancer_interface()
    return 0


if __name__ == "__main__":
    sys.exit(main())
