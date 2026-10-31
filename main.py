"""Point d'entrée du projet.

    python main.py             ouvre l'interface graphique
    python main.py entrainer   entraîne le modèle et l'enregistre dans backend/sauvegarde/
"""

import argparse
import sys

from backend.entrainement import entrainer_et_sauvegarder, resume_entrainement
from backend.exceptions import ErreurProjet


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

    # Import tardif : l'entraînement fonctionne même sans interface graphique (ex. serveur de CI).
    from frontend.fenetre import lancer_interface
    lancer_interface()
    return 0


if __name__ == "__main__":
    sys.exit(main())
