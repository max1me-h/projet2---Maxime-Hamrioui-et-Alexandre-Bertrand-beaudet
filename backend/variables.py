"""Variables d'entrée du modèle : chacune sait valider la saisie d'un utilisateur.

Hiérarchie :
    Variable (abstraite)
    ├── VariableNumerique        champ texte, nombre dans un intervalle
    └── VariableCategorielle     liste déroulante de valeurs possibles
        └── VariableOrdinale     liste déroulante d'une échelle codée par des entiers
"""

import math
from abc import ABC, abstractmethod

from backend.exceptions import DonneeInvalideError

NON_RENSEIGNE = "— Non renseigné —"


class Variable(ABC):
    """Caractéristique d'un employé que l'utilisateur peut saisir."""

    # Façon dont le modèle reçoit la variable : « numerique » ou « categoriel ».
    encodage = "numerique"

    def __init__(self, nom: str, libelle: str, groupe: str, valeur_typique: object) -> None:
        """Reçoit le nom de colonne, le libellé affiché, le groupe du formulaire et la
        valeur typique (médiane ou valeur la plus fréquente, affichée comme repère)."""
        self._nom = nom
        self._libelle = libelle
        self._groupe = groupe
        self._valeur_typique = valeur_typique

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Libellé en français affiché dans l'interface."""
        return self._libelle

    @property
    def groupe(self) -> str:
        """Section du formulaire où la variable apparaît."""
        return self._groupe

    @property
    def valeur_typique(self) -> object:
        """Médiane ou valeur la plus fréquente, affichée comme repère dans le formulaire."""
        return self._valeur_typique

    def lire(self, saisie: str | None) -> object | None:
        """Renvoie None si le champ est vide (non renseigné), sinon la valeur validée.

        Lève DonneeInvalideError si la saisie n'est pas vide mais invalide.
        """
        if saisie is None or saisie.strip() in ("", NON_RENSEIGNE):
            return None
        return self.valider(saisie)

    @abstractmethod
    def valider(self, saisie: str) -> object:
        """Convertit la saisie texte en valeur pour le modèle.

        Lève DonneeInvalideError si la saisie est vide, mal formée ou hors des valeurs permises.
        """

    @abstractmethod
    def formater(self, valeur: object) -> str:
        """Renvoie le texte à afficher dans le formulaire pour une valeur du jeu de données."""

    @abstractmethod
    def aide(self) -> str:
        """Renvoie un court texte qui décrit les valeurs acceptées."""

    @abstractmethod
    def vers_dict(self) -> dict:
        """Renvoie un dictionnaire sérialisable en JSON qui décrit la variable."""

    def _erreur(self, message: str) -> DonneeInvalideError:
        """Construit une erreur dont le message commence par le libellé de la variable."""
        return DonneeInvalideError(f"{self._libelle} : {message}", self._nom)

    def __str__(self) -> str:
        return f"{self._libelle} ({self.aide()})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nom={self._nom!r})"

    def __eq__(self, autre: object) -> bool:
        if not isinstance(autre, Variable):
            return NotImplemented
        return self.vers_dict() == autre.vers_dict()

    def __hash__(self) -> int:
        return hash(self._nom)


class VariableNumerique(Variable):
    """Variable saisie comme un nombre compris dans un intervalle."""

    def __init__(self, nom: str, libelle: str, groupe: str, valeur_typique: float,
                 minimum: float, maximum: float, entier: bool = True) -> None:
        """Reçoit en plus les bornes permises et si la valeur doit être entière.

        Lève ValueError si minimum > maximum.
        """
        if minimum > maximum:
            raise ValueError(f"{nom} : minimum {minimum} supérieur au maximum {maximum}")
        super().__init__(nom, libelle, groupe, valeur_typique)
        self._minimum = minimum
        self._maximum = maximum
        self._entier = entier

    @property
    def minimum(self) -> float:
        """Plus petite valeur permise."""
        return self._minimum

    @property
    def maximum(self) -> float:
        """Plus grande valeur permise."""
        return self._maximum

    def valider(self, saisie: str) -> float | int:
        """Accepte « 5 000 » ou « 3,5 » ; refuse le texte, les non-entiers et le hors-intervalle."""
        texte = saisie.strip().replace(" ", "").replace(" ", "").replace(",", ".")
        if texte == "":
            raise self._erreur("le champ est vide")
        try:
            valeur = float(texte)
        except ValueError:
            raise self._erreur(f"« {saisie.strip()} » n'est pas un nombre") from None
        if not math.isfinite(valeur):
            raise self._erreur(f"« {saisie.strip()} » n'est pas un nombre valide")
        if self._entier and not valeur.is_integer():
            raise self._erreur("la valeur doit être un nombre entier")
        if not self._minimum <= valeur <= self._maximum:
            raise self._erreur(f"la valeur doit être comprise entre "
                               f"{self.formater(self._minimum)} et {self.formater(self._maximum)}")
        return int(valeur) if self._entier else valeur

    def formater(self, valeur: object) -> str:
        """Écrit l'entier sans décimale, sinon le nombre tel quel."""
        nombre = float(valeur)
        return str(int(nombre)) if self._entier else f"{nombre:g}"

    def aide(self) -> str:
        return f"de {self.formater(self._minimum)} à {self.formater(self._maximum)}"

    def vers_dict(self) -> dict:
        return {"nom": self._nom, "libelle": self._libelle, "groupe": self._groupe,
                "type": "numerique", "valeur_typique": self._valeur_typique,
                "minimum": self._minimum, "maximum": self._maximum, "entier": self._entier}


class VariableCategorielle(Variable):
    """Variable choisie dans une liste de valeurs possibles."""

    encodage = "categoriel"

    def __init__(self, nom: str, libelle: str, groupe: str, valeur_typique: object,
                 valeurs: list, libelles_valeurs: dict | None = None) -> None:
        """Reçoit en plus les valeurs possibles et, au besoin, leur traduction affichée.

        Lève ValueError si la liste de valeurs est vide.
        """
        if not valeurs:
            raise ValueError(f"{nom} : aucune valeur possible")
        super().__init__(nom, libelle, groupe, valeur_typique)
        self._valeurs = list(valeurs)
        self._libelles_valeurs = dict(libelles_valeurs or {})

    @property
    def valeurs(self) -> list:
        """Valeurs possibles telles qu'elles apparaissent dans le jeu de données."""
        return list(self._valeurs)

    def choix(self) -> list[str]:
        """Textes proposés dans la liste déroulante, dans l'ordre des valeurs."""
        return [self.formater(valeur) for valeur in self._valeurs]

    def valider(self, saisie: str) -> object:
        """Accepte le texte affiché ou la valeur brute ; refuse tout le reste."""
        texte = saisie.strip()
        if texte == "":
            raise self._erreur("aucune valeur choisie")
        for valeur in self._valeurs:
            if texte in (self.formater(valeur), str(valeur)):
                return valeur
        raise self._erreur(f"« {texte} » ne fait pas partie des choix possibles")

    def formater(self, valeur: object) -> str:
        return self._libelles_valeurs.get(valeur, str(valeur))

    def aide(self) -> str:
        return f"{len(self._valeurs)} choix possibles"

    def vers_dict(self) -> dict:
        return {"nom": self._nom, "libelle": self._libelle, "groupe": self._groupe,
                "type": "categorielle", "valeur_typique": self._valeur_typique,
                "valeurs": self._valeurs,
                "libelles_valeurs": [self._libelles_valeurs.get(v, "") for v in self._valeurs]}


class VariableOrdinale(VariableCategorielle):
    """Échelle ordonnée codée par des entiers (ex. satisfaction de 1 à 4).

    Elle se saisit comme une catégorie, mais le modèle la reçoit comme un nombre,
    car l'ordre des niveaux a un sens.
    """

    encodage = "numerique"

    def formater(self, valeur: object) -> str:
        """Affiche « 3 — Élevée » quand un libellé existe, sinon seulement « 3 »."""
        texte = self._libelles_valeurs.get(valeur, "")
        return f"{valeur} — {texte}" if texte else str(valeur)

    def aide(self) -> str:
        return f"échelle de {self._valeurs[0]} à {self._valeurs[-1]}"

    def vers_dict(self) -> dict:
        description = super().vers_dict()
        description["type"] = "ordinale"
        return description


def variable_depuis_dict(description: dict) -> Variable:
    """Reconstruit une variable à partir de sa description JSON (voir vers_dict).

    Lève DonneeInvalideError si le type est inconnu ou si une clé obligatoire manque.
    """
    try:
        nature = description["type"]
        commun = (description["nom"], description["libelle"], description["groupe"],
                  description["valeur_typique"])
        if nature == "numerique":
            return VariableNumerique(*commun, description["minimum"], description["maximum"],
                                     description.get("entier", True))
        if nature in ("categorielle", "ordinale"):
            valeurs = description["valeurs"]
            libelles = {valeur: texte for valeur, texte
                        in zip(valeurs, description.get("libelles_valeurs", [])) if texte}
            classe = VariableOrdinale if nature == "ordinale" else VariableCategorielle
            return classe(*commun, valeurs, libelles)
    except KeyError as erreur:
        raise DonneeInvalideError(f"Description de variable incomplète : clé {erreur} absente") from None
    raise DonneeInvalideError(f"Type de variable inconnu : {nature!r}")
