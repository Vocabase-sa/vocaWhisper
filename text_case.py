"""
Restauration de la casse et de la ponctuation.
===============================================

Les moteurs CTC (wav2vec2) et certains modèles NeMo (LinTO) transcrivent en
minuscules, sans ponctuation : c'est le principe de leur vocabulaire, qui ne
contient que des caractères. « je voudrais le docteur brisbois » est donc une
sortie correcte, pas une erreur.

Ce module remet en forme en aval, en trois passes :

    1. Noms propres — correspondance EXACTE (insensible à la casse) avec
       noms_propres.txt, y compris les noms composés (« de ronge » -> « DE RONGE »).
       Complémentaire du fuzzy : celui-ci rattrape les approximations, celui-là
       garantit la casse de ce qui est déjà bien reconnu.
    2. Majuscule en début de phrase.
    3. Ponctuation finale si elle manque.

Le traitement est IDEMPOTENT : appliqué à un texte déjà mis en forme (sortie
Whisper), il ne le dégrade pas.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOMS_PROPRES_FILE = os.path.join(BASE_DIR, "noms_propres.txt")

# Mots qui ne doivent jamais être capitalisés comme des noms, même présents
# dans l'annuaire : ce sont des titres, pas des identités.
_TITLES = frozenset({
    "docteur", "doctor", "dr", "monsieur", "madame", "mme", "mr",
    "professeur", "pr", "maitre", "maître",
})


def load_names(path=None):
    """Charge les noms propres, du plus long au plus court.

    L'ordre importe : « DE RONGE » doit être testé avant « RONGE », sans quoi
    seule la seconde moitié serait capitalisée.
    """
    path = path or NOMS_PROPRES_FILE
    if not os.path.isfile(path):
        return []

    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)

    return sorted(names, key=lambda n: len(n.split()), reverse=True)


def format_name(name: str, style: str = "title") -> str:
    """Met un nom à la casse demandée.

    « title » (défaut) : Brisbois, De Ronge — première lettre de chaque mot.
    « upper »          : BRISBOIS, DE RONGE.
    « asis »           : la forme exacte de noms_propres.txt.

    Les particules restent capitalisées : distinguer « de Ronge » de « De Ronge »
    demanderait de connaître l'usage propre à chaque patronyme.
    """
    if style == "asis":
        return name
    if style == "upper":
        return name.upper()

    # Title : capitaliser chaque mot, y compris après un tiret ou une apostrophe
    def cap(match):
        return match.group(0)[0].upper() + match.group(0)[1:].lower()

    return re.sub(r"[^\s\-']+", cap, name)


def restore_proper_nouns(text: str, names, style: str = "title") -> str:
    """Rétablit la casse des noms de l'annuaire trouvés dans le texte."""
    for name in names:
        if name.lower() in _TITLES:
            continue
        # \b...\b sur la séquence complète : gère les noms composés d'un bloc
        pattern = re.compile(
            r"\b" + r"\s+".join(re.escape(w) for w in name.split()) + r"\b",
            re.IGNORECASE,
        )
        text = pattern.sub(format_name(name, style), text)
    return text


def capitalize_sentences(text: str) -> str:
    """Met une majuscule au premier mot et après chaque ponctuation forte."""
    if not text:
        return text

    def upper_first(match):
        return match.group(0).upper()

    # Début du texte
    text = re.sub(r"^\s*([a-zà-ÿ])", upper_first, text)
    # Après . ! ? suivis d'un espace
    text = re.sub(r"([.!?]\s+)([a-zà-ÿ])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def add_final_punctuation(text: str, mark: str = ".") -> str:
    """Ajoute une ponctuation finale si le texte n'en a pas."""
    text = text.rstrip()
    if text and text[-1] not in ".!?…:;,":
        text += mark
    return text


def restore(text: str, names=None, proper_nouns=True,
            sentences=True, punctuation=True, style="title") -> str:
    """Applique la remise en forme complète.

    Args:
        text: transcription brute.
        names: annuaire ; chargé depuis noms_propres.txt si omis.
        proper_nouns: rétablir la casse des noms de l'annuaire.
        sentences: majuscule en début de phrase.
        punctuation: point final si absent.
        style: « title » (Brisbois), « upper » (BRISBOIS) ou « asis ».
    """
    if not text or not text.strip():
        return text

    if proper_nouns:
        if names is None:
            names = load_names()
        if names:
            text = restore_proper_nouns(text, names, style)

    if sentences:
        text = capitalize_sentences(text)

    if punctuation:
        text = add_final_punctuation(text)

    return text
