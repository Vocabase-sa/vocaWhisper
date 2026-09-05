"""Correction des noms propres par fuzzy matching (rapidfuzz).

Charge la liste de noms depuis noms_propres.txt et corrige les mots
de la transcription qui s'en approchent au-dela du seuil configure.

Strategie en 2 passes (identique a stt-api) :
  1) Groupes multi-mots (4, 3, 2) pour les noms composes
  2) Mots individuels restants
"""

import os
import re
import logging
import unicodedata

from rapidfuzz import fuzz


def _strip_accents(text: str) -> str:
    """Retire les accents, pour comparer aux listes ecrites sans diacritiques."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOMS_PROPRES_FILE = os.path.join(BASE_DIR, "noms_propres.txt")

# Mots de titre ignores en debut de groupe multi-mots
_TITLE_WORDS = frozenset({
    "docteur", "doctor", "dr", "monsieur", "madame", "mme", "mr", "professeur", "pr",
})

# Mots jamais remplaces par un nom propre, meme si le score le justifie.
# Sans cette protection, un annuaire un peu fourni ecrase le francais courant :
# "docteur" devient "CRETEUR" (score 86), "voudrais" devient "VOUDRAI", etc.
# Le risque croit avec la taille de l'annuaire, puisque chaque mot est compare
# a TOUS les noms.
_PROTECTED_WORDS = _TITLE_WORDS | frozenset({
    # pronoms et determinants
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "te", "se",
    "le", "la", "les", "un", "une", "des", "du", "au", "aux", "ce", "cet", "cette",
    "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre",
    "leur", "leurs", "qui", "que", "quoi", "dont", "ou",
    # verbes frequents
    "est", "sont", "etre", "suis", "ete", "avoir", "avait", "aurait", "fait",
    "faire", "peut", "peux", "pouvez", "veux", "veut", "voulez", "voudrais",
    "voudrait", "aimerais", "cherche", "cherchez", "appelle", "parler", "passer",
    "avoir", "dire", "voir", "savoir", "mettre", "prendre",
    # mots outils
    "pas", "plus", "moins", "tres", "bien", "mal", "oui", "non", "merci",
    "bonjour", "bonsoir", "revoir", "salut", "avec", "sans", "pour", "par",
    "sur", "sous", "dans", "chez", "vers", "entre", "mais", "donc", "car",
    "puis", "aussi", "encore", "deja", "toujours", "jamais", "peut-etre",
    "sil", "plait", "plait", "silvous",
    # metier
    "service", "consultation", "rendez", "vous", "urgence", "urgences",
    "secretariat", "accueil", "poste", "numero", "telephone", "appel",
})


def load_noms_propres() -> list[str]:
    """Charge la liste de noms propres depuis noms_propres.txt."""
    if not os.path.exists(NOMS_PROPRES_FILE):
        return []
    names = []
    with open(NOMS_PROPRES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def _best_fuzzy(candidate: str, names_list: list[str]) -> tuple[str | None, int]:
    """Retourne (best_match, best_score) pour un candidat contre la liste de noms."""
    best_score = 0
    best_match = None
    candidate_lower = candidate.lower()
    for name in names_list:
        score = fuzz.ratio(candidate_lower, name.lower())
        if score > best_score:
            best_score = score
            best_match = name
    return best_match, best_score


def fuzzy_match_names(text: str, names_list: list[str], threshold: int = 75) -> str:
    """Corrige les noms propres dans le texte par fuzzy matching.

    Args:
        text: texte transcrit a corriger
        names_list: liste de noms propres de reference
        threshold: score minimum (0-100) pour accepter une correction

    Returns:
        texte corrige
    """
    if not names_list or not text:
        return text

    names_lower = {n.lower() for n in names_list}
    words = list(re.finditer(r"\S+", text))
    if not words:
        return text

    replacements = []
    matched_indices: set[int] = set()

    # --- Passe 1 : groupes multi-mots (4, 3, 2) ---
    # Ne matche que contre les noms composes (multi-mots) de la liste
    multi_word_names = [n for n in names_list if " " in n.strip()]

    for group_size in range(4, 1, -1):
        for i in range(len(words) - group_size + 1):
            if any(j in matched_indices for j in range(i, i + group_size)):
                continue

            group_words = words[i : i + group_size]
            start = group_words[0].start()
            end = group_words[-1].end()

            # Exclure le titre en debut de groupe
            first_clean = group_words[0].group().rstrip(".,;:!?").lower().rstrip(".")
            if first_clean in _TITLE_WORDS:
                if group_size == 2:
                    continue
                fuzzy_start = group_words[1].start()
                candidate = text[fuzzy_start:end]
            else:
                fuzzy_start = start
                candidate = text[start:end]

            if len(candidate) < 3:
                continue
            if candidate.lower() in names_lower:
                for j in range(i, i + group_size):
                    matched_indices.add(j)
                continue

            # Ne comparer qu'avec les noms composes pour eviter
            # qu'un groupe "Prenom Nom" ecrase le prenom
            if not multi_word_names:
                continue
            best_match, best_score = _best_fuzzy(candidate, multi_word_names)

            if best_match and best_score >= threshold:
                trailing_word = group_words[-1].group()
                trailing_clean = trailing_word.rstrip(".,;:!?")
                trailing = trailing_word[len(trailing_clean) :]
                replacements.append((fuzzy_start, end, best_match + trailing, best_score))
                for j in range(i, i + group_size):
                    matched_indices.add(j)

    # --- Passe 2 : mots individuels ---
    for idx, match in enumerate(words):
        if idx in matched_indices:
            continue
        word = match.group()
        clean = word.rstrip(".,;:!?")
        if len(clean) < 3:
            continue
        if clean.lower() in names_lower:
            continue
        # Ne jamais transformer un mot courant en nom propre : la passe 1
        # protege deja les titres, la passe 2 doit en faire autant.
        if _strip_accents(clean.lower()) in _PROTECTED_WORDS:
            continue

        best_match, best_score = _best_fuzzy(clean, names_list)

        if best_match and best_score >= threshold:
            trailing = word[len(clean) :]
            replacements.append((match.start(), match.end(), best_match + trailing, best_score))

    if not replacements:
        return text

    result = text
    for start, end, replacement, score in sorted(replacements, key=lambda x: x[0], reverse=True):
        original = result[start:end]
        logger.info(f"[FUZZY] '{original}' -> '{replacement}' (score={score})")
        result = result[:start] + replacement + result[end:]

    return result


def apply_fuzzy_corrections(text: str, threshold: int = 75) -> str:
    """Point d'entree : charge les noms et applique le fuzzy matching."""
    names = load_noms_propres()
    if not names:
        return text
    return fuzzy_match_names(text, names, threshold)
