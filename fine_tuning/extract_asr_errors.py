#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection des erreurs ASR sur les noms de médecins/personnes dans calls3.

Principe (demandé par Frédéric) :
  1. On récupère tous les appels dont la transcription cite un titre
     (docteur / dr / professeur / madame / monsieur / mme).
  2. Pour chaque appel, on extrait le nom reconnu par Whisper et on le compare
     (fuzzy) à l'annuaire réel `vocapack_aa.persons`, restreint à l'organisation
     de l'appel (calls3.destination -> organisation.phone_number_id -> org_id).
  3. Classement :
        - MATCH FRANC  (score >= HIGH)  -> Whisper a bon -> on jette (rien à apprendre).
        - CORRIGEABLE  (MID <= score < HIGH) -> on sait quel médecin était visé
                       -> fichier d'apprentissage (audio + transcription CORRIGÉE).
        - INCERTAIN    (score < MID)    -> on ne peut pas trancher
                       -> fichier pour révision HUMAINE.

L'audio historique n'est plus sur disque : on écrit donc juste le NOM du fichier
audio (tel que référencé en base) + la transcription. Frédéric fournit les sons.

Dépendances : aucune obligatoire.
  - fuzzy : rapidfuzz si installé, sinon repli sur difflib (stdlib).
  - DB    : via le client `mysql` en subprocess (pas de driver Python requis).

Usage :
  python3 extract_asr_errors.py                 # toutes les organisations
  python3 extract_asr_errors.py --dest 6318     # une seule (ex. TIVO)
  python3 extract_asr_errors.py --limit 500     # échantillon de test
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CALLS_DB = "vocabase"          # base contenant calls3
DIR_DB = "vocapack_aa"         # base contenant persons / service_names / organisation

HIGH = 90   # >= HIGH  : Whisper correct -> on jette
MID = 72    # [MID,HIGH): corrigeable automatiquement
            # <  MID    : incertain -> humain

MIN_NAME_LEN = 3   # on ignore les noms de famille trop courts (bruit)

TITLES = ["docteur", "doctor", "professeur", "madame", "monsieur", "mme", "mr", "dr", "pr"]
# regex des titres (ordre = priorité de capture, insensible casse/accents)
_TITLE_RE = re.compile(r"\b(docteur|doctor|professeur|madame|monsieur|mme|mr|dr|pr)\b\.?", re.IGNORECASE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output", "asr_errors")

# --------------------------------------------------------------------------- #
# Fuzzy backend (rapidfuzz sinon difflib)
# --------------------------------------------------------------------------- #
try:
    from rapidfuzz import fuzz as _rf_fuzz
    def ratio(a, b):
        # max(ratio, token_sort_ratio) : robuste aux noms composés / inversés
        return max(_rf_fuzz.ratio(a, b), _rf_fuzz.token_sort_ratio(a, b))
    FUZZY_BACKEND = "rapidfuzz"
except ImportError:  # repli stdlib
    from difflib import SequenceMatcher
    def ratio(a, b):
        return SequenceMatcher(None, a, b).ratio() * 100.0
    FUZZY_BACKEND = "difflib"


# --------------------------------------------------------------------------- #
# Accès base via client mysql
# --------------------------------------------------------------------------- #
def run_mysql(db, sql):
    """Exécute une requête et renvoie une liste de lignes (listes de champs)."""
    proc = subprocess.run(
        ["mysql", db, "-N", "-B", "--default-character-set=utf8mb4", "-e", sql],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("Requête MySQL en échec")
    rows = []
    for line in proc.stdout.splitlines():
        rows.append(line.split("\t"))
    return rows


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def norm(s):
    """minuscule, sans accent, ponctuation -> espace, espaces compactés."""
    s = strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# Chargement annuaire
# --------------------------------------------------------------------------- #
def load_dest_to_org():
    """destination (calls3) -> org_id, via organisation.phone_number_id (;-séparé)."""
    rows = run_mysql(DIR_DB,
        "SELECT organisation_id, organisation_code, phone_number_id FROM organisation;")
    dest2org = {}
    org_code = {}
    for org_id, code, phones in rows:
        org_code[org_id] = code
        for token in (phones or "").split(";"):
            token = token.strip()
            if token:
                dest2org[token] = org_id
    return dest2org, org_code

def load_firstnames():
    """Ensemble des prénoms connus (normalisés) -> garde-fou anti faux positifs."""
    rows = run_mysql(DIR_DB,
        "SELECT DISTINCT FirstName FROM persons "
        "WHERE FirstName IS NOT NULL AND FirstName <> '';")
    fn = set()
    for (first,) in rows:
        for tok in norm(first).split():
            if len(tok) >= 3:
                fn.add(tok)
    return fn

def is_firstname(candidate, firstnames):
    """True si le 1er mot du candidat est (fuzzy) un prénom connu."""
    w = norm(candidate).split()
    if not w:
        return False
    head = w[0]
    if head in firstnames:
        return True
    return any(ratio(head, f) >= 90 for f in firstnames)

def load_persons_by_org():
    """org_id -> liste de (canonical_lastname, norm_key)."""
    rows = run_mysql(DIR_DB,
        "SELECT org_id, LastName FROM persons "
        "WHERE (Deleted=0 OR Deleted IS NULL) AND LastName IS NOT NULL AND LastName <> '';")
    by_org = defaultdict(list)
    for org_id, last in rows:
        last = last.strip()
        if len(strip_accents(last)) < MIN_NAME_LEN:
            continue
        by_org[org_id].append((last, norm(last)))
    # dédoublonnage par org
    for org_id in by_org:
        seen = {}
        for canon, key in by_org[org_id]:
            seen.setdefault(key, canon)
        by_org[org_id] = [(c, k) for k, c in seen.items()]
    return by_org


# --------------------------------------------------------------------------- #
# Extraction du nom candidat depuis une transcription
# --------------------------------------------------------------------------- #
def split_hypotheses(transcription):
    """'<br>'-séparé ; retire le score '(NN%)'. Renvoie [(texte, conf|None), ...]."""
    out = []
    for part in transcription.split("<br>"):
        part = part.strip()
        if not part:
            continue
        m = re.search(r"\((\d+)%\)\s*$", part)
        conf = int(m.group(1)) if m else None
        text = re.sub(r"\s*\(\d+%\)\s*$", "", part).strip()
        if text:
            out.append((text, conf))
    return out

def candidate_names(text):
    """Après chaque titre, renvoie les n-grams (1..3 mots) candidats au nom.

    Renvoie une liste de (candidate_string, title_matched).
    """
    cands = []
    for m in _TITLE_RE.finditer(text):
        after = text[m.end():]
        # mots suivants (on s'arrête à un autre titre ou fin)
        words = re.findall(r"[A-Za-zÀ-ÿ'\-]+", after)
        # coupe si on retombe sur un titre
        clean = []
        for w in words[:4]:
            if norm(w) in [norm(t) for t in TITLES]:
                break
            clean.append(w)
        for n in (3, 2, 1):
            if len(clean) >= n:
                cands.append((" ".join(clean[:n]), m.group(1)))
    return cands


def best_person_match(cands, persons):
    """Meilleur (score, canonical, candidate) d'un candidat vs annuaire org."""
    best = (0.0, None, None)
    for cand, _title in cands:
        ck = norm(cand)
        if len(ck) < MIN_NAME_LEN:
            continue
        for canon, key in persons:
            sc = ratio(ck, key)
            if sc > best[0]:
                best = (sc, canon, cand)
    return best


def correct_name_in_full(full_text, canonical, low=55):
    """Dans le texte complet de l'appel, remplace le nom mal reconnu par le nom
    canonique, en s'ancrant après chaque titre (Docteur/Madame/...).

    Renvoie (texte_corrigé, a_remplacé).
    """
    canon_norm = norm(canonical)
    tokens = list(re.finditer(r"[A-Za-zÀ-ÿ'\-]+", full_text))
    title_norms = {norm(t) for t in TITLES}
    spans = []  # (start, end) à remplacer par canonical

    for i, m in enumerate(tokens):
        if norm(m.group()) not in title_norms:
            continue
        best = (0.0, None)  # (score, (start,end))
        for n in (3, 2, 1):
            if i + n < len(tokens) + 1 and i + 1 < len(tokens):
                grp = tokens[i + 1 : i + 1 + n]
                if not grp:
                    continue
                cand = " ".join(g.group() for g in grp)
                sc = ratio(norm(cand), canon_norm)
                if sc > best[0]:
                    best = (sc, (grp[0].start(), grp[-1].end()))
        if best[1] and best[0] >= low:
            spans.append(best[1])

    if not spans:
        return full_text, False
    # appliquer de droite à gauche pour préserver les offsets
    result = full_text
    for start, end in sorted(set(spans), reverse=True):
        result = result[:start] + canonical + result[end:]
    return result, True


def build_corrected(best_hyp_text, candidate, canonical):
    """Remplace le nom mal reconnu par le nom canonique dans l'hypothèse."""
    # remplacement insensible casse/accents du 1er mot du candidat -> canonique
    words = candidate.split()
    if not words:
        return best_hyp_text
    pattern = re.compile(re.escape(words[0]), re.IGNORECASE)
    corrected, n = pattern.subn(canonical, best_hyp_text, count=1)
    if n == 0:
        corrected = best_hyp_text + " -> " + canonical
    # si candidat multi-mots, on retire les mots restants du nom erroné
    for w in words[1:]:
        corrected = re.sub(r"\s*\b" + re.escape(w) + r"\b", "", corrected, count=1, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", corrected).strip()


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", help="filtrer sur une destination (ex. 6318)")
    ap.add_argument("--limit", type=int, help="limiter le nombre d'appels (test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fuzzy backend :", FUZZY_BACKEND)

    dest2org, org_code = load_dest_to_org()
    persons_by_org = load_persons_by_org()
    firstnames = load_firstnames()
    all_persons = [p for lst in persons_by_org.values() for p in lst]
    print("Organisations :", len(org_code),
          "| personnes indexées :", sum(len(v) for v in persons_by_org.values()))

    where = ("(transcription LIKE '%docteur%' OR transcription LIKE '%doctor%' "
             "OR transcription LIKE '%professeur%' OR transcription LIKE '%madame%' "
             "OR transcription LIKE '%monsieur%' OR transcription LIKE '% mme %' "
             "OR transcription LIKE '% dr %') "
             "AND transcription NOT IN ('[SILENCE]','')")
    if args.dest:
        where += " AND TRIM(destination) = '%s'" % args.dest.replace("'", "")
    limit = " LIMIT %d" % args.limit if args.limit else ""

    sql = (
        "SELECT call_id, date, TRIM(destination), "
        "REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(audio,'monitor/',-1),\"'\",1),'',''), "
        "transcription, COALESCE(complete_transcription,'') "
        "FROM calls3 WHERE %s ORDER BY date DESC%s;" % (where, limit)
    )
    rows = run_mysql(CALLS_DB, sql)
    print("Appels à analyser :", len(rows))

    p_auto = os.path.join(OUT_DIR, "train_auto.csv")
    p_human = os.path.join(OUT_DIR, "review_human.csv")
    p_corr = os.path.join(OUT_DIR, "corrections_suggested.txt")

    stats = Counter()
    corr_pairs = Counter()  # (erreur_norm, canonique) -> count

    f_auto = open(p_auto, "w", newline="", encoding="utf-8")
    f_human = open(p_human, "w", newline="", encoding="utf-8")
    w_auto = csv.writer(f_auto)
    w_human = csv.writer(f_human)
    # même format que fine_tuning/data/transcriptions.csv (directement entraînable)
    w_auto.writerow(["audio_file", "transcription"])
    w_human.writerow(["audio_file", "call_id", "date", "org", "raw_transcription",
                      "hypothese", "nom_reconnu", "suggestion", "score"])

    for r in rows:
        if len(r) < 6:
            continue
        call_id, date, dest, audio_file, transcription = r[0], r[1], r[2], r[3], r[4]
        # mysql -B échappe les sauts de ligne/tab : on les restaure en espace
        complete = r[5].replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\").strip()
        org_id = dest2org.get(dest)
        persons = persons_by_org.get(org_id) or all_persons  # repli : tout l'annuaire
        code = org_code.get(org_id, dest)

        hyps = split_hypotheses(transcription)
        if not hyps:
            stats["sans_hypothese"] += 1
            continue

        # meilleur match toutes hypothèses confondues
        best = (0.0, None, None, None)  # score, canonical, candidate, hyp_text
        for hyp_text, _conf in hyps:
            cands = candidate_names(hyp_text)
            if not cands:
                continue
            sc, canon, cand = best_person_match(cands, persons)
            if sc > best[0]:
                best = (sc, canon, cand, hyp_text)

        score, canonical, candidate, hyp_text = best

        if canonical is None:
            # aucun nom exploitable après le titre -> humain
            stats["humain"] += 1
            w_human.writerow([audio_file, call_id, date, code, transcription,
                              hyps[0][0], "", "", ""])
            continue

        if score >= HIGH:
            stats["jete_whisper_ok"] += 1
            continue

        if score >= MID and is_firstname(candidate, firstnames):
            # le mot reconnu est d'abord un prénom : trop risqué en auto -> humain
            stats["humain_prenom"] += 1
            w_human.writerow([audio_file, call_id, date, code, transcription,
                              hyp_text, candidate, canonical or "", round(score, 1)])
        elif score >= MID:
            # cible = transcription complète de l'appel, nom corrigé dedans
            usable_full = complete and complete not in ("[SILENCE]",)
            if usable_full:
                corrected, done = correct_name_in_full(complete, canonical)
                if not done:
                    # le nom n'apparaît pas (assez proche) dans le texte complet :
                    # on garde le texte complet tel quel (nom déjà correct ou absent)
                    corrected = complete
            else:
                corrected = build_corrected(hyp_text, candidate, canonical)
            stats["train_auto"] += 1
            w_auto.writerow([audio_file, corrected])
            corr_pairs[(norm(candidate), canonical)] += 1
        else:
            stats["humain"] += 1
            w_human.writerow([audio_file, call_id, date, code, transcription,
                              hyp_text, candidate, canonical or "", round(score, 1)])

    f_auto.close()
    f_human.close()

    # fichier de corrections suggéré (format corrections.txt : erreur -> correction)
    with open(p_corr, "w", encoding="utf-8") as f:
        f.write("# Corrections de noms suggérées (erreur -> correction) [freq]\n")
        f.write("# Généré par extract_asr_errors.py — À RELIRE avant usage.\n\n")
        for (err, canon), cnt in corr_pairs.most_common():
            f.write("%s -> %s   # x%d\n" % (err, canon, cnt))

    print("\n=== Résultats ===")
    for k in ("jete_whisper_ok", "train_auto", "humain", "humain_prenom", "sans_hypothese"):
        print("  %-18s %d" % (k, stats[k]))
    print("\nÉcrits :")
    print("  ", p_auto, "(apprentissage auto : audio_file, transcription corrigée)")
    print("  ", p_human, "(révision humaine)")
    print("  ", p_corr, "(suggestions pour corrections.txt)")


if __name__ == "__main__":
    main()
