# Journal des versions — VocaWhisper

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [1.6.0] — 2026-09-05

Version consacrée à la fiabilité et au débit de l'API de transcription, à la
suite de l'incident du 4 septembre 2026.

### Corrigé

**L'API pouvait se figer indéfiniment sur un seul audio.**
Le 4 septembre à 21:47 UTC, une transcription est partie en boucle de
répétition et a occupé le moteur pendant **10 h 15** (36 907 s pour 92 s
d'audio), en gardant le verrou d'inférence. Toutes les requêtes valides
suivantes attendaient sans jamais recevoir de réponse, tandis que `/health`
— qui ne prenait pas ce verrou — continuait de répondre « ok, modèle chargé ».
Le phénomène était récurrent : 14 h 03 le 30 août, 5 h le 3 septembre.

Trois protections indépendantes ont été ajoutées :

- **Garde-fous anti-boucle** : `condition_on_previous_text` désactivé (il
  réinjectait le texte répété et alimentait la boucle), `max_new_tokens`
  borné à 200, pénalité de répétition, blocage des trigrammes répétés, et
  filtrage VAD des silences et musiques d'attente — avec un réglage permissif
  adapté à la voix VoIP 8 kHz, vérifié sans perte de mots.
- **Verrou borné** à 120 s : la requête reçoit un `503` au lieu d'attendre sans
  fin.
- **`/health` teste réellement l'inférence** : statut `stuck` et code `503` dès
  qu'une transcription dépasse 180 s.

**Le débit se dégradait continûment (RTF ×14 en 36 h d'uptime).**
Chaque requête HTTP était traitée dans un thread Werkzeug neuf, or CUDA et
CTranslate2 attachent au thread appelant des ressources (contexte, handles
cuBLAS, workspaces) que la disparition du thread Python ne libère pas. Le
nombre de threads restait pourtant stable, ce qui rendait la fuite invisible
aux métriques système.

Mesures à durée d'audio contrôlée, sur 1 200 requêtes :

| Configuration | Dégradation / 1 000 requêtes |
|---|---|
| Moteur seul, thread unique | −0,2 % |
| Moteur, thread neuf par requête | +21,4 % |
| Service complet via HTTP | +23,7 % |

Toutes les inférences passent désormais par un **thread worker unique**
alimenté par une file bornée à 32 entrées. Une requête expirée est marquée
abandonnée et ignorée par le worker, pour ne pas mobiliser le GPU au profit
d'un client déjà parti.

**Le journal grossissait sans limite** (754 Mo constatés) et chaque message y
était écrit deux fois. Rotation à 50 Mo sur 5 fichiers, écriture unique, et
`stderr` redirigé vers un fichier distinct afin de ne pas bloquer la rotation
sous Windows.

### Modifié

- **Modèle par défaut : `large-v3-turbo`** avec `beam_size=1`, mesuré **4,4×
  plus rapide** que `large-v3` avec `beam_size=5` sur de la parole française au
  profil VoIP, sans perte de qualité. Chargement ramené de 6,3 s à 2,0 s.
- RTF de production : **0,337 → 0,019**. Débit constaté : ~110 fichiers/minute.

### Ajouté

- Moteurs de transcription alternatifs : Moonshine, wav2vec2, FastConformer,
  avec repli automatique sur le moteur local.
- Scripts de fine-tuning (Whisper, Moonshine, wav2vec2) et d'extraction des
  erreurs de reconnaissance.
- Remise en forme de la casse et de la ponctuation (`text_case`), utile aux
  moteurs CTC qui transcrivent en minuscules non ponctuées.
- Transcription de médias par lots (`batch/`) et banc de comparaison des
  moteurs (`utils/bench_engines.py`).

### Sécurité

- `.gitignore` exclut désormais explicitement les exports de données
  hospitalières, les extraits de base de production et les sorties de
  fine-tuning, qui n'ont pas à être versionnés.

### Points d'attention pour les intégrations

Deux comportements de l'API changent et peuvent demander une adaptation côté
client :

- `GET /health` renvoie **503** lorsque le moteur est bloqué (`status: "stuck"`)
  ou en cours de chargement (`status: "loading"`). Il expose aussi
  `inference_busy_s`, `last_success_age_s` et `uptime_s`.
- `POST /transcribe` peut renvoyer **503** après 120 s d'attente, ou lorsque la
  file est saturée. Une nouvelle tentative est appropriée dans ces deux cas.

## [1.0] — antérieur

Première version publiée.
