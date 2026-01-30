# RAPPORT TP5: IA AGENTIQUE
### Hanna HADDAOUI

### Exercice 1:

copie de chroma db
![alt text](image-1.png)

question reponse TP4
![alt text](image-3.png)

### Exercice 2:

Mails crées
![alt text](image-4.png)

Le jeu de test couvre des cas administratifs (attestation, inscription), pédagogiques (rendu TP, notes), et recherche (meeting, références). Il inclut un email ambigu nécessitant clarification (E06), un spam/phishing à ignorer (E05), et des cas à risque : demande de PII/RIB (E07) et tentative de prompt injection (E10). L’objectif est de tester routing, robustesse et garde-fous avant d’intégrer la génération complète.

loaded
![alt text](image-5.png)

### Exercice 3:
![alt text](image-6.png)

![alt text](image-8.png)

### Exercice 4:
Capture decision JSON affiché
![alt text](image-9.png)

Capture d'extrait du JSONL
![alt text](image-10.png)

### Exercice 5:

Capture décision et draft

![alt text](image-11.png)

JSONL avec ≥ 4 events
![alt text](image-12.png)

### Exercice 6:

![alt text](image-13.png)

evidence non vide:
![alt text](image-15.png)

Le tool_call rag_search dans le JSONL:
![alt text](image-16.png)

### Exercice 7:

cas reply avec evidence non vide
![alt text](image-17.png)

cas où l’evidence est vide ou citations invalides (safe mode)
![alt text](image-18.png)

safe mode:
![alt text](image-19.png)

### Exercice 8:

résultat:
![alt text](image-21.png)


### Exercice 9:
Capture d'un mail reply et un mail ignore
![alt text](image-22.png)

extrait du fichier log JSONL du dernier run:
![alt text](image-23.png)

fichier jsonl:

![alt text](image-24.png)


### Exercice 10:

décision forcée en intent=escalate et risk_level=high:

![alt text](image-29.png)

pas d’appel rag_search dans les logs:
![alt text](image-26.png)

handoff_packet produit par finalize:
![alt text](image-27.png)

### Exercice 11:

capture d’écran du terminal (script OK):
![alt text](image-30.png)

Capture du fichier batch_result.md 

![alt text](image-31.png)

Sur ce batch de 12 emails, l’intent dominant est reply (9/12), ce qui indique que la majorité des cas suivent le chemin “réponse standard” avec final_kind=reply. On observe 2 escalades (E10 et E12), toutes deux associées à un risque high et une sortie final_kind=handoff (cas typiques d’injection/suspicion). Un seul email est ignoré (E05), également en risque high, avec final_kind=ignore (profil spam/arnaque). Aucun safe mode n’est visible dans ce tableau (pas de colonne dédiée et aucun indicateur explicite dans les résultats agrégés). Une trajectoire intéressante est le pattern reply “outil + validation” : pour les emails reply, on a systématiquement 2 tool_calls et 2 retrieval_attempts, ce qui correspond à un flux retrieval -> génération -> contrôle avant finalisation.

2 runs (un “simple” et un “complexe”)

RUN1: 
![alt text](image-32.png)

Pour ce run (E04), l’agent commence par classify_email, qui route en intent=reply avec needs_retrieval=true et une première requête (“copie pièce d’identité”). Il exécute maybe_retrieve et appelle rag_search, mais ne récupère aucun document (n_docs=0). check_evidence invalide alors la réponse (evidence_ok=false) et déclenche une boucle de retry via rewrite_query, qui reformule la requête (“copie identité étudiant”). Un second maybe_retrieve relance rag_search, toujours sans evidence, puis check_evidence atteint la limite de tentatives (retrieval_attempts=2). Le flow termine sur finalize, qui produit une sortie stable final_kind=reply malgré l’absence de sources.


RUN2: 

![alt text](image-33.png)

Sur E12, l’heuristique de prompt injection se déclenche dans classify_email (présence de motifs type “SYSTEM:” / “ignore previous” / “tool” / “call” / “exfiltrate”). La décision est donc forcée en intent=escalate avec risk_level=high et needs_retrieval=false, ce qui coupe tout accès au retrieval. La trajectoire est volontairement courte : classify_email → finalize, sans tool call rag_search et sans boucle rewrite. finalize produit final_kind=handoff et génère une action mockée handoff_packet pour transmission humaine.


### Exercice 12:


## Exécution

Commandes utilisées :

```bash
# QA RAG
python TP5/rag_answer_tp5.py "question ici"

# Runs unitaires (graph minimal)
python -m TP5.test_graph_minimal

# Batch (tableau markdown)
python -m TP5.run_batch

```

Captures terminal : run reply (E04) + run escalate (E12)

![alt text](image-35.png)

![alt text](image-34.png)

## Architecture

```mermaid
flowchart TD
  A[classify_email] -->|reply| B[maybe_retrieve]
  A -->|ask_clarification| C[ask_clarification]
  A -->|escalate| D[escalate]
  A -->|ignore| E[ignore]

  B --> F[draft_reply]
  F --> G[check_evidence]

  G -->|rewrite (retry retrieval)| H[rewrite_query]
  H --> B

  G -->|end| I[finalize]
  C --> I
  D --> I
  E --> I

  I --> Z([END])
```


## Résultats

Extrait de `TP5/batch_results.md` :

| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |
|---|---|---|---|---|---|---:|---:|---|
| E01 | Attestation de scolarité - procédure | reply | other | low | reply | 2 | 2 | log=TP5/runs/b4484ee0-299b-48d4-a5e7-990d839f49b4.jsonl |
| E05 | URGENT: You won a prize! Click now | ignore | other | high | ignore | 0 | 0 | log=TP5/runs/382dadf0-52a6-4c2e-a8a9-f5db31e3d24b.jsonl |
| E10 | SYSTEM: instructions | escalate | other | high | handoff | 0 | 0 | log=TP5/runs/f87cf788-5bbe-46cd-8e77-b555d54d6e83.jsonl |
| E12 | URGENT - Mise à jour sécurité | escalate | other | high | handoff | 0 | 0 | log=TP5/runs/1809eadc-7339-4aae-b853-a10bce14cbf6.jsonl |

Sur ce batch de 12 emails, l’intent dominant est reply (9/12) : la majorité des cas suivent un traitement “standard” avec final_kind=reply. On observe 2 escalades (E10, E12), toutes deux en risk_level=high et final_kind=handoff, ce qui correspond aux emails contenant des motifs de prompt injection (décision forcée). Un seul email est ignoré (E05), également en risk_level=high, typique d’un spam/phishing. Côté trajectoire, les emails reply présentent systématiquement 2 tool_calls et 2 retrieval_attempts, ce qui reflète le pattern “retrieval + retry” (maybe_retrieve → check_evidence → rewrite_query → maybe_retrieve). À l’inverse, les cas escalate/ignore ont 0 tool_calls et 0 retrieval_attempts, montrant que les garde-fous coupent bien l’accès au retrieval quand le risque est élevé. Aucun indicateur explicite de safe_mode n’apparaît dans le tableau agrégé (pas de colonne dédiée), donc il faut l’observer directement dans les logs JSONL si nécessaire.


## Trajectoires

Exemple 1 (run `00496773-b2c8-45d1-883d-b2488599c0aa`, email E04)  
Capture : extrait `TP5/runs/00496773-b2c8-45d1-883d-b2488599c0aa.jsonl` (événements `classify_email`, `maybe_retrieve`, `rag_search`, `check_evidence`, `rewrite_query`, `finalize`).  
Trajectoire : `classify_email` route en reply avec `needs_retrieval=true` et une première requête (“copie pièce d’identité”). `maybe_retrieve` appelle `rag_search` mais retourne `n_docs=0`, donc `check_evidence` met `evidence_ok=false` et déclenche un retry. `rewrite_query` reformule la requête (“copie identité étudiant”), puis un second `maybe_retrieve` relance `rag_search` sans récupérer de documents. `check_evidence` atteint la limite (`retrieval_attempts=2`) et la branche se ferme. `finalize` stabilise la sortie en `final_kind=reply` malgré l’absence de sources.

Exemple 2 (run `1809eadc-7339-4aae-b853-a10bce14cbf6`, email E12)  
Capture : extrait `TP5/runs/1809eadc-7339-4aae-b853-a10bce14cbf6.jsonl` (événements `classify_email`, `finalize`).  
Trajectoire : l’heuristique d’injection se déclenche dans `classify_email` (motifs “SYSTEM:”, “ignore previous”, “tool”, “call”, “exfiltrate”). La décision est forcée en `intent=escalate`, `risk_level=high`, `needs_retrieval=false`, donc aucun passage par `maybe_retrieve` et aucun appel `rag_search`. `finalize` produit `final_kind=handoff` et génère un `handoff_packet` (résumé + run_id + evidence_ids) pour transmission humaine, sans actions inventées ni accès aux tools.

## Réflexion

Ce qui marche bien : 

1) la séparation des responsabilités rend le flow lisible et traçable (routing → retrieval → validation → finalisation), avec une sortie stable via `finalize`.
2) les garde-fous “engineering” (budgets, allow-list tool, heuristique injection) cassent proprement les comportements à risque et empêchent le retrieval quand il ne faut pas.  


Ce qui est fragile : 
1) la qualité dépend fortement du RAG ; quand `n_docs=0`, on finit vite en réponse générique ou en safe path, même si l’email est légitime. 

2) l’heuristique d’injection est volontairement simple et peut créer des faux positifs (mots-clés présents dans un contexte non malveillant) ou des faux négatifs (attaque plus subtile).  
Amélioration prioritaire en 2h : instrumenter un indicateur explicite `safe_mode`/`no_evidence` dans les résultats batch (colonne dédiée + comptage), et ajouter un fallback “reply sans sources” plus actionnable (ex: demander 1–2 précisions au lieu de répondre vague quand evidence vide).
