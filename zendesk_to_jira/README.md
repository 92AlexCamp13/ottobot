# Outil Zendesk → Jira

Petit outil en ligne de commande pour convertir des tickets de support **Zendesk**
en tickets **Jira** (projet `BUGOTTO`), avec une **validation humaine** avant chaque
création. Conçu pour tourner sur ta machine.

Principe : tu **tagges** les tickets à convertir avec `to-jira` dans Zendesk, tu
lances l'outil, tu **valides** chaque ticket, et l'outil crée les Jira (avec
pièces jointes) puis **boucle** côté Zendesk (clé Jira inscrite + tag retiré).

Deux façons de l'utiliser, qui partagent le **même moteur** :
- une **interface web locale** dans ton navigateur — la plus simple (voir §4) ;
- la **ligne de commande** dans le terminal (voir §5 et §6).

---

## 1. Prérequis

- **Python 3** installé (vérifie avec `python3 --version`).
- Un compte **Zendesk** (avec un token API) et un compte **Atlassian/Jira**
  (avec un token API perso — pas besoin d'être admin Jira).

## 2. Installation (une seule fois)

Dans un terminal, place-toi dans ce dossier puis :

```bash
python3 -m venv .venv           # cree un environnement isole
source .venv/bin/activate       # l'active (a refaire a chaque nouveau terminal)
pip install -r requirements.txt # requests, python-dotenv + l'interface web (FastAPI, uvicorn, jinja2)
```

> À chaque nouvelle session de terminal, refais juste `source .venv/bin/activate`
> avant de lancer un script.

## 3. Configuration (une seule fois)

```bash
cp .env.example .env            # cree ton fichier de secrets (jamais commite)
```

Ouvre `.env` et renseigne :

| Variable | Où la trouver |
|---|---|
| `ZENDESK_SUBDOMAIN` | déjà rempli : `vodfactoryhelp` |
| `ZENDESK_EMAIL` | ton email Zendesk |
| `ZENDESK_API_TOKEN` | Admin Center → Apps and integrations → Zendesk API → Token access |
| `JIRA_BASE_URL` | déjà rempli : `https://vodfactory.atlassian.net` |
| `JIRA_EMAIL` | ton email Atlassian |
| `JIRA_API_TOKEN` | https://id.atlassian.com/manage-profile/security/api-tokens |
| `JIRA_PROJECT_KEY` | déjà rempli : `BUGOTTO` |

Les 5 lignes d'**IDs techniques** (champs custom, assignés) se découvrent une fois
avec les scripts utilitaires (voir §8). Elles sont normalement déjà remplies.

> La clé `ANTHROPIC_API_KEY` sert à **reformuler la description** des tickets en
> rapport de bug (interface web). Si elle est absente, l'outil fonctionne quand
> même : il garde le texte brut du ticket.

**Vérifie que tout est bon :**

```bash
python check_setup.py
```

Tu dois voir deux `[OK]` (Zendesk + Jira) et le projet `BUGOTTO` accessible.

---

## 4. Interface web locale (le plus simple)

Une petite application qui tourne **sur ta machine** : tu qualifies les tickets
dans ton navigateur, avec des menus déroulants au lieu de questions au clavier.

### Lancer

- **Double-clic** sur **`lancer.command`** (dans ce dossier, depuis le Finder).
  > Le Finder masque souvent l'extension : le fichier apparaît comme **`lancer`**
  > avec une icône de Terminal. Au tout premier lancement, macOS peut bloquer :
  > clic droit → **Ouvrir** → **Ouvrir** (à faire une seule fois).
- **Ou** en terminal : `.venv/bin/python lancer.py`

Une fenêtre de Terminal s'ouvre, le serveur démarre, et ton navigateur s'ouvre
tout seul sur **http://127.0.0.1:8000**.

### Utiliser

1. La page d'accueil liste tes tickets taggés `to-jira` (+ un champ pour ouvrir
   un ticket par son **ID** en mode test).
2. Clique **« Traiter »** sur un ticket → la page affiche le ticket et un
   **formulaire** : la description est **reformulée en rapport de bug par l'IA**
   (relis-la, corrige-la si besoin), puis choisis système / onglet, BO-FO,
   priorité, assigné, plateforme.
3. Valide → le Jira est créé (avec pièces jointes + écriture côté Zendesk) et la
   page de résultat affiche la **clé Jira cliquable**.

### Arrêter

Ferme la fenêtre du Terminal, ou fais **Ctrl+C** dedans.

> 🔒 Le serveur n'écoute que sur `127.0.0.1` (ta machine) : il est **inaccessible
> depuis le réseau**. Les clés API ne quittent jamais le serveur. Port modifiable
> via `WEB_PORT` dans le `.env` (8000 par défaut).

---

## 5. Usage en ligne de commande — par lot

1. Dans **Zendesk**, ajoute le tag **`to-jira`** sur les tickets à convertir.
   > ⚠️ La recherche Zendesk a un délai d'indexation : attends **1–2 min** après
   > avoir taggué avant de lancer l'outil.
2. Lance le traitement par lot :
   ```bash
   python run_batch.py
   ```
3. Pour **chaque** ticket, l'outil te pose quelques questions (système Android/iOS
   ou onglet BO + BO/FO, priorité, confirmation de l'assigné et de la plateforme),
   affiche un **récapitulatif + la description**, puis demande confirmation.
   - `Oui, créer maintenant` → le Jira est créé, les pièces jointes transférées,
     et le ticket Zendesk mis à jour (clé inscrite + tag `to-jira` retiré).
   - `Non, passer ce ticket` → rien n'est créé, on passe au suivant.
4. Un **bilan** final liste les créés et les sautés.

## 6. Usage en ligne de commande — un seul ticket (par son ID)

Pour traiter un ticket précis sans passer par les tags :

```bash
python create_jira.py 4927      # 4927 = l'ID du ticket Zendesk
```

Même déroulé interactif que ci-dessus, mais sur ce seul ticket.

## 7. Comprendre les traces

Chaque action est enregistrée dans **`conversions.csv`** (ouvrable dans Excel) :
horodatage, ticket Zendesk, statut, clé Jira, détail. Statuts possibles :

| Statut | Sens |
|---|---|
| `cree` | Jira créé + clé inscrite côté Zendesk + tag retiré (tout est bon) |
| `cree_writeback_ko` | ⚠️ **Jira créé mais la clé n'a PAS pu être inscrite côté Zendesk** : à corriger à la main (voir ci-dessous), sinon un re-run recréerait un doublon |
| `deja_converti` | ticket sauté car déjà converti (champ « Clé Jira liée » rempli) |
| `abandonne` | tu as choisi de ne pas créer (CLI) |
| `echec_creation` | la création Jira a échoué (rien n'a été créé — re-run sans risque) |
| `erreur` | erreur inattendue sur ce ticket (le lot continue) |

**Anti-doublon** : un ticket déjà converti porte une « Clé Jira liée » non vide ;
l'outil le détecte (via **ce champ Zendesk uniquement**, pas via le CSV) et le
**saute**. Le champ est écrit **avant** le transfert des pièces jointes, donc une
panne en cours de route ne crée jamais de doublon — **sauf** si l'écriture du champ
elle-même échoue (statut `cree_writeback_ko`).

> **Corriger un `cree_writeback_ko`** : ouvre le ticket Zendesk, inscris la clé Jira
> dans le champ « Clé Jira liée » et retire le tag `to-jira`. (Grep le CSV pour les
> repérer : `grep cree_writeback_ko conversions.csv`.)

---

## 8. Scripts utilitaires (rarement nécessaires)

Ces scripts servent à la mise en place ou au diagnostic, pas à l'usage courant :

| Script | Rôle |
|---|---|
| `check_setup.py` | vérifie que les auth Zendesk + Jira fonctionnent |
| `discover_ids.py` | liste les champs/comptes pour remplir les IDs du `.env` |
| `create_jira_key_field.py` | crée (une fois) le champ Zendesk « Clé Jira liée » |
| `inspect_jira_meta.py` | montre ce que Jira attend (types, priorités, champs) |
| `fetch_ticket.py <id>` | affiche un ticket Zendesk (lecture seule) |
| `transform_ticket.py <id>` | montre le brouillon Jira en mémoire (aucune écriture) |
| `review_ticket.py <id>` | revue interactive sans création (aucune écriture) |
| `transfer_attachments.py <cle> <id>` | (re)transfère les PJ d'un ticket vers une issue |
| `writeback_zendesk.py <id> <cle>` | réécrit la clé + retire le tag (test étape 7) |

**Cœur du programme** : `noyau.py` contient toute la logique partagée (le
« moteur ») ; `create_jira.py` (`traiter_un_ticket`, `creer_depuis_fields`) et
`run_batch.py` l'utilisent pour le CLI. L'interface web vit dans `web.py`
(routes FastAPI) + `templates/` (pages HTML), lancée par `lancer.py` /
`lancer.command` — elle appelle le **même moteur**, sans logique métier propre.

## 9. Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| `401 Unauthorized` | email ou token API incorrect dans le `.env` |
| `406` au téléchargement d'une PJ | déjà géré (en-tête `Accept: */*`) — si ça revient, signale-le |
| `0 ticket tagué` alors que tu as taggué | délai d'indexation Zendesk : attends 1–2 min |
| `Échec de création` (HTTP 400) | l'outil affiche la réponse Jira : elle indique le champ fautif |
| (web) `address already in use` au lancement | le port 8000 est déjà pris : ferme l'autre serveur, ou change `WEB_PORT` dans le `.env` |
| (web) description en texte brut + « Reformulation indisponible » | `ANTHROPIC_API_KEY` absente/invalide ou API Claude injoignable — non bloquant, le champ reste éditable |
| (web) page « Une erreur est survenue » | le message affiché donne la cause (ticket introuvable, config manquante…) ; le serveur, lui, reste actif |
| Erreur transitoire (5xx, timeout, coupure) | **re-essayée 3 fois automatiquement** sur les appels idempotents (GET / écriture du champ / retrait du tag). La **création Jira (POST) n'est PAS re-essayée** — pour ne jamais créer deux issues si la 1re a abouti sans réponse |
| Statut `cree_writeback_ko` dans le CSV | Jira créé mais clé non inscrite côté Zendesk → corrige à la main (voir §7), sinon doublon au prochain run |

## 10. Hors périmètre

- L'interface web tourne **uniquement en local** (localhost) : pas d'hébergement,
  pas d'accès réseau, pas de multi-utilisateur, pas d'authentification.
- Pas de synchronisation continue (on lance l'outil à la demande).
- Pas de mise à jour des statuts Jira → Zendesk après coup.
- Les pièces jointes sont transférées ; les images *inline* du corps d'email le
  sont si Zendesk les expose comme pièces jointes.
