# Outil Zendesk → Jira

Petit outil en ligne de commande pour convertir des tickets de support **Zendesk**
en tickets **Jira** (projet `BUGOTTO`), avec une **validation humaine** avant chaque
création. Conçu pour tourner sur ta machine.

Principe : tu **tagges** les tickets à convertir avec `to-jira` dans Zendesk, tu
lances l'outil, tu **valides** chaque ticket à l'écran, et l'outil crée les Jira
(avec pièces jointes) puis **boucle** côté Zendesk (clé Jira inscrite + tag retiré).

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
pip install -r requirements.txt # installe requests + python-dotenv
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
avec les scripts utilitaires (voir §7). Elles sont normalement déjà remplies.

**Vérifie que tout est bon :**

```bash
python check_setup.py
```

Tu dois voir deux `[OK]` (Zendesk + Jira) et le projet `BUGOTTO` accessible.

---

## 4. Usage quotidien (le cas normal : par lot)

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

## 5. Usage en mode test (un seul ticket par son ID)

Pour traiter un ticket précis sans passer par les tags :

```bash
python create_jira.py 4927      # 4927 = l'ID du ticket Zendesk
```

Même déroulé interactif que ci-dessus, mais sur ce seul ticket.

## 6. Comprendre les traces

Chaque action est enregistrée dans **`conversions.csv`** (ouvrable dans Excel) :
horodatage, ticket Zendesk, statut (`cree`, `deja_converti`, `abandonne`,
`echec_creation`, `erreur`), clé Jira, détail.

**Anti-doublon** : un ticket déjà converti porte une « Clé Jira liée » non vide ;
l'outil le détecte et le **saute** systématiquement. Impossible de créer deux fois
le même Jira.

---

## 7. Scripts utilitaires (rarement nécessaires)

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

**Cœur du programme** : `noyau.py` contient toute la logique partagée ;
`create_jira.py` (`traiter_un_ticket`) et `run_batch.py` s'appuient dessus.

## 8. Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| `401 Unauthorized` | email ou token API incorrect dans le `.env` |
| `406` au téléchargement d'une PJ | déjà géré (en-tête `Accept: */*`) — si ça revient, signale-le |
| `0 ticket tagué` alors que tu as taggué | délai d'indexation Zendesk : attends 1–2 min |
| `Échec de création` (HTTP 400) | l'outil affiche la réponse Jira : elle indique le champ fautif |

## 9. Hors périmètre (v1)

- Pas d'interface graphique (CLI uniquement).
- Pas de synchronisation continue (on lance l'outil à la demande).
- Pas de mise à jour des statuts Jira → Zendesk après coup.
- Les pièces jointes sont transférées ; les images *inline* du corps d'email le
  sont si Zendesk les expose comme pièces jointes.
