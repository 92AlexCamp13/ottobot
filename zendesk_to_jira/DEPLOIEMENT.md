# Déploiement & sécurité — outil Zendesk → Jira

Document destiné à la **mise en production** (Mac mini interne VOD Factory) et à
l'**audit de sécurité**. Il décrit l'architecture, les secrets, le chiffrement,
l'authentification et la surface réseau.

---

## 1. Architecture et flux de données

```
  Navigateur (3 agents)            Mac mini interne (serveur)            Services externes
 ┌──────────────────┐   HTTPS    ┌──────────────────────────┐   HTTPS   ┌──────────────────┐
 │ pages HTML +      │ ◀───────▶ │ Tailscale serve (443)     │          │ Zendesk API       │
 │ formulaires       │ (tailnet) │   └─▶ uvicorn 127.0.0.1   │ ───────▶ │ Jira API          │
 │ (aucun secret)    │           │        FastAPI (web.py)   │          │ Anthropic API     │
 └──────────────────┘           │        moteur (noyau.py)  │          └──────────────────┘
                                 │        comptes.db + .env  │
                                 └──────────────────────────┘
```

- Le **navigateur ne reçoit que du HTML** : aucune clé API, aucun token, aucun
  secret n'est jamais envoyé au client (ni dans le HTML, ni en URL, ni en JS).
- Les appels aux **API externes** (Zendesk, Jira, Anthropic) partent **du serveur
  uniquement**, sur les connexions HTTPS de ces fournisseurs.

## 2. Hébergement (Mac mini)

- **Exécution permanente** : service `launchd`
  (`com.vodfactory.zendesk-jira.plist`) → `RunAtLoad` (démarrage au boot) +
  `KeepAlive` (redémarrage si crash). Logs dans `logs/` (gitignoré).
- **Lancement** : `demarrer_serveur.sh` → `uvicorn web:app --host 127.0.0.1`.
  L'app écoute **uniquement en local** ; elle n'est pas liée à `0.0.0.0`.
- **Exposition réseau** : assurée par **Tailscale** (`tailscale serve`), qui
  publie l'app en **HTTPS sur le réseau privé (tailnet)** et proxifie vers
  `127.0.0.1:8000`. Conséquence : **aucun port ouvert** sur le LAN ni sur
  Internet ; le serveur est injoignable hors du tailnet.
- **Accès** : limité aux appareils du tailnet VODF (les 3 agents). Un tiers, même
  sur le réseau de l'entreprise, ne voit pas le service.

### Mise en place (une fois, sur le mini)
1. Installer Python 3, Tailscale, et récupérer le code (sans `.venv` ni `.env`).
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Créer le `.env` (voir §3) avec des secrets **propres à la prod**.
4. `tailscale up` (connecter le mini au tailnet), puis exposer l'app :
   `tailscale serve --bg 8000` (vérifier avec `tailscale serve status`).
5. Installer le service launchd (voir en-tête du fichier `.plist`).
6. Durcissement OS (voir §8).
7. Premier accès → page `/setup` : créer le compte admin, puis les 2 autres via
   `/admin` ; chaque agent renseigne son token Jira dans `/profil`.

## 3. Secrets et configuration

Tous les secrets vivent dans le fichier **`.env`** (jamais committé — voir
`.gitignore`). Aucun secret n'est en dur dans le code.

| Variable | Rôle | Sensibilité |
|---|---|---|
| `ANTHROPIC_API_KEY` | reformulation IA des descriptions | **secret** |
| `ZENDESK_API_TOKEN` | profil Zendesk **partagé** par l'équipe | **secret** |
| `ZENDESK_EMAIL`, `ZENDESK_SUBDOMAIN` | compte Zendesk | sensible |
| `JIRA_*` (EMAIL, API_TOKEN, PROJECT_KEY, IDs) | config Jira **partagée** (lecture meta) | **secret** (token) |
| `APP_SECRET_KEY` | signature des cookies de session | **secret** |
| `TOKEN_ENCRYPTION_KEY` | clé Fernet de chiffrement des tokens Jira en base | **secret critique** |
| `APP_DB_PATH` | chemin de `comptes.db` (optionnel) | non sensible |
| `APP_ENV` | `prod` → cookies `secure` (HTTPS) | non sensible |

- **Génération en prod** (NE PAS réutiliser les valeurs de dev) :
  - `APP_SECRET_KEY` : `python -c "import secrets; print(secrets.token_urlsafe(32))"`
  - `TOKEN_ENCRYPTION_KEY` : `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- **Permissions** : restreindre le `.env` au compte serveur → `chmod 600 .env`.

### La clé Anthropic (point d'attention demandé)
`ANTHROPIC_API_KEY` est lue côté serveur (`noyau.reformuler_description`) et
utilisée pour un appel `requests` direct à `api.anthropic.com`. Elle **n'est
jamais** : envoyée au navigateur, insérée dans une page/URL, ni journalisée.
Elle ne transite que du Mac mini vers Anthropic (HTTPS). Son exposition n'est
donc possible que pour quelqu'un ayant accès au **système de fichiers du mini** —
d'où le durcissement OS (§8).

## 4. Chiffrement et hachage (au repos)

- **Mots de passe** des comptes : hachés avec **bcrypt** (sel + coût), jamais en
  clair. Stockés dans `comptes.db`. Code : `comptes._hacher` / `_verifier_hash`.
- **Tokens API Jira** des utilisateurs : chiffrés avec **Fernet** (AES) via
  `TOKEN_ENCRYPTION_KEY` avant stockage. Code : `comptes.definir_token_jira` /
  `lire_token_jira`. En base, on ne voit qu'un blob chiffré.
- Si `TOKEN_ENCRYPTION_KEY` change, les tokens deviennent illisibles → chaque
  agent doit re-saisir le sien (erreur gérée proprement, pas de crash).

## 5. Authentification et sessions

- Login **identifiant + mot de passe** (page `/login`) ; 1er compte via `/setup`.
- **Sessions** : `SessionMiddleware` (Starlette), cookie **signé** avec
  `APP_SECRET_KEY`. Le cookie ne contient **que** l'`user_id` — jamais de token
  ni de mot de passe. Flags : `httponly`, `samesite=lax`, `secure` si
  `APP_ENV=prod` (HTTPS), expiration **8 h**.
- **Rôles** : `admin` (gère les comptes via `/admin`) et utilisateur. Routes
  protégées par dépendance (`requiert_connexion` / `requiert_admin`).
- **Attribution** : chaque création Jira passe par le **token de l'agent
  connecté** → créateur = l'agent (visible dans son « Tickets que j'ai créés »).
  Zendesk utilise un **profil partagé** (token du `.env`) — choix assumé.

## 6. Données stockées sur le serveur

- `comptes.db` (SQLite) : comptes (identifiant, **hash** bcrypt, rôle), et par
  utilisateur l'email Jira + **token Jira chiffré** + accountId/nom Jira. Fichier
  **gitignoré**, sur le disque local du mini (persistant).
- `conversions.csv` : journal des conversions (id ticket, statut, clé Jira).
  Gitignoré. Pas de secret.
- `logs/` : sortie du serveur. Gitignoré.

## 7. Surface réseau (résumé pour l'audit)

| Aspect | Posture |
|---|---|
| Écoute de l'app | `127.0.0.1:8000` **uniquement** (pas `0.0.0.0`) |
| Exposition | via Tailscale `serve` (HTTPS, tailnet privé) |
| Ports ouverts (LAN/Internet) | **aucun** |
| Accès | appareils du tailnet VODF seulement |
| Transport | HTTPS (cert automatique Tailscale) |
| Secrets vers le client | **jamais** (HTML/URL/JS exempts de secrets) |

## 8. Durcissement du Mac mini (recommandé)

- **FileVault** activé (chiffrement disque → protège `.env` et `comptes.db` au repos).
- Mises à jour macOS automatiques ; verrouillage d'écran ; pas de partage inutile.
- **Veille désactivée** pour le service (Réglages > Batterie/Énergie ;
  éventuellement `caffeinate`), sinon l'app s'interrompt.
- Compte serveur dédié, `chmod 600 .env`, sauvegarde chiffrée de `comptes.db`.
- Tailscale : limiter le partage (ACL) aux 3 agents ; révoquer un appareil perdu.

## 9. Hypothèses de confiance / hors périmètre

- Le **profil Zendesk est partagé** : les écritures Zendesk (champ « Clé Jira
  liée », tag, lien connecteur) sont attribuées à ce compte commun, pas à l'agent.
- Pas de double authentification ; la sécurité d'accès repose sur le tailnet + le
  login applicatif.
- Les comptes Jira des agents ont des **droits limités** (backlogs) → surface
  d'impact réduite en cas de fuite d'un token.

## 10. Procédures sensibles

- **Rotation `TOKEN_ENCRYPTION_KEY`** : invalide les tokens chiffrés → prévenir
  l'équipe, chacun re-saisit son token dans `/profil`.
- **Rotation `APP_SECRET_KEY`** : déconnecte toutes les sessions (re-login).
- **Révoquer un agent** : le supprimer dans `/admin` + retirer son appareil du
  tailnet.
- **Mise à jour du code** : `git pull` (ou copie) puis recharger le service
  launchd (`launchctl unload … && launchctl load …`). `comptes.db` est préservée.
