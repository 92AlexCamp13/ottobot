"""
noyau.py — Le COEUR partage de l'outil Zendesk -> Jira.

Ce module ne s'execute pas seul : il regroupe les briques reutilisees par les
autres scripts (revue, creation, reecriture, traitement par lot). C'est la mise
en pratique du principe d'architecture du brief (§5) : un coeur unique, importe
par les differentes couches.

Contenu :
  - Constantes metier (type Bug, priorites, echeances, tags, exceptions)
  - Connexions HTTP authentifiees (Zendesk, Jira)
  - Lecture d'un ticket Zendesk
  - Detection type/assigne via les tags
  - Match de la plateforme (Option A : nom normalise)
  - Conversion description -> ADF
  - Calcul de l'echeance
"""

import csv
import os
import re
import unicodedata
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# ERREUR METIER (robustesse, brief §6)
# ============================================================================

class ErreurOutil(Exception):
    """Erreur previsible de l'outil (config manquante, ticket introuvable, API KO).

    Le moteur LEVE cette exception au lieu de faire sys.exit : ainsi chaque point
    d'entree la rattrape a sa facon. Le CLI affiche le message et s'arrete ; le
    web l'attrape (gestionnaire FastAPI) et rend une page d'erreur lisible, sans
    tuer le serveur ni montrer de trace brute. C'est le pendant "robustesse" du
    principe 'un seul moteur, plusieurs entrees' (brief §4).
    """


# ============================================================================
# CONSTANTES METIER (decouvertes via inspect_jira_meta.py)
# ============================================================================

ISSUE_TYPE_BUG_ID = "10004"          # type "Bug" dans BUGOTTO

PRIORITES = {                        # nom lisible -> id Jira
    "Mineur": "5",
    "Medium": "10003",
    "Majeur": "10002",
    "Bloquant": "10000",
    "Incident": "4",
}

ECHEANCE_JOURS = {                   # jours a ajouter selon la priorite
    "Mineur": 30, "Medium": 30, "Majeur": 7, "Bloquant": 2, "Incident": 2,
}

TAG_APP_MOBILE = "application_s__mobile"
TAG_APP_TV = "application_s__tv"

# Tag declencheur : pose a la main sur les tickets a convertir, retire apres coup.
TAG_DECLENCHEUR = "to-jira"

# Exceptions de match plateforme : a remplir SEULEMENT pour les cas que
# l'auto-match normalise rate. Cle = valeur Zendesk, valeur = libelle Jira exact.
EXCEPTIONS_PLATEFORME = {
    # "valeur_zendesk": "Libelle Jira exact (NNN)",
}


# ============================================================================
# OUTILS DE BASE
# ============================================================================

def get_required(nom: str) -> str:
    """Lit une variable obligatoire du .env, ou leve ErreurOutil avec un message clair."""
    valeur = os.getenv(nom, "").strip()
    if not valeur:
        raise ErreurOutil(f"Variable manquante dans le .env : {nom}")
    return valeur


def _session_avec_retry() -> requests.Session:
    """Cree une session HTTP qui REESSAIE automatiquement les erreurs transitoires.

    Re-essais (3 max, avec pause croissante 0,5s -> 1s -> 2s) sur :
      - coupures reseau / timeouts ;
      - statuts 429 (trop de requetes) et 5xx (serveur Zendesk/Jira en vrac).

    POINT CRUCIAL : on n'autorise les re-essais QUE pour les methodes idempotentes
    (GET, PUT, DELETE). Le POST est volontairement EXCLU : reessayer la creation
    d'une issue Jira (POST) risquerait d'en creer DEUX si la 1re a abouti mais que
    la reponse s'est perdue. Le PUT du champ 'Cle Jira liee' et le DELETE du tag,
    eux, sont idempotents -> sans risque a rejouer (c'est la qu'on veut le filet).
    """
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "PUT", "DELETE"]),  # PAS de POST
        raise_on_status=False,
    )
    s = requests.Session()
    adaptateur = HTTPAdapter(max_retries=retry)
    s.mount("https://", adaptateur)
    s.mount("http://", adaptateur)
    return s


def session_zendesk() -> tuple[requests.Session, str]:
    sub = get_required("ZENDESK_SUBDOMAIN")
    s = _session_avec_retry()
    s.auth = (f"{get_required('ZENDESK_EMAIL')}/token", get_required("ZENDESK_API_TOKEN"))
    s.headers.update({"Accept": "application/json"})
    return s, f"https://{sub}.zendesk.com/api/v2"


def session_jira() -> tuple[requests.Session, str]:
    base = get_required("JIRA_BASE_URL").rstrip("/")
    s = _session_avec_retry()
    s.auth = (get_required("JIRA_EMAIL"), get_required("JIRA_API_TOKEN"))
    s.headers.update({"Accept": "application/json"})
    return s, base


def session_jira_pour(email: str, token: str) -> tuple[requests.Session, str]:
    """Session Jira authentifiee avec les identifiants d'UN utilisateur donne.

    Meme instance Jira que session_jira (JIRA_BASE_URL, partagee par l'equipe),
    mais email + token specifiques : c'est ce qui permet, en multi-utilisateur,
    que la creation soit attribuee a CET utilisateur (createur = lui dans Jira).
    Le CLI continue d'utiliser session_jira() (identifiants du .env).
    """
    base = get_required("JIRA_BASE_URL").rstrip("/")
    s = _session_avec_retry()
    s.auth = (email, token)
    s.headers.update({"Accept": "application/json"})
    return s, base


def verifier_identite_jira(email: str, token: str) -> dict | None:
    """Valide des identifiants Jira en appelant /myself.

    Renvoie {"accountId", "nom"} si les identifiants sont bons, sinon None
    (token/email invalides, ou Jira injoignable). Defensif : ne leve jamais.
    Sert a valider le token saisi par un utilisateur dans son profil.
    """
    try:
        jira, base = session_jira_pour(email, token)
        rep = jira.get(f"{base}/rest/api/3/myself", timeout=15)
    except requests.RequestException:
        return None
    if rep.status_code != 200:
        return None
    d = rep.json()
    return {"accountId": d.get("accountId"), "nom": d.get("displayName")}


def compte_jira_courant(jira: requests.Session, base: str) -> dict:
    """Renvoie le compte Jira authentifie (toi) : {accountId, nom}.

    C'est ce compte que Jira utilise par defaut comme RAPPORTEUR (demandeur) d'une
    issue creee via l'API, tant qu'on ne precise pas explicitement le champ
    'reporter'. On l'affiche pour que tu voies que le demandeur Jira est bien toi,
    et jamais le demandeur Zendesk d'origine.
    """
    rep = jira.get(f"{base}/rest/api/3/myself", timeout=15)
    if rep.status_code != 200:
        return {"accountId": None, "nom": "(compte courant inconnu)"}
    d = rep.json()
    return {"accountId": d.get("accountId"), "nom": d.get("displayName")}


# ============================================================================
# LECTURE D'UN TICKET ZENDESK
# ============================================================================

def lire_champ_custom(custom_fields: list, field_id: str):
    for champ in custom_fields:
        if str(champ.get("id")) == str(field_id):
            return champ.get("value")
    return None


def charger_ticket(zd: requests.Session, base: str, ticket_id: str) -> dict:
    """Recupere un ticket Zendesk + le nom du demandeur, sous forme de dict simple."""
    rep = zd.get(f"{base}/tickets/{ticket_id}.json", timeout=15)
    if rep.status_code == 404:
        raise ErreurOutil(f"Ticket Zendesk {ticket_id} introuvable (404).")
    if rep.status_code != 200:
        raise ErreurOutil(f"Erreur Zendesk (HTTP {rep.status_code}) sur le ticket {ticket_id}.")
    t = rep.json().get("ticket", {})

    nom, email = "(inconnu)", ""
    if t.get("requester_id"):
        ru = zd.get(f"{base}/users/{t['requester_id']}.json", timeout=15)
        if ru.status_code == 200:
            u = ru.json().get("user", {})
            nom, email = u.get("name", "(inconnu)"), u.get("email", "")

    return {
        "id": t.get("id"),
        "subject": t.get("subject", ""),
        "description": t.get("description", "") or "",
        "tags": t.get("tags", []),
        "created_at": t.get("created_at", ""),
        "demandeur": f"{nom} <{email}>",
        "plateforme_zendesk": lire_champ_custom(t.get("custom_fields", []),
                                                get_required("ZENDESK_CLIENT_FIELD_ID")),
        "cle_jira_existante": lire_champ_custom(t.get("custom_fields", []),
                                                get_required("ZENDESK_JIRA_KEY_FIELD_ID")),
    }


def url_ticket_zendesk(base_zd: str, ticket_id) -> str:
    """Reconstruit l'URL agent d'un ticket a partir de l'URL d'API."""
    return f"{base_zd.replace('/api/v2', '')}/agent/tickets/{ticket_id}"


def lister_tickets_a_traiter(zd: requests.Session, base: str) -> list:
    """Renvoie les tickets tagges 'to-jira' sous forme [{id, subject}].

    Version "web-friendly" de la recherche faite par run_batch.chercher_tickets_a_traiter :
    cette derniere ne renvoie que les IDs (suffisant pour le lot en terminal),
    alors que la page d'accueil web a besoin du SUJET pour etre lisible. Le sujet
    est deja present dans la reponse de recherche Zendesk : aucun appel en plus.
    On suit la pagination 'next_page' pour ne rien manquer.
    """
    tickets = []
    url = f"{base}/search.json"
    params = {"query": f"type:ticket tags:{TAG_DECLENCHEUR}"}
    while url:
        rep = zd.get(url, params=params, timeout=20)
        if rep.status_code != 200:
            break
        data = rep.json()
        for resultat in data.get("results", []):
            if resultat.get("id"):
                tickets.append({"id": resultat["id"],
                                "subject": resultat.get("subject", "(sans sujet)")})
        # 'next_page' est une URL complete (ou None), params deja inclus dedans.
        url, params = data.get("next_page"), None
    return tickets


# ============================================================================
# DETECTION TYPE + ASSIGNE (via les tags)
# ============================================================================

def assignes_possibles() -> list:
    """Les deux destinataires Jira connus, sous forme [{id, label}].

    Source unique pour le MENU "Assigne" du formulaire web (on ne reinvente pas
    les valeurs : ce sont les memes que celles utilisees par le CLI dans
    review_ticket, cf. detecter_type_et_assigne et la bascule a la validation).
    """
    return [
        {"id": get_required("JIRA_ASSIGNEE_TECH_VODF"), "label": "Tech VODF"},
        {"id": get_required("JIRA_ASSIGNEE_SOUFIANE"), "label": "EL AMRANI Soufiane"},
    ]


def detecter_type_et_assigne(tags: list) -> dict:
    """Type de ticket + assigne suggere, deduits des tags (regle verrouillee)."""
    if TAG_APP_MOBILE in tags:
        return {"type": "App Mobile",
                "gabarit_resume": "[App Mobile] {os} - {titre}",
                "assigne_id": get_required("JIRA_ASSIGNEE_SOUFIANE"),
                "assigne_label": "EL AMRANI Soufiane"}
    if TAG_APP_TV in tags:
        return {"type": "App TV",
                "gabarit_resume": "[App TV] {os} - {titre}",
                "assigne_id": get_required("JIRA_ASSIGNEE_SOUFIANE"),
                "assigne_label": "EL AMRANI Soufiane"}
    return {"type": "Web",
            "gabarit_resume": "[{onglet}] {bofo} - {titre}",
            "assigne_id": get_required("JIRA_ASSIGNEE_TECH_VODF"),
            "assigne_label": "Tech VODF"}


# ============================================================================
# MATCH PLATEFORME (Option A : nom normalise)
# ============================================================================

def normaliser(libelle: str) -> str:
    """Forme canonique d'un libelle : sans code (NNN), sans accents, a-z0-9."""
    if not libelle:
        return ""
    sans_code = re.sub(r"\s*\([^)]*\)\s*$", "", libelle)
    sans_accents = unicodedata.normalize("NFKD", sans_code).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", sans_accents.lower())


def charger_options_plateforme(jira: requests.Session, base: str) -> list:
    """Liste des options du champ multi-select 'Plateforme' (via createmeta)."""
    projet = get_required("JIRA_PROJECT_KEY")
    champ = get_required("JIRA_PLATFORM_FIELD_ID")
    url = f"{base}/rest/api/3/issue/createmeta/{projet}/issuetypes/{ISSUE_TYPE_BUG_ID}"
    rep = jira.get(url, timeout=15)
    if rep.status_code != 200:
        raise ErreurOutil(f"Lecture des options plateforme impossible (HTTP {rep.status_code}).")
    data = rep.json()
    champs = data.get("values") or data.get("fields") or []
    if isinstance(champs, dict):
        champs = [{**v, "fieldId": v.get("fieldId", k)} for k, v in champs.items()]
    for c in champs:
        if c.get("fieldId") == champ:
            return c.get("allowedValues", [])
    return []


def matcher_plateforme(valeur_zendesk: str, options: list) -> dict | None:
    """Trouve l'option Jira correspondant a la valeur Zendesk -> {id,label} ou None."""
    if not valeur_zendesk:
        return None
    if valeur_zendesk in EXCEPTIONS_PLATEFORME:
        cible = normaliser(EXCEPTIONS_PLATEFORME[valeur_zendesk])
    else:
        cible = normaliser(valeur_zendesk)
    for opt in options:
        libelle = opt.get("value") or opt.get("name") or ""
        if normaliser(libelle) == cible:
            return {"id": opt.get("id"), "label": libelle}
    return None


# ============================================================================
# DESCRIPTION -> ADF
# ============================================================================

def texte_vers_adf(texte: str) -> dict:
    """Convertit du texte en document ADF (format de description Jira).

    Une ligne de la forme **Titre** devient un titre de section (heading ADF) :
    c'est la convention produite par reformuler_description. Le texte brut (CLI,
    sans ces marqueurs) reste rendu en simples paragraphes, comme avant.

    NB : on n'ajoute PLUS de ligne « Ticket Zendesk source » : la traçabilité
    Zendesk <-> Jira est désormais assurée par le lien natif du connecteur
    (lier_zendesk_jira) + le champ « Clé Jira liée » côté Zendesk.
    """
    contenu = []
    for ligne in texte.split("\n"):
        ligne = ligne.strip()
        if not ligne:
            continue
        titre = re.fullmatch(r"\*\*(.+)\*\*", ligne)
        if titre:
            contenu.append({"type": "heading", "attrs": {"level": 3},
                            "content": [{"type": "text", "text": titre.group(1).strip()}]})
        else:
            contenu.append({"type": "paragraph",
                            "content": [{"type": "text", "text": ligne}]})
    return {"type": "doc", "version": 1, "content": contenu}


# ============================================================================
# REFORMULATION DE LA DESCRIPTION  (via Claude / API Anthropic)
# ============================================================================

# Modele Anthropic standard VODF (cf. note interne). Ce n'est pas un secret :
# seule la cle (ANTHROPIC_API_KEY, dans le .env) l'est.
MODELE_ANTHROPIC = "claude-haiku-4-5-20251001"

# Consigne donnee a Claude (en "system", au niveau racine — convention VODF, pas de prefill).
# Les titres sont demandes au format **Titre** : texte_vers_adf les transforme alors
# en titres de section (heading ADF) dans la description Jira. Le texte brut du CLI
# (sans ces marqueurs) reste rendu en simples paragraphes -> aucune regression.
PROMPT_REFORMULATION = (
    "Tu transformes un signalement client (issu d'un ticket de support) en un "
    "rapport de bug clair, factuel et DETAILLE pour une equipe technique. "
    "Reformule en francais en TROIS sections. Garde ces titres EXACTS, chacun "
    "entoure de doubles asterisques et seul sur sa ligne, suivi de son contenu :\n"
    "**Description detaillee**\n"
    "**Comportement observe**\n"
    "**Comportement attendu**\n\n"
    "Regles :\n"
    "- Exploite TOUTES les informations utiles du ticket et sois aussi precis et "
    "detaille que possible, MAIS n'invente jamais de versions, plateformes ou "
    "details absents. Si une section manque d'information, ecris "
    "'Non precise dans le ticket.'.\n"
    "- NOMME toujours le client / la plateforme dans la formulation, de maniere "
    "naturelle (ex. « Sur le BO de Benshi… », « Une utilisatrice Benshi remonte "
    "que… »). Utilise le NOM du client, sans le code entre parentheses.\n"
    "- SITUE clairement le probleme quand le ticket le permet : back-office (BO, "
    "interface d'administration utilisee par le client) OU site / front-office "
    "(FO, vu par les utilisateurs finaux). Distingue aussi un probleme rencontre "
    "par le client (admin) d'un probleme remonte par / affectant un utilisateur "
    "final. Si le ticket ne le precise pas, ne l'invente pas.\n"
    "- EVITE les redites : ne repete pas la meme information d'une section a "
    "l'autre ; chaque section apporte un angle different.\n"
    "- Reponds UNIQUEMENT avec le rapport, sans preambule ni commentaire."
)

# Glossaire metier EDITABLE : un fichier texte a cote du code, que tu peux enrichir
# toi-meme (jargon du projet, contexte Otto/VODF...) SANS toucher au code Python.
# Son contenu est injecte dans la consigne de Claude a chaque reformulation. Chemin
# base sur l'emplacement de ce fichier -> trouvable quel que soit le dossier courant.
CHEMIN_GLOSSAIRE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossaire.md")


def _charger_glossaire() -> str:
    """Renvoie le contenu de glossaire.md, ou '' s'il est absent/illisible.

    Defensif (jamais d'exception) : sans glossaire, la reformulation marche comme
    avant. C'est un ENRICHISSEMENT du contexte, pas un point de blocage.
    """
    try:
        with open(CHEMIN_GLOSSAIRE, encoding="utf-8") as fichier:
            return fichier.read().strip()
    except OSError:
        return ""


def reformuler_description(texte_ticket: str, sujet: str = "",
                           contexte: str = "") -> str | None:
    """Reformule la description brute d'un ticket en rapport de bug structure (Claude).

    'contexte' (optionnel) = infos connues de l'outil a injecter pour mieux ancrer
    la reformulation (ex. type de ticket, plateforme/client concerne). Le modele
    est invite a mentionner la plateforme dans la description.

    Renvoie le texte reformule, ou None si l'appel echoue pour une raison
    quelconque (cle absente, reseau, erreur API, reponse inattendue).

    IMPORTANT (robustesse) : cette fonction ne leve JAMAIS d'exception et ne fait
    JAMAIS sys.exit. La reformulation est un CONFORT, jamais un point de blocage :
    l'appelant doit gerer le None en retombant sur le texte brut du ticket. Ainsi
    le serveur web reste utilisable meme si l'IA est indisponible, et la creation
    Jira ne depend jamais de cet appel externe.
    """
    cle = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not cle or not (texte_ticket or "").strip():
        return None

    # Consigne = regles de base + (si present) le glossaire metier editable.
    systeme = PROMPT_REFORMULATION
    glossaire = _charger_glossaire()
    if glossaire:
        systeme += ("\n\n--- Contexte projet et glossaire (sers-t'en pour bien "
                    "interpreter le ticket et employer les bons termes ; n'invente "
                    "rien au-dela de ce que dit le ticket) ---\n" + glossaire)

    try:
        rep = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": cle,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODELE_ANTHROPIC,
                "max_tokens": 1500,
                "system": systeme,
                "messages": [{
                    "role": "user",
                    "content": f"Sujet du ticket : {sujet}\n"
                               + (f"{contexte}\n" if contexte else "")
                               + f"\nDescription brute du client :\n{texte_ticket}",
                }],
            },
            timeout=30,
        )
        if rep.status_code != 200:
            print(f"  [!] Reformulation IA indisponible (HTTP {rep.status_code}).")
            return None
        # La reponse Anthropic est une liste de blocs ; on concatene les blocs texte.
        blocs = rep.json().get("content", [])
        textes = [b.get("text", "") for b in blocs if b.get("type") == "text"]
        resultat = "\n".join(t for t in textes if t).strip()
        return resultat or None
    except requests.RequestException as erreur:
        print(f"  [!] Reformulation IA indisponible : {erreur}")
        return None


# ============================================================================
# ECHEANCE
# ============================================================================

# ============================================================================
# PIECES JOINTES (Zendesk -> Jira)
# ============================================================================

def lister_pieces_jointes(zd: requests.Session, base: str, ticket_id) -> list:
    """Recupere toutes les pieces jointes d'un ticket Zendesk.

    Les fichiers ne sont pas sur le ticket lui-meme mais sur ses COMMENTAIRES.
    On parcourt donc les commentaires et on collecte leurs 'attachments'. On
    deduplique par 'id' (un meme fichier peut apparaitre sur plusieurs commentaires).
    """
    rep = zd.get(f"{base}/tickets/{ticket_id}/comments.json", timeout=20)
    if rep.status_code != 200:
        print(f"  [!] Lecture des commentaires impossible (HTTP {rep.status_code}).")
        return []

    pieces, vus = [], set()
    for commentaire in rep.json().get("comments", []):
        for pj in commentaire.get("attachments", []):
            if pj.get("id") in vus:
                continue
            vus.add(pj.get("id"))
            pieces.append({
                "file_name": pj.get("file_name"),
                "content_url": pj.get("content_url"),
                "content_type": pj.get("content_type") or "application/octet-stream",
                "size": pj.get("size"),
            })
    return pieces


def transferer_pieces_jointes(zd, base_zd, jira, base_jira, ticket_id, cle_jira) -> int:
    """Telecharge les pieces jointes Zendesk et les reuploade sur l'issue Jira.

    Renvoie le nombre de fichiers transferes avec succes.

    Detail important cote Jira : l'upload exige l'en-tete 'X-Atlassian-Token:
    no-check' (protection anti-CSRF), et le fichier est envoye en multipart (le
    parametre 'files=' de requests s'en charge ; il ne faut SURTOUT pas fixer
    soi-meme un Content-Type JSON ici).
    """
    pieces = lister_pieces_jointes(zd, base_zd, ticket_id)
    if not pieces:
        print("  Aucune piece jointe sur ce ticket Zendesk.")
        return 0

    print(f"  {len(pieces)} piece(s) jointe(s) a transferer :")
    url_upload = f"{base_jira}/rest/api/3/issue/{cle_jira}/attachments"
    headers_upload = {"X-Atlassian-Token": "no-check"}
    transferees = 0

    for pj in pieces:
        nom = pj["file_name"]
        # 1. Telechargement depuis Zendesk (session authentifiee).
        #    On surcharge l'en-tete Accept : la session demande du JSON par defaut,
        #    or ici on recupere un fichier binaire (image...). Sans ce "*/*", Zendesk
        #    repond 406 Not Acceptable (il ne peut pas servir du JSON pour un JPG).
        dl = zd.get(pj["content_url"], headers={"Accept": "*/*"}, timeout=60)
        if dl.status_code != 200:
            print(f"    [X] {nom} : telechargement Zendesk echoue (HTTP {dl.status_code}).")
            continue

        # 2. Upload vers Jira en multipart.
        fichier = {"file": (nom, dl.content, pj["content_type"])}
        up = jira.post(url_upload, headers=headers_upload, files=fichier, timeout=120)
        if up.status_code in (200, 201):
            print(f"    [OK] {nom} ({pj['size']} octets) transferee.")
            transferees += 1
        else:
            print(f"    [X] {nom} : upload Jira echoue (HTTP {up.status_code}) {up.text[:150]}")

    return transferees


# ============================================================================
# REECRITURE COTE ZENDESK (boucler la traçabilite + anti-doublon)
# ============================================================================

def ecrire_retour_zendesk(zd: requests.Session, base: str, ticket_id, cle_jira: str) -> bool:
    """Renseigne le champ 'Cle Jira liee' et retire le tag declencheur 'to-jira'.

    Deux ecritures distinctes :
      1. PUT du ticket avec le champ custom -> ne touche QUE ce champ (les autres
         champs custom restent intacts, Zendesk fusionne par id).
      2. DELETE sur l'endpoint tags dedie -> retire UNIQUEMENT 'to-jira', sans
         reecrire la liste complete des tags (plus sur, pas d'effet de bord).

    Renvoie True si le champ a bien ete ecrit (l'etape critique anti-doublon).
    """
    field_id = int(get_required("ZENDESK_JIRA_KEY_FIELD_ID"))

    # 1. Ecrire la cle Jira dans le champ custom (etape CRITIQUE anti-doublon).
    #    Le PUT est idempotent -> la session le rejoue sur erreur transitoire.
    #    Si malgre les re-essais ca echoue (HTTP ou reseau), on renvoie False :
    #    l'appelant loguera un statut distinct pour reperer l'orphelin.
    payload = {"ticket": {"custom_fields": [{"id": field_id, "value": cle_jira}]}}
    try:
        rep = zd.put(f"{base}/tickets/{ticket_id}.json", json=payload, timeout=20)
    except requests.RequestException as erreur:
        print(f"  [X] Ecriture du champ 'Cle Jira liee' impossible (reseau) : {erreur}")
        return False
    if rep.status_code != 200:
        print(f"  [X] Ecriture du champ 'Cle Jira liee' echouee (HTTP {rep.status_code}).")
        print(f"      {rep.text[:200]}")
        return False
    print(f"  [OK] Champ 'Cle Jira liee' = {cle_jira}")

    # 2. Retirer le tag declencheur. Non bloquant : le champ (memoire anti-doublon)
    #    est deja ecrit, c'est l'essentiel. Un echec ici (HTTP ou reseau) ne remet
    #    pas en cause le succes (un re-run reverrait le tag mais sauterait le ticket
    #    grace au champ deja rempli).
    try:
        rep_tag = zd.delete(f"{base}/tickets/{ticket_id}/tags.json",
                            json={"tags": [TAG_DECLENCHEUR]}, timeout=20)
        if rep_tag.status_code == 200:
            print(f"  [OK] Tag '{TAG_DECLENCHEUR}' retire.")
        else:
            print(f"  [!] Retrait du tag '{TAG_DECLENCHEUR}' : HTTP {rep_tag.status_code} "
                  "(le champ est ecrit, c'est le plus important).")
    except requests.RequestException as erreur:
        print(f"  [!] Retrait du tag '{TAG_DECLENCHEUR}' impossible (reseau) : {erreur} "
              "(le champ est ecrit, c'est le plus important).")

    return True


def lier_zendesk_jira(zd: requests.Session, base: str, ticket_id, issue_id, issue_key: str) -> bool:
    """Cree le lien NATIF via le connecteur Jira de Zendesk (ANCIENNE integration).

    Contrairement au champ texte 'Cle Jira liee' (anti-doublon), ce lien est le
    vrai lien du connecteur : il apparait des DEUX cotes automatiquement (panneau
    'Linked Jira issues' cote Zendesk, panneau Zendesk cote Jira).

    API "legacy" (celle active sur l'instance VODF) : POST /api/services/jira/links,
    corps {ticket_id, issue_id (id NUMERIQUE Jira), issue_key}. Aucun external_id.
    NB : la NOUVELLE integration utilise /api/v2/integrations/jira/{external_id}/links
    -> si VODF migre un jour, c'est ici qu'il faudra adapter l'URL.

    Defensif : ne leve JAMAIS. Renvoie True si le lien est cree, False sinon
    (id inconnu, pas les droits, reseau...) -> non bloquant, c'est un PLUS,
    jamais un point de blocage de la creation Jira.
    """
    if not issue_id:
        print("  [!] Lien connecteur Jira saute : id numerique de l'issue inconnu.")
        return False

    # L'endpoint legacy est sous /api/services/ (pas /api/v2/) : on repart de la racine.
    url = base.replace("/api/v2", "") + "/api/services/jira/links"
    payload = {"ticket_id": str(ticket_id), "issue_id": str(issue_id), "issue_key": issue_key}
    try:
        rep = zd.post(url, json=payload, timeout=20)
    except requests.RequestException as erreur:
        print(f"  [!] Lien connecteur Jira impossible (reseau) : {erreur}")
        return False
    if rep.status_code in (200, 201):
        print(f"  [OK] Lien Zendesk #{ticket_id} <-> {issue_key} cree via le connecteur.")
        return True
    print(f"  [!] Lien connecteur Jira non cree (HTTP {rep.status_code}) : {rep.text[:150]}")
    return False


# ============================================================================
# JOURNALISATION (trace des conversions, dans un CSV ouvrable sous Excel)
# ============================================================================

FICHIER_LOG = "conversions.csv"

def journaliser(ticket_id, statut: str, cle_jira: str = "", detail: str = "") -> None:
    """Ajoute une ligne au journal des conversions.

    statut : 'cree' | 'deja_converti' | 'abandonne' | 'echec_creation' | 'erreur'.
    Le fichier est cree avec son en-tete au premier appel. Format CSV pour que tu
    puisses l'ouvrir directement dans Excel et garder une trace de ton activite.
    """
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fichier_neuf = not os.path.exists(FICHIER_LOG)
    with open(FICHIER_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if fichier_neuf:
            writer.writerow(["horodatage", "ticket_zendesk", "statut", "cle_jira", "detail"])
        writer.writerow([horodatage, ticket_id, statut, cle_jira, detail])


def calculer_echeance(nom_priorite: str) -> str:
    """Date d'echeance (format Jira AAAA-MM-JJ) = aujourd'hui + N jours selon la prio.

    NB : on compte a partir d'AUJOURD'HUI (jour de creation du Jira). Pour partir
    de la date de creation du ticket Zendesk, il suffirait de remplacer
    datetime.now() par la date 'created_at' du ticket.
    """
    jours = ECHEANCE_JOURS[nom_priorite]
    return (datetime.now() + timedelta(days=jours)).strftime("%Y-%m-%d")


# ============================================================================
# ASSEMBLAGE DU "fields" JIRA  (fonction PURE, partagee CLI <-> web)
# ============================================================================

def construire_fields(ticket: dict, infos: dict, match: dict,
                      choix: dict) -> tuple[dict, dict]:
    """Assemble le 'fields' Jira a partir des choix de qualification.

    C'est le coeur de l'architecture du brief (§4) : une fonction PURE, qui ne
    pose AUCUNE question et n'affiche rien. Tous les choix de l'utilisateur
    arrivent par le dict 'choix'. Du coup elle est appelee a l'identique par :
      - le CLI  (review_ticket.revue_interactive remplit 'choix' au clavier) ;
      - le web  (une route FastAPI remplira 'choix' depuis un formulaire HTML).

    Parametres
    ----------
    ticket : le ticket Zendesk charge (charger_ticket).
    infos  : sortie de detecter_type_et_assigne (type, gabarit_resume, assigne_*).
    match  : la plateforme DEJA resolue -> {"id": ..., "label": ...}.
    choix  : les reponses de qualification. Cles attendues :
               - App Mobile / App TV : "systeme" = "Android" | "iOS"
               - Web                 : "onglet" (texte) + "bofo" = "BO" | "FO"
               - toujours : "nom_prio" (une cle de PRIORITES),
                            "assigne_id", "assigne_label".

    Renvoie un couple (fields, meta_affichage) :
      - fields         : exactement la structure attendue par {"fields": {...}} de Jira ;
      - meta_affichage : quelques libelles lisibles pour le recapitulatif.
    """
    # --- 1. Resume (nomenclature) selon le type de ticket ------------------
    if infos["type"] in ("App Mobile", "App TV"):
        resume = infos["gabarit_resume"].format(os=choix["systeme"],
                                                 titre=ticket["subject"])
    else:  # Web
        resume = infos["gabarit_resume"].format(onglet=choix["onglet"],
                                                bofo=choix["bofo"],
                                                titre=ticket["subject"])

    # --- 2. Echeance derivee de la priorite --------------------------------
    echeance = calculer_echeance(choix["nom_prio"])

    # --- 3. Description -> ADF ---------------------------------------------
    # Si 'choix' fournit une description (cas web : reformulation editee par
    # l'utilisateur), on l'utilise ; sinon on retombe sur le texte brut du
    # ticket (cas CLI actuel, comportement inchange).
    texte_description = choix.get("description") or ticket["description"]
    description_adf = texte_vers_adf(texte_description)

    # --- 4. Assemblage du "fields" Jira ------------------------------------
    # NB : on NE met PAS de champ "reporter". Du coup Jira affecte le rapporteur
    # (= demandeur) au compte authentifie, c'est-a-dire TOI. Le demandeur Zendesk
    # d'origine n'est donc jamais utilise comme demandeur Jira.
    fields = {
        "project": {"key": get_required("JIRA_PROJECT_KEY")},
        "issuetype": {"id": ISSUE_TYPE_BUG_ID},
        "summary": resume,
        "description": description_adf,
        "priority": {"id": PRIORITES[choix["nom_prio"]]},
        "duedate": echeance,
        "assignee": {"accountId": choix["assigne_id"]},
        get_required("JIRA_PLATFORM_FIELD_ID"): [{"id": match["id"]}],
    }
    # Libelles lisibles a part, juste pour l'affichage du recapitulatif.
    meta_affichage = {"priorite": choix["nom_prio"],
                      "assigne": choix["assigne_label"],
                      "plateforme": match["label"]}
    return fields, meta_affichage
