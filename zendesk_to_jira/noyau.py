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
import sys
import unicodedata
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()


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
    """Lit une variable obligatoire du .env, ou arrete avec un message clair."""
    valeur = os.getenv(nom, "").strip()
    if not valeur:
        print(f"  [X] Variable manquante dans le .env : {nom}")
        sys.exit(1)
    return valeur


def session_zendesk() -> tuple[requests.Session, str]:
    sub = get_required("ZENDESK_SUBDOMAIN")
    s = requests.Session()
    s.auth = (f"{get_required('ZENDESK_EMAIL')}/token", get_required("ZENDESK_API_TOKEN"))
    s.headers.update({"Accept": "application/json"})
    return s, f"https://{sub}.zendesk.com/api/v2"


def session_jira() -> tuple[requests.Session, str]:
    base = get_required("JIRA_BASE_URL").rstrip("/")
    s = requests.Session()
    s.auth = (get_required("JIRA_EMAIL"), get_required("JIRA_API_TOKEN"))
    s.headers.update({"Accept": "application/json"})
    return s, base


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
        print(f"  [X] Ticket {ticket_id} introuvable (404).")
        sys.exit(1)
    if rep.status_code != 200:
        print(f"  [X] Erreur HTTP {rep.status_code} : {rep.text[:200]}")
        sys.exit(1)
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


# ============================================================================
# DETECTION TYPE + ASSIGNE (via les tags)
# ============================================================================

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
        print(f"  [X] Lecture options plateforme impossible (HTTP {rep.status_code}).")
        sys.exit(1)
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

def texte_vers_adf(texte: str, url_source: str) -> dict:
    """Convertit du texte brut en document ADF + ajoute le lien Zendesk source."""
    contenu = []
    for ligne in texte.split("\n"):
        ligne = ligne.strip()
        if ligne:
            contenu.append({"type": "paragraph",
                            "content": [{"type": "text", "text": ligne}]})
    contenu.append({"type": "rule"})
    contenu.append({"type": "paragraph", "content": [
        {"type": "text", "text": "Ticket Zendesk source : "},
        {"type": "text", "text": url_source,
         "marks": [{"type": "link", "attrs": {"href": url_source}}]},
    ]})
    return {"type": "doc", "version": 1, "content": contenu}


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

    # 1. Ecrire la cle Jira dans le champ custom.
    payload = {"ticket": {"custom_fields": [{"id": field_id, "value": cle_jira}]}}
    rep = zd.put(f"{base}/tickets/{ticket_id}.json", json=payload, timeout=20)
    if rep.status_code != 200:
        print(f"  [X] Ecriture du champ 'Cle Jira liee' echouee (HTTP {rep.status_code}).")
        print(f"      {rep.text[:200]}")
        return False
    print(f"  [OK] Champ 'Cle Jira liee' = {cle_jira}")

    # 2. Retirer le tag declencheur.
    rep_tag = zd.delete(f"{base}/tickets/{ticket_id}/tags.json",
                        json={"tags": [TAG_DECLENCHEUR]}, timeout=20)
    if rep_tag.status_code == 200:
        print(f"  [OK] Tag '{TAG_DECLENCHEUR}' retire.")
    else:
        # Non bloquant : le champ (memoire anti-doublon) est deja ecrit, c'est l'essentiel.
        print(f"  [!] Retrait du tag '{TAG_DECLENCHEUR}' : HTTP {rep_tag.status_code} "
              "(le champ est ecrit, c'est le plus important).")

    return True


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
