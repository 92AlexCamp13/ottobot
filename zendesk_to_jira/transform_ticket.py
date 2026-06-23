"""
transform_ticket.py — Transforme un ticket Zendesk en BROUILLON de ticket Jira.

ETAPE 3 du plan. On lit un ticket Zendesk (comme a l'etape 2) et on construit,
EN MEMOIRE, le "fields" que Jira attend pour creer un Bug. On AFFICHE ce brouillon
mais on NE CREE RIEN : aucune ecriture, ni Zendesk ni Jira.

L'idee : separer clairement ce qui est AUTO-DEDUIT (type, assigne, plateforme,
description) de ce que tu SAISIRAS A LA VALIDATION (priorite, et les morceaux
<Android/iOS> ou <Onglet BO> du resume). Cette etape interactive viendra apres.

Usage :
    python transform_ticket.py 4927
"""

import argparse
import os
import re
import sys
import unicodedata

import requests
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONSTANTES METIER (decouvertes via inspect_jira_meta.py — voir createmeta)
# ============================================================================

# Type d'issue "Bug" dans le projet BUGOTTO.
ISSUE_TYPE_BUG_ID = "10004"

# Priorites BUGOTTO : nom lisible -> id Jira a envoyer.
PRIORITES = {
    "Mineur": "5",
    "Medium": "10003",
    "Majeur": "10002",
    "Bloquant": "10000",
    "Incident": "4",
}

# Regle d'echeance : nombre de jours a ajouter a la date de creation, selon la prio.
ECHEANCE_JOURS = {"Mineur": 30, "Medium": 30, "Majeur": 7, "Bloquant": 2, "Incident": 2}

# Tags Zendesk qui signalent une app -> assignation a Soufiane.
TAG_APP_MOBILE = "application_s__mobile"
TAG_APP_TV = "application_s__tv"

# Petite table d'EXCEPTIONS pour le match plateforme : a remplir UNIQUEMENT pour
# les cas que l'auto-match normalise rate. Cle = valeur Zendesk, valeur = libelle
# Jira exact. Vide pour l'instant : on verra a l'usage.
EXCEPTIONS_PLATEFORME = {
    # "valeur_zendesk": "Libelle Jira exact (NNN)",
}


# ============================================================================
# SESSIONS HTTP
# ============================================================================

def get_required(nom: str) -> str:
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


# ============================================================================
# 1. LECTURE DU TICKET ZENDESK (comme etape 2)
# ============================================================================

def lire_champ_custom(custom_fields: list, field_id: str):
    for champ in custom_fields:
        if str(champ.get("id")) == str(field_id):
            return champ.get("value")
    return None


def charger_ticket(zd: requests.Session, base: str, ticket_id: str) -> dict:
    """Recupere le ticket + le nom du demandeur. Renvoie un dict simple a exploiter."""
    rep = zd.get(f"{base}/tickets/{ticket_id}.json", timeout=15)
    if rep.status_code == 404:
        print(f"  [X] Ticket {ticket_id} introuvable (404).")
        sys.exit(1)
    if rep.status_code != 200:
        print(f"  [X] Erreur HTTP {rep.status_code} : {rep.text[:200]}")
        sys.exit(1)
    t = rep.json().get("ticket", {})

    # Demandeur : le ticket ne donne qu'un id, on resout le nom/email.
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
        "demandeur": f"{nom} <{email}>",
        "plateforme_zendesk": lire_champ_custom(t.get("custom_fields", []),
                                                get_required("ZENDESK_CLIENT_FIELD_ID")),
        "cle_jira_existante": lire_champ_custom(t.get("custom_fields", []),
                                                get_required("ZENDESK_JIRA_KEY_FIELD_ID")),
    }


# ============================================================================
# 2. DETECTION TYPE + ASSIGNE (via les tags)
# ============================================================================

def detecter_type_et_assigne(tags: list) -> dict:
    """Determine le type de ticket et l'assigne suggere a partir des tags.

    Regle (verrouillee) : un tag d'app (mobile ou TV) -> Soufiane ; sinon -> Tech VODF.
    Le 'gabarit_resume' montre le format de nomenclature attendu, avec des
    <placeholders> pour les morceaux que tu saisiras a la validation.
    """
    if TAG_APP_MOBILE in tags:
        return {"type": "App Mobile",
                "gabarit_resume": "[App Mobile] <Android/iOS> - {titre}",
                "assigne_id": get_required("JIRA_ASSIGNEE_SOUFIANE"),
                "assigne_label": "EL AMRANI Soufiane"}
    if TAG_APP_TV in tags:
        return {"type": "App TV",
                "gabarit_resume": "[App TV] <Android/iOS> - {titre}",
                "assigne_id": get_required("JIRA_ASSIGNEE_SOUFIANE"),
                "assigne_label": "EL AMRANI Soufiane"}
    return {"type": "Web",
            "gabarit_resume": "[<Onglet BO>] <BO/FO> - {titre}",
            "assigne_id": get_required("JIRA_ASSIGNEE_TECH_VODF"),
            "assigne_label": "Tech VODF"}


# ============================================================================
# 3. MATCH PLATEFORME (Option A : nom normalise)
# ============================================================================

def normaliser(libelle: str) -> str:
    """Reduit un libelle a sa forme canonique pour la comparaison.

    Etapes : retirer le code final '(303)', enlever les accents, passer en
    minuscules, ne garder que lettres+chiffres. Ainsi 'Studio17 (303)' et le
    tagger Zendesk 'studio17' donnent tous deux 'studio17'.
    """
    if not libelle:
        return ""
    sans_code = re.sub(r"\s*\([^)]*\)\s*$", "", libelle)            # retire " (303)"
    sans_accents = unicodedata.normalize("NFKD", sans_code)         # decompose les accents
    sans_accents = sans_accents.encode("ascii", "ignore").decode()  # supprime les accents
    return re.sub(r"[^a-z0-9]", "", sans_accents.lower())           # garde a-z0-9


def charger_options_plateforme(jira: requests.Session, base: str) -> list:
    """Recupere la liste des options du champ multi-select 'Plateforme'."""
    projet = get_required("JIRA_PROJECT_KEY")
    champ = get_required("JIRA_PLATFORM_FIELD_ID")
    url = f"{base}/rest/api/3/issue/createmeta/{projet}/issuetypes/{ISSUE_TYPE_BUG_ID}"
    rep = jira.get(url, timeout=15)
    if rep.status_code != 200:
        print(f"  [X] Impossible de lire les options plateforme (HTTP {rep.status_code}).")
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
    """Trouve l'option Jira correspondant a la valeur Zendesk. Renvoie {id,label} ou None.

    1. On regarde d'abord la table d'exceptions (cas particuliers).
    2. Sinon, on compare les formes normalisees (Option A).
    """
    if not valeur_zendesk:
        return None

    # 1. Exception explicite ?
    if valeur_zendesk in EXCEPTIONS_PLATEFORME:
        cible = normaliser(EXCEPTIONS_PLATEFORME[valeur_zendesk])
    else:
        cible = normaliser(valeur_zendesk)

    # 2. Recherche par nom normalise.
    for opt in options:
        libelle = opt.get("value") or opt.get("name") or ""
        if normaliser(libelle) == cible:
            return {"id": opt.get("id"), "label": libelle}
    return None


# ============================================================================
# 4. DESCRIPTION -> ADF (format obligatoire de l'API Jira v3)
# ============================================================================

def texte_vers_adf(texte: str, url_source: str) -> dict:
    """Convertit du texte brut en document ADF, et ajoute le lien Zendesk source.

    ADF (Atlassian Document Format) = la structure JSON qu'attend Jira v3 pour les
    champs riches. Le plus simple : un paragraphe par ligne non vide.
    """
    contenu = []
    for ligne in texte.split("\n"):
        ligne = ligne.strip()
        if ligne:
            contenu.append({"type": "paragraph",
                            "content": [{"type": "text", "text": ligne}]})

    # Bloc de tracabilite : lien vers le ticket Zendesk d'origine.
    contenu.append({"type": "rule"})
    contenu.append({"type": "paragraph", "content": [
        {"type": "text", "text": "Ticket Zendesk source : "},
        {"type": "text", "text": url_source,
         "marks": [{"type": "link", "attrs": {"href": url_source}}]},
    ]})

    return {"type": "doc", "version": 1, "content": contenu}


# ============================================================================
# 5. ASSEMBLAGE DU BROUILLON + AFFICHAGE
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le brouillon Jira d'un ticket Zendesk.")
    parser.add_argument("ticket_id", help="ID du ticket Zendesk")
    args = parser.parse_args()

    zd, base_zd = session_zendesk()
    jira, base_jira = session_jira()

    print("=" * 70)
    print(f"Brouillon Jira pour le ticket Zendesk #{args.ticket_id}")
    print("=" * 70)

    ticket = charger_ticket(zd, base_zd, args.ticket_id)

    # Garde-fou anti-doublon : si la cle Jira est deja remplie, on previent.
    if ticket["cle_jira_existante"]:
        print(f"\n  /!\\ Ce ticket a DEJA une cle Jira ({ticket['cle_jira_existante']!r}).")
        print("      En conditions reelles, on le sauterait. On continue ici pour la demo.")

    infos = detecter_type_et_assigne(ticket["tags"])
    options = charger_options_plateforme(jira, base_jira)
    match = matcher_plateforme(ticket["plateforme_zendesk"], options)

    url_source = f"{base_zd.replace('/api/v2', '')}/agent/tickets/{ticket['id']}"
    description_adf = texte_vers_adf(ticket["description"], url_source)

    # --- Affichage lisible -------------------------------------------------
    print(f"\n  Type detecte (via tags) : {infos['type']}")
    print(f"  Tags                    : {', '.join(ticket['tags'])}")

    print("\n  -- AUTO-DEDUIT --")
    print(f"  Resume (gabarit) : {infos['gabarit_resume'].format(titre=ticket['subject'])}")
    print(f"  Assigne suggere  : {infos['assigne_label']}  (id {infos['assigne_id']})")
    print(f"  Demandeur        : {ticket['demandeur']}")
    if match:
        print(f"  Plateforme       : '{ticket['plateforme_zendesk']}' -> "
              f"{match['label']} (id {match['id']})  [match OK]")
    else:
        print(f"  Plateforme       : '{ticket['plateforme_zendesk']}' -> "
              f"AUCUN MATCH /!\\  (a choisir a la validation)")

    print("\n  -- A SAISIR A LA VALIDATION (etape suivante) --")
    print(f"  Priorite         : <a choisir parmi {list(PRIORITES)}>")
    print(f"  Echeance         : <calculee selon la priorite : {ECHEANCE_JOURS}>")
    if infos["type"] == "Web":
        print("  Onglet BO + BO/FO : <a saisir>")
    else:
        print("  Android / iOS    : <a saisir>")

    print("\n  -- DESCRIPTION convertie en ADF (extrait du contenu) --")
    for bloc in description_adf["content"][:3]:
        if bloc["type"] == "paragraph":
            print(f"    | {bloc['content'][0]['text'][:80]}")
    print(f"    ... ({len(description_adf['content'])} blocs ADF au total, lien source inclus)")

    print("\n" + "=" * 70)
    print("Brouillon construit en memoire. AUCUNE creation effectuee.")
    print("=" * 70)


if __name__ == "__main__":
    main()
