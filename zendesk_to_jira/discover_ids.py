"""
discover_ids.py — Decouvre les identifiants techniques dont l'outil a besoin.

ETAPE 1b, en LECTURE SEULE : ce script n'ecrit rien, ni dans Zendesk ni dans
Jira. Il interroge les deux API et t'affiche, sous forme de listes lisibles :

  1. Les champs personnalises (custom) de tes tickets Zendesk, avec leur ID.
     -> pour reperer le champ "Client / Plateforme".
  2. Les champs personnalises de Jira, avec leur identifiant customfield_XXXXX.
     -> pour reperer le champ "Plateforme (client ID)".
  3. Les comptes Jira correspondant a "Soufiane" et a "VODF".
     -> pour recuperer les accountId d'assignation.

Tu n'as plus qu'a recopier les bons IDs dans ton .env.

Usage :
    python discover_ids.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def get_required(nom_variable: str) -> str:
    """Recupere une variable obligatoire du .env, ou arrete avec un message clair."""
    valeur = os.getenv(nom_variable, "").strip()
    if not valeur:
        print(f"  [X] Variable manquante dans le .env : {nom_variable}")
        sys.exit(1)
    return valeur


# ============================================================================
# 1. CHAMPS PERSONNALISES ZENDESK
# ============================================================================

def lister_champs_zendesk() -> None:
    """Affiche les champs de ticket Zendesk (on met en avant les champs custom).

    L'endpoint /api/v2/ticket_fields.json renvoie TOUS les champs de ticket :
    les champs systeme (sujet, description, statut, priorite...) ET les champs
    personnalises crees par ton organisation. Ce sont ces derniers qui nous
    interessent : ils ont un ID numerique long, c'est ce qu'on stocke dans le .env.
    """
    print("\n" + "=" * 70)
    print("1. CHAMPS DE TICKET ZENDESK")
    print("=" * 70)

    subdomain = get_required("ZENDESK_SUBDOMAIN")
    email = get_required("ZENDESK_EMAIL")
    token = get_required("ZENDESK_API_TOKEN")
    auth = (f"{email}/token", token)

    url = f"https://{subdomain}.zendesk.com/api/v2/ticket_fields.json"
    reponse = requests.get(url, auth=auth, timeout=15)
    if reponse.status_code != 200:
        print(f"  [X] Echec (HTTP {reponse.status_code}) : {reponse.text[:200]}")
        return

    champs = reponse.json().get("ticket_fields", [])

    # Les champs "systeme" ont un type standard ; on les separe des customs pour
    # que tu reperes vite les champs maison (Client, et plus tard Cle Jira liee).
    types_systeme = {"subject", "description", "status", "priority", "tickettype",
                     "group", "assignee", "custom_status"}

    print("\n  -- Champs personnalises (ceux qui nous interessent) --")
    print(f"  {'ID':<16} {'TYPE':<14} TITRE")
    print("  " + "-" * 60)
    for champ in champs:
        if champ.get("type") not in types_systeme:
            print(f"  {champ['id']:<16} {champ.get('type',''):<14} {champ.get('title','')}")

    print("\n  (Repere la ligne du champ qui contient le CLIENT / la PLATEFORME :")
    print("   son ID va dans ZENDESK_CLIENT_FIELD_ID.")
    print("   Le champ 'Cle Jira liee' n'existe probablement pas encore : on le")
    print("   creera juste apres, a l'etape suivante.)")


# ============================================================================
# 2. CHAMPS PERSONNALISES JIRA
# ============================================================================

def lister_champs_jira() -> None:
    """Affiche les champs custom Jira (customfield_XXXXX) avec leur nom.

    L'endpoint /rest/api/3/field liste TOUS les champs Jira. On ne garde que les
    champs custom (cle "custom": true), car c'est parmi eux que se trouve
    "Plateforme (client ID)". Les champs natifs (summary, priority...) portent un
    nom simple et ne nous interessent pas ici.
    """
    print("\n" + "=" * 70)
    print("2. CHAMPS PERSONNALISES JIRA")
    print("=" * 70)

    base_url = get_required("JIRA_BASE_URL").rstrip("/")
    email = get_required("JIRA_EMAIL")
    token = get_required("JIRA_API_TOKEN")
    auth = (email, token)
    headers = {"Accept": "application/json"}

    url = f"{base_url}/rest/api/3/field"
    reponse = requests.get(url, auth=auth, headers=headers, timeout=15)
    if reponse.status_code != 200:
        print(f"  [X] Echec (HTTP {reponse.status_code}) : {reponse.text[:200]}")
        return

    champs = reponse.json()
    customs = [c for c in champs if c.get("custom")]

    print(f"\n  {'IDENTIFIANT':<22} NOM")
    print("  " + "-" * 60)
    for champ in sorted(customs, key=lambda c: c.get("name", "")):
        print(f"  {champ['id']:<22} {champ.get('name','')}")

    print("\n  (Repere la ligne 'Plateforme (client ID)' : son identifiant")
    print("   customfield_XXXXX va dans JIRA_PLATFORM_FIELD_ID.)")


# ============================================================================
# 3. COMPTES JIRA (pour les assignations)
# ============================================================================

def chercher_utilisateurs_jira(terme: str) -> None:
    """Recherche des comptes Jira par nom/email et affiche leur accountId.

    Jira n'assigne pas un ticket "a un nom" mais "a un accountId". On utilise
    /rest/api/3/user/search qui prend un terme (nom ou email) et renvoie les
    comptes correspondants. On affichera accountId + nom + email pour que tu
    identifies sans ambiguite le bon compte.
    """
    base_url = get_required("JIRA_BASE_URL").rstrip("/")
    email = get_required("JIRA_EMAIL")
    token = get_required("JIRA_API_TOKEN")
    auth = (email, token)
    headers = {"Accept": "application/json"}

    url = f"{base_url}/rest/api/3/user/search"
    reponse = requests.get(url, auth=auth, headers=headers,
                           params={"query": terme}, timeout=15)
    if reponse.status_code != 200:
        print(f"  [X] Echec recherche '{terme}' (HTTP {reponse.status_code})")
        return

    resultats = reponse.json()
    print(f"\n  Recherche '{terme}' -> {len(resultats)} resultat(s) :")
    if not resultats:
        print("    (aucun. Le compte porte peut-etre un autre nom : dis-le moi.)")
    for u in resultats:
        actif = "actif" if u.get("active") else "inactif"
        print(f"    accountId : {u.get('accountId')}")
        print(f"        nom   : {u.get('displayName')} <{u.get('emailAddress','?')}> ({actif})")


def lister_comptes_jira() -> None:
    print("\n" + "=" * 70)
    print("3. COMPTES JIRA POUR LES ASSIGNATIONS")
    print("=" * 70)
    # On cherche les deux destinataires d'assignation prevus par la regle metier.
    chercher_utilisateurs_jira("Soufiane")
    chercher_utilisateurs_jira("VODF")

    print("\n  (Recopie l'accountId de El Amrani Soufiane dans JIRA_ASSIGNEE_SOUFIANE,")
    print("   et celui du compte Tech VODF dans JIRA_ASSIGNEE_TECH_VODF.)")


# ============================================================================
# POINT D'ENTREE
# ============================================================================

def main() -> None:
    print("Decouverte des identifiants (lecture seule, rien n'est modifie)")
    lister_champs_zendesk()
    lister_champs_jira()
    lister_comptes_jira()
    print("\n" + "=" * 70)
    print("Termine. Recopie les IDs reperes dans ton .env, puis dis-le moi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
