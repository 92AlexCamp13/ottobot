"""
fetch_ticket.py — Recupere UN ticket Zendesk par son ID et l'affiche (lecture seule).

ETAPE 2 du plan. Aucune ecriture, ni dans Zendesk ni dans Jira : on se contente
de lire un ticket et de montrer proprement les donnees dont l'outil aura besoin
pour construire le futur Jira (etape 3). C'est aussi l'occasion de voir la vraie
valeur du champ "Plateforme" (un tagger) sur un cas concret.

Usage :
    python fetch_ticket.py 12345        # 12345 = l'ID du ticket Zendesk
"""

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def get_required(nom_variable: str) -> str:
    valeur = os.getenv(nom_variable, "").strip()
    if not valeur:
        print(f"  [X] Variable manquante dans le .env : {nom_variable}")
        sys.exit(1)
    return valeur


def faire_session() -> tuple[requests.Session, str]:
    """Prepare une session HTTP authentifiee + l'URL de base de l'API Zendesk.

    On regroupe l'auth dans une Session pour ne pas la repeter a chaque appel
    (on fera 2 requetes : le ticket, puis le demandeur).
    """
    subdomain = get_required("ZENDESK_SUBDOMAIN")
    email = get_required("ZENDESK_EMAIL")
    token = get_required("ZENDESK_API_TOKEN")

    session = requests.Session()
    session.auth = (f"{email}/token", token)  # rappel : "/token" fait partie du username
    session.headers.update({"Accept": "application/json"})

    base_url = f"https://{subdomain}.zendesk.com/api/v2"
    return session, base_url


def lire_champ_custom(custom_fields: list, field_id: int):
    """Retrouve la valeur d'un champ custom dans la liste 'custom_fields' du ticket.

    Le ticket Zendesk porte ses champs custom sous la forme :
        [{"id": 7010347166749, "value": "..."}, ...]
    On cherche celui dont l'id correspond a notre champ, et on renvoie sa valeur
    (ou None si le champ n'est pas present / vide).
    """
    for champ in custom_fields:
        if str(champ.get("id")) == str(field_id):
            return champ.get("value")
    return None


def main() -> None:
    # --- Lecture de l'argument : l'ID du ticket ----------------------------
    parser = argparse.ArgumentParser(description="Affiche un ticket Zendesk par son ID.")
    parser.add_argument("ticket_id", help="ID numerique du ticket Zendesk a afficher")
    args = parser.parse_args()

    session, base_url = faire_session()

    # IDs de nos 2 champs custom, lus depuis le .env (decouverts a l'etape 1b).
    id_champ_plateforme = get_required("ZENDESK_CLIENT_FIELD_ID")
    id_champ_cle_jira = get_required("ZENDESK_JIRA_KEY_FIELD_ID")

    # --- 1. Recuperer le ticket --------------------------------------------
    print("=" * 70)
    print(f"Ticket Zendesk #{args.ticket_id}")
    print("=" * 70)

    url_ticket = f"{base_url}/tickets/{args.ticket_id}.json"
    reponse = session.get(url_ticket, timeout=15)

    if reponse.status_code == 404:
        print(f"  [X] Aucun ticket avec l'ID {args.ticket_id} (404).")
        sys.exit(1)
    if reponse.status_code != 200:
        print(f"  [X] Erreur HTTP {reponse.status_code} : {reponse.text[:200]}")
        sys.exit(1)

    ticket = reponse.json().get("ticket", {})

    # --- 2. Recuperer le demandeur (le ticket ne donne qu'un requester_id) --
    nom_demandeur, email_demandeur = "(inconnu)", ""
    requester_id = ticket.get("requester_id")
    if requester_id:
        rep_user = session.get(f"{base_url}/users/{requester_id}.json", timeout=15)
        if rep_user.status_code == 200:
            user = rep_user.json().get("user", {})
            nom_demandeur = user.get("name", "(inconnu)")
            email_demandeur = user.get("email", "")

    # --- 3. Afficher proprement --------------------------------------------
    custom_fields = ticket.get("custom_fields", [])
    plateforme = lire_champ_custom(custom_fields, id_champ_plateforme)
    cle_jira = lire_champ_custom(custom_fields, id_champ_cle_jira)

    print(f"\n  Sujet      : {ticket.get('subject')}")
    print(f"  Statut     : {ticket.get('status')}")
    print(f"  Priorite   : {ticket.get('priority')}        (priorite Zendesk)")
    print(f"  Cree le    : {ticket.get('created_at')}")
    print(f"  Demandeur  : {nom_demandeur} <{email_demandeur}>")
    print(f"  Tags       : {', '.join(ticket.get('tags', [])) or '(aucun)'}")

    print("\n  -- Champs custom qui nous interessent --")
    print(f"  Plateforme    : {plateforme!r}   <- valeur brute du tagger")
    if cle_jira:
        print(f"  Cle Jira liee : {cle_jira!r}   /!\\ DEJA REMPLI -> ticket deja converti")
    else:
        print(f"  Cle Jira liee : (vide)   -> OK, jamais converti")

    print("\n  -- Description --")
    description = ticket.get("description", "") or "(vide)"
    print("  " + description.replace("\n", "\n  "))

    print("\n" + "=" * 70)
    print("Lecture terminee (aucune modification effectuee).")
    print("=" * 70)


if __name__ == "__main__":
    main()
