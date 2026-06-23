"""
create_jira_key_field.py — Cree (une seule fois) le champ Zendesk "Cle Jira liee".

ETAPE 1b, operation d'ECRITURE. C'est le SEUL script de cette etape qui modifie
quelque chose : il ajoute un champ personnalise a la configuration de tes tickets
Zendesk. Il ne touche a aucun ticket, juste a la definition des champs.

Ce champ est la "memoire" de notre anti-doublon (brief §8) : une fois un Jira
cree, l'outil y inscrira la cle (ex. BUGOTTO-1234). Avant toute creation, l'outil
verifiera que ce champ est vide -> pas de doublon possible.

SECURITE / IDEMPOTENCE : le script verifie d'abord si un champ du meme titre
existe deja. Si oui, il NE recree RIEN (sinon on aurait deux champs identiques)
et se contente d'afficher l'ID existant. Tu peux donc le relancer sans risque.

Usage :
    python create_jira_key_field.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# Titre exact du champ a creer. On le definit en constante pour qu'il serve a la
# fois a la recherche (existe-t-il deja ?) et a la creation.
TITRE_CHAMP = "Clé Jira liée"
DESCRIPTION_CHAMP = (
    "Clé du ticket Jira créé à partir de ce ticket Zendesk "
    "(rempli automatiquement par l'outil Zendesk → Jira). "
    "Si ce champ est rempli, le ticket a déjà été converti : ne pas le retraiter."
)


def get_required(nom_variable: str) -> str:
    valeur = os.getenv(nom_variable, "").strip()
    if not valeur:
        print(f"  [X] Variable manquante dans le .env : {nom_variable}")
        sys.exit(1)
    return valeur


def main() -> None:
    subdomain = get_required("ZENDESK_SUBDOMAIN")
    email = get_required("ZENDESK_EMAIL")
    token = get_required("ZENDESK_API_TOKEN")
    auth = (f"{email}/token", token)
    base = f"https://{subdomain}.zendesk.com/api/v2/ticket_fields.json"

    print("=" * 70)
    print(f"Creation du champ Zendesk : \"{TITRE_CHAMP}\"")
    print("=" * 70)

    # --- 1. Anti-doublon : le champ existe-t-il deja ? ----------------------
    print("\n[1/2] Verification qu'un champ du meme titre n'existe pas deja...")
    reponse = requests.get(base, auth=auth, timeout=15)
    if reponse.status_code != 200:
        print(f"  [X] Lecture des champs impossible (HTTP {reponse.status_code}).")
        sys.exit(1)

    for champ in reponse.json().get("ticket_fields", []):
        if champ.get("title", "").strip().lower() == TITRE_CHAMP.lower():
            print(f"  [!] Un champ \"{TITRE_CHAMP}\" existe deja (ID {champ['id']}).")
            print("      On ne recree rien. Recopie cet ID dans le .env :")
            print(f"\n      ZENDESK_JIRA_KEY_FIELD_ID={champ['id']}\n")
            return

    # --- 2. Creation --------------------------------------------------------
    print("  [OK] Aucun champ existant. On peut le creer.")
    print("\n[2/2] Creation du champ (type texte)...")

    # Payload de creation. "type": "text" = champ texte sur une ligne, parfait
    # pour stocker une cle comme BUGOTTO-1234.
    payload = {
        "ticket_field": {
            "type": "text",
            "title": TITRE_CHAMP,
            "description": DESCRIPTION_CHAMP,
        }
    }

    reponse = requests.post(base, json=payload, auth=auth, timeout=15)
    if reponse.status_code in (200, 201):
        nouveau = reponse.json().get("ticket_field", {})
        print(f"  [OK] Champ cree. ID = {nouveau.get('id')}")
        print("\n  Recopie cette ligne dans ton .env :")
        print(f"\n      ZENDESK_JIRA_KEY_FIELD_ID={nouveau.get('id')}\n")
    else:
        print(f"  [X] Echec de creation (HTTP {reponse.status_code}) :")
        print(f"      {reponse.text[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
