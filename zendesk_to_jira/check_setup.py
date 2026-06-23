"""
check_setup.py — Verifie que la configuration est correcte AVANT de coder le reste.

Ce script ne cree rien, ne modifie rien : il se contente de lire le .env et de
demander a Zendesk et a Jira "qui suis-je ?". Si les deux repondent, c'est que
nos credentials sont bons et qu'on peut batir l'outil dessus en confiance.

Pourquoi commencer par ca ?
- Une auth qui marche est le socle de tout le reste. Mieux vaut la valider sur
  une requete inoffensive ("qui suis-je") que de decouvrir un 401 au moment de
  creer un vrai ticket.
- Ca te fait voir concretement comment on s'authentifie sur chaque API.

Usage :
    cp .env.example .env      # puis remplis le .env
    python check_setup.py
"""

import os
import sys

import requests
from dotenv import load_dotenv


# ============================================================================
# 1. CHARGEMENT DU .env
# ============================================================================
# load_dotenv() lit le fichier .env et injecte son contenu dans les "variables
# d'environnement" du processus. On les relit ensuite avec os.getenv(...).
# Interet : les secrets vivent dans le .env (jamais committe), pas dans le code.
load_dotenv()


def get_required(nom_variable: str) -> str:
    """Recupere une variable d'environnement obligatoire, ou arrete le script.

    Si la variable est absente ou vide, on prefere s'arreter tout de suite avec
    un message clair plutot que de laisser une requete echouer plus loin avec une
    erreur obscure.
    """
    valeur = os.getenv(nom_variable, "").strip()
    if not valeur:
        print(f"  [X] Variable manquante ou vide dans le .env : {nom_variable}")
        print("      -> Ouvre ton .env et renseigne cette valeur, puis relance.")
        sys.exit(1)
    return valeur


# ============================================================================
# 2. VERIFICATION ZENDESK
# ============================================================================

def verifier_zendesk() -> None:
    """Appelle l'endpoint Zendesk "users/me" pour confirmer que l'auth marche."""
    print("\n[Zendesk] Verification de l'authentification...")

    subdomain = get_required("ZENDESK_SUBDOMAIN")
    email = get_required("ZENDESK_EMAIL")
    token = get_required("ZENDESK_API_TOKEN")

    # Auth Zendesk par token API : le username est litteralement "email/token"
    # (le "/token" fait partie de la chaine, ce n'est pas un separateur). Sans
    # ce suffixe, Zendesk repond 401. Le mot de passe est le token lui-meme.
    auth = (f"{email}/token", token)

    # "users/me" = "donne-moi les infos du compte qui fait cette requete".
    # C'est l'appel le plus sur pour tester une auth : aucune ecriture, et il
    # echoue clairement si les credentials sont faux.
    url = f"https://{subdomain}.zendesk.com/api/v2/users/me.json"

    try:
        reponse = requests.get(url, auth=auth, timeout=15)
    except requests.RequestException as erreur:
        print(f"  [X] Impossible de joindre Zendesk : {erreur}")
        sys.exit(1)

    if reponse.status_code == 200:
        moi = reponse.json().get("user", {})
        print(f"  [OK] Connecte en tant que : {moi.get('name')} <{moi.get('email')}>")
        print(f"       Role Zendesk : {moi.get('role')}")
    elif reponse.status_code == 401:
        print("  [X] 401 Unauthorized : email ou token API incorrect.")
        print("      -> Verifie ZENDESK_EMAIL et ZENDESK_API_TOKEN dans le .env.")
        sys.exit(1)
    else:
        print(f"  [X] Reponse inattendue : HTTP {reponse.status_code}")
        print(f"      {reponse.text[:300]}")
        sys.exit(1)


# ============================================================================
# 3. VERIFICATION JIRA
# ============================================================================

def verifier_jira() -> None:
    """Appelle l'endpoint Jira "myself" pour confirmer que l'auth marche."""
    print("\n[Jira] Verification de l'authentification...")

    base_url = get_required("JIRA_BASE_URL").rstrip("/")
    email = get_required("JIRA_EMAIL")
    token = get_required("JIRA_API_TOKEN")
    projet = get_required("JIRA_PROJECT_KEY")

    # Auth Jira Cloud : Basic Auth avec (email, token API). Plus simple que
    # Zendesk : pas de suffixe "/token" a ajouter, l'email brut suffit.
    auth = (email, token)
    headers = {"Accept": "application/json"}

    # "myself" = l'equivalent Jira de "users/me" : les infos de mon compte.
    url = f"{base_url}/rest/api/3/myself"

    try:
        reponse = requests.get(url, auth=auth, headers=headers, timeout=15)
    except requests.RequestException as erreur:
        print(f"  [X] Impossible de joindre Jira : {erreur}")
        sys.exit(1)

    if reponse.status_code == 200:
        moi = reponse.json()
        print(f"  [OK] Connecte en tant que : {moi.get('displayName')} <{moi.get('emailAddress')}>")
        # On garde l'accountId sous les yeux : c'est ainsi que Jira identifie un
        # utilisateur (et c'est ce format qu'on utilisera pour les assignations).
        print(f"       Ton accountId : {moi.get('accountId')}")
    elif reponse.status_code == 401:
        print("  [X] 401 Unauthorized : email ou token API incorrect.")
        print("      -> Verifie JIRA_EMAIL et JIRA_API_TOKEN dans le .env.")
        sys.exit(1)
    else:
        print(f"  [X] Reponse inattendue : HTTP {reponse.status_code}")
        print(f"      {reponse.text[:300]}")
        sys.exit(1)

    # Verification bonus : le projet de destination existe-t-il et y as-tu acces ?
    url_projet = f"{base_url}/rest/api/3/project/{projet}"
    reponse_projet = requests.get(url_projet, auth=auth, headers=headers, timeout=15)
    if reponse_projet.status_code == 200:
        nom_projet = reponse_projet.json().get("name")
        print(f"  [OK] Projet '{projet}' accessible : {nom_projet}")
    else:
        print(f"  [!] Projet '{projet}' introuvable ou inaccessible "
              f"(HTTP {reponse_projet.status_code}).")
        print("      -> Verifie JIRA_PROJECT_KEY, ou tes droits sur ce projet.")


# ============================================================================
# 4. POINT D'ENTREE
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Verification de la configuration Zendesk -> Jira")
    print("=" * 70)

    verifier_zendesk()
    verifier_jira()

    print("\n" + "=" * 70)
    print("[OK] Configuration valide. On peut passer a la suite (etape 1b).")
    print("=" * 70)


if __name__ == "__main__":
    main()
