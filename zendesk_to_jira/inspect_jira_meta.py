"""
inspect_jira_meta.py — Affiche ce que Jira attend pour creer un Bug dans BUGOTTO.

ETAPE 3 (preparation), LECTURE SEULE. Pour construire un brouillon de ticket Jira
VALIDE, il faut connaitre les regles du projet : quels champs sont obligatoires,
quels sont les noms/IDs exacts des priorites, et quel format attend le champ
"Plateforme". On interroge pour cela les "createmeta" de Jira.

On procede en 2 temps :
  1. Lister les types d'issue du projet -> recuperer l'ID du type "Bug".
  2. Lister les champs disponibles pour ce type -> obligatoires, priorites,
     champ Plateforme, etc.

Usage :
    python inspect_jira_meta.py
"""

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


def main() -> None:
    base_url = get_required("JIRA_BASE_URL").rstrip("/")
    email = get_required("JIRA_EMAIL")
    token = get_required("JIRA_API_TOKEN")
    projet = get_required("JIRA_PROJECT_KEY")
    champ_plateforme = get_required("JIRA_PLATFORM_FIELD_ID")

    session = requests.Session()
    session.auth = (email, token)
    session.headers.update({"Accept": "application/json"})

    print("=" * 70)
    print(f"Metadonnees de creation pour le projet {projet}")
    print("=" * 70)

    # --- 1. Trouver l'ID du type d'issue "Bug" -----------------------------
    url_types = f"{base_url}/rest/api/3/issue/createmeta/{projet}/issuetypes"
    rep = session.get(url_types, timeout=15)
    if rep.status_code != 200:
        print(f"  [X] Echec lecture des types d'issue (HTTP {rep.status_code}) :")
        print(f"      {rep.text[:300]}")
        sys.exit(1)

    data = rep.json()
    # Selon la version/instance Jira, les types sont sous "values" OU "issueTypes".
    # On gere les deux pour ne pas dependre du format exact de la reponse.
    types = data.get("values") or data.get("issueTypes") or []
    print(f"\n  Types d'issue disponibles ({len(types)} trouve(s)) :")
    id_bug = None
    for t in types:
        marque = ""
        if t.get("name", "").lower() == "bug":
            id_bug = t.get("id")
            marque = "   <- on utilisera celui-ci"
        print(f"    id={t.get('id'):<8} {t.get('name')}{marque}")

    if not types:
        # DIAGNOSTIC : aucun type renvoye. On affiche la reponse brute de Jira
        # (clefs + total) pour comprendre, puis on teste la permission de creation.
        print("\n  [DIAGNOSTIC] Reponse brute de Jira (clefs) :", list(data.keys()))
        print(f"               total declare par Jira : {data.get('total')}")

        print("\n  [DIAGNOSTIC] Test de ta permission 'Creer des tickets' sur BUGOTTO...")
        url_perm = f"{base_url}/rest/api/3/mypermissions"
        rep_perm = session.get(url_perm, params={
            "projectKey": projet,
            "permissions": "CREATE_ISSUES",
        }, timeout=15)
        if rep_perm.status_code == 200:
            perms = rep_perm.json().get("permissions", {})
            ci = perms.get("CREATE_ISSUES", {})
            autorise = ci.get("havePermission")
            print(f"               CREATE_ISSUES havePermission = {autorise}")
            if autorise is False:
                print("               -> Ton compte ne peut PAS creer d'issue ici via l'API.")
                print("                  (Tu crees peut-etre tes Jira via un formulaire/portail.)")
        else:
            print(f"               (test permission : HTTP {rep_perm.status_code})")

        print("\n  [DIAGNOSTIC] Verification du type de projet (JSM ?)...")
        rep_proj = session.get(f"{base_url}/rest/api/3/project/{projet}", timeout=15)
        if rep_proj.status_code == 200:
            p = rep_proj.json()
            print(f"               style       : {p.get('style')}      (classic ou next-gen)")
            print(f"               projectTypeKey : {p.get('projectTypeKey')}   "
                  "(software / service_desk / business)")
        print("\n  Colle-moi tout ce bloc DIAGNOSTIC, on en deduira la marche a suivre.")
        sys.exit(1)

    if not id_bug:
        noms = ", ".join(t.get("name", "?") for t in types)
        print(f"\n  [!] Pas de type nomme exactement 'Bug'. Types dispo : {noms}")
        print("      Dis-moi lequel utiliser, je l'adapterai.")
        sys.exit(1)

    # --- 2. Lister les champs pour le type "Bug" ---------------------------
    url_champs = f"{base_url}/rest/api/3/issue/createmeta/{projet}/issuetypes/{id_bug}"
    rep = session.get(url_champs, timeout=15)
    if rep.status_code != 200:
        print(f"  [X] Echec lecture des champs (HTTP {rep.status_code}).")
        sys.exit(1)

    # Meme prudence que pour les types : champs sous "values", "fields"... selon l'instance.
    champs_data = rep.json()
    champs = champs_data.get("values") or champs_data.get("fields") or []

    # Normalisation : si Jira renvoie un DICT {fieldId: {...}} au lieu d'une LISTE,
    # on le convertit en liste en injectant l'identifiant manquant.
    if isinstance(champs, dict):
        champs = [{**v, "fieldId": v.get("fieldId", cle)} for cle, v in champs.items()]

    # 2a. Champs OBLIGATOIRES : ceux qu'on DOIT fournir, sinon la creation echoue.
    print("\n  -- Champs OBLIGATOIRES a la creation --")
    for c in champs:
        if c.get("required"):
            print(f"    {c.get('fieldId'):<22} {c.get('name')}")

    # 2b. Valeurs autorisees pour la PRIORITE (nos noms custom Mineur/Bloquant...).
    print("\n  -- Priorites acceptees (nom + id) --")
    for c in champs:
        if c.get("fieldId") == "priority":
            for v in c.get("allowedValues", []):
                print(f"    id={v.get('id'):<6} {v.get('name')}")

    # 2c. Le champ PLATEFORME : type, et valeurs autorisees s'il s'agit d'une liste.
    print(f"\n  -- Champ Plateforme ({champ_plateforme}) --")
    trouve = False
    for c in champs:
        if c.get("fieldId") == champ_plateforme:
            trouve = True
            schema = c.get("schema", {})
            print(f"    Nom   : {c.get('name')}")
            print(f"    Type  : {schema.get('type')} / {schema.get('custom','').split(':')[-1]}")
            valeurs = c.get("allowedValues", [])
            if valeurs:
                print(f"    -> liste a choix : on devra envoyer une des valeurs ci-dessous")
                print(f"       (cherche 'studio17' ou son equivalent) :")
                for v in valeurs:
                    # selon le type, le libelle est sous 'value' ou 'name'
                    libelle = v.get("value") or v.get("name") or v.get("id")
                    print(f"         id={v.get('id'):<8} {libelle}")
            else:
                print("    -> champ libre (pas de liste imposee) : on enverra du texte")
    if not trouve:
        print("    [!] Ce champ n'est pas propose a la creation pour le type Bug.")
        print("        (Il faudra peut-etre le remplir autrement, ou il a un autre id.)")

    print("\n" + "=" * 70)
    print("Lecture terminee. On s'en sert pour construire le brouillon (etape 3).")
    print("=" * 70)


if __name__ == "__main__":
    main()
