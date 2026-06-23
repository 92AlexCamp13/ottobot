"""
review_ticket.py — Revue interactive d'un ticket, un par un (ETAPE 5).

On lit un ticket Zendesk, on construit le brouillon auto-deduit (via noyau.py),
puis on TE DEMANDE les infos qui ne peuvent pas l'etre (priorite, Android/iOS ou
onglet BO + BO/FO, confirmation de l'assigne et de la plateforme). On assemble
enfin le "fields" Jira complet et on l'affiche pour relecture.

IMPORTANT : a ce stade, on NE CREE TOUJOURS RIEN dans Jira. Cette etape produit
le ticket "pret a creer" ; la creation reelle (avec pieces jointes) sera l'etape 6.

Usage :
    python review_ticket.py 4927
"""

import argparse
import re
import sys

import noyau


# ============================================================================
# PETITS OUTILS DE SAISIE (robustes : ils insistent jusqu'a une reponse valide)
# ============================================================================

def demander_choix(question: str, options: list) -> str:
    """Affiche un menu numerote et renvoie l'option choisie."""
    print(f"\n  {question}")
    for i, opt in enumerate(options, start=1):
        print(f"    [{i}] {opt}")
    while True:
        saisie = input("  > Ton choix (numero) : ").strip()
        # On extrait le premier nombre de la saisie : tolere "1", "[1]", " 1 "...
        nombres = re.findall(r"\d+", saisie)
        if nombres and 1 <= int(nombres[0]) <= len(options):
            return options[int(nombres[0]) - 1]
        print("    (Reponse invalide : tape le numero d'une option.)")


def demander_texte(question: str) -> str:
    """Demande un texte non vide."""
    while True:
        saisie = input(f"  > {question} : ").strip()
        if saisie:
            return saisie
        print("    (Ne peut pas etre vide.)")


def choisir_plateforme(options: list) -> dict:
    """Cas ou l'auto-match a echoue : on cherche par mot-cle puis on choisit."""
    print("\n  Aucune plateforme trouvee automatiquement. Cherchons-la.")
    while True:
        terme = noyau.normaliser(demander_texte("Tape une partie du nom du client"))
        trouves = [o for o in options
                   if terme in noyau.normaliser(o.get("value") or o.get("name") or "")]
        if not trouves:
            print("    (Aucune option ne correspond, reessaie.)")
            continue
        libelles = [o.get("value") or o.get("name") for o in trouves[:15]]
        choix = demander_choix("Plateformes correspondantes :", libelles)
        opt = next(o for o in trouves if (o.get("value") or o.get("name")) == choix)
        return {"id": opt.get("id"), "label": choix}


# ============================================================================
# REVUE D'UN TICKET -> renvoie le "fields" Jira pret a creer
# ============================================================================

def revue_interactive(ticket: dict, infos: dict, match, options: list) -> dict:
    """Pose les questions manquantes et assemble le 'fields' Jira complet."""

    # --- 1. Resume (nomenclature) ------------------------------------------
    if infos["type"] in ("App Mobile", "App TV"):
        systeme = demander_choix("Systeme concerne ?", ["Android", "iOS"])
        resume = infos["gabarit_resume"].format(os=systeme, titre=ticket["subject"])
    else:  # Web
        onglet = demander_texte("Onglet du BO concerne (ex. Catalogue)")
        bofo = demander_choix("Back-office ou Front-office ?", ["BO", "FO"])
        resume = infos["gabarit_resume"].format(onglet=onglet, bofo=bofo, titre=ticket["subject"])

    # --- 2. Priorite + echeance derivee ------------------------------------
    nom_prio = demander_choix("Priorite ?", list(noyau.PRIORITES))
    echeance = noyau.calculer_echeance(nom_prio)

    # --- 3. Assigne (confirmer la suggestion ou basculer) ------------------
    print(f"\n  Assigne suggere (via tags) : {infos['assigne_label']}")
    if demander_choix("On garde cet assigne ?", ["Oui", "Non, choisir l'autre"]) == "Oui":
        assigne_id, assigne_label = infos["assigne_id"], infos["assigne_label"]
    else:
        # Bascule vers l'autre destinataire connu.
        if infos["assigne_label"] == "Tech VODF":
            assigne_id = noyau.get_required("JIRA_ASSIGNEE_SOUFIANE")
            assigne_label = "EL AMRANI Soufiane"
        else:
            assigne_id = noyau.get_required("JIRA_ASSIGNEE_TECH_VODF")
            assigne_label = "Tech VODF"

    # --- 4. Plateforme (confirmer le match ou choisir) ---------------------
    if match:
        print(f"\n  Plateforme trouvee : {match['label']}  (depuis '{ticket['plateforme_zendesk']}')")
        if demander_choix("On garde cette plateforme ?", ["Oui", "Non, choisir"]) == "Non, choisir":
            match = choisir_plateforme(options)
    else:
        match = choisir_plateforme(options)

    # --- 5. Description -> ADF ---------------------------------------------
    zd_base = noyau.session_zendesk()[1]
    url_source = noyau.url_ticket_zendesk(zd_base, ticket["id"])
    description_adf = noyau.texte_vers_adf(ticket["description"], url_source)

    # --- 6. Assemblage du "fields" Jira ------------------------------------
    # C'est exactement la structure que l'API Jira attend dans {"fields": {...}}.
    # NB : on NE met PAS de champ "reporter". Du coup Jira affecte le rapporteur
    # (= demandeur) au compte authentifie, c'est-a-dire TOI. Le demandeur Zendesk
    # d'origine n'est donc jamais utilise comme demandeur Jira.
    fields = {
        "project": {"key": noyau.get_required("JIRA_PROJECT_KEY")},
        "issuetype": {"id": noyau.ISSUE_TYPE_BUG_ID},
        "summary": resume,
        "description": description_adf,
        "priority": {"id": noyau.PRIORITES[nom_prio]},
        "duedate": echeance,
        "assignee": {"accountId": assigne_id},
        noyau.get_required("JIRA_PLATFORM_FIELD_ID"): [{"id": match["id"]}],
    }
    # On garde quelques libelles lisibles a part, juste pour l'affichage du recap.
    meta_affichage = {"priorite": nom_prio, "assigne": assigne_label, "plateforme": match["label"]}
    return fields, meta_affichage


def afficher_recap(fields: dict, meta: dict) -> None:
    print("\n" + "=" * 70)
    print("RECAPITULATIF DU TICKET JIRA A CREER")
    print("=" * 70)
    print(f"  Projet      : {fields['project']['key']}")
    print(f"  Type        : Bug (id {fields['issuetype']['id']})")
    print(f"  Resume      : {fields['summary']}")
    if meta.get("rapporteur"):
        print(f"  Demandeur   : {meta['rapporteur']} (ton compte Jira)")
    print(f"  Priorite    : {meta['priorite']} (id {fields['priority']['id']})")
    print(f"  Echeance    : {fields['duedate']}")
    print(f"  Assigne     : {meta['assigne']}")
    print(f"  Plateforme  : {meta['plateforme']}")
    nb_blocs = len(fields["description"]["content"])
    print(f"  Description : {nb_blocs} blocs ADF (lien Zendesk source inclus)")
    print("=" * 70)


# ============================================================================
# POINT D'ENTREE
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Revue interactive d'un ticket Zendesk.")
    parser.add_argument("ticket_id", help="ID du ticket Zendesk")
    args = parser.parse_args()

    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = noyau.session_jira()

    ticket = noyau.charger_ticket(zd, base_zd, args.ticket_id)

    print("=" * 70)
    print(f"Revue du ticket Zendesk #{ticket['id']} : {ticket['subject']}")
    print("=" * 70)

    # Garde-fou anti-doublon.
    if ticket["cle_jira_existante"]:
        print(f"\n  /!\\ Ce ticket porte deja la cle Jira {ticket['cle_jira_existante']!r}.")
        print("      En conditions reelles on le sauterait. (Demo : on continue.)")

    infos = noyau.detecter_type_et_assigne(ticket["tags"])
    options = noyau.charger_options_plateforme(jira, base_jira)
    match = noyau.matcher_plateforme(ticket["plateforme_zendesk"], options)

    compte = noyau.compte_jira_courant(jira, base_jira)
    print(f"\n  Type detecte : {infos['type']}")
    print(f"  Signale cote Zendesk par : {ticket['demandeur']}  (info, PAS le demandeur Jira)")
    print(f"  Demandeur Jira (rapporteur) : {compte['nom']}  (ton compte)")

    fields, meta = revue_interactive(ticket, infos, match, options)
    meta["rapporteur"] = compte["nom"]
    afficher_recap(fields, meta)

    # Confirmation finale : pour l'instant elle ne fait que confirmer la relecture.
    # La creation reelle (POST Jira + pieces jointes) sera branchee a l'etape 6.
    print()
    if demander_choix("Ce brouillon est-il correct ?", ["Oui, pret a creer", "Non, abandonner"]) \
            == "Oui, pret a creer":
        print("\n  [OK] Brouillon valide. (Creation non encore implementee -> etape 6.)")
    else:
        print("\n  Abandonne. Rien n'a ete cree.")
        sys.exit(0)


if __name__ == "__main__":
    main()
