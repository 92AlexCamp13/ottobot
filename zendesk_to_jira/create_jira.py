"""
create_jira.py — Pipeline COMPLET pour un ticket : revue -> creation -> PJ -> reecriture.

ETAPE 6+7 reunies. Si tu confirmes, l'outil :
  1. cree le ticket Jira (etape 6a),
  2. transfere les pieces jointes Zendesk -> Jira (etape 6b),
  3. reecrit la cle Jira dans le ticket Zendesk + retire le tag 'to-jira' (etape 7).

La logique "traiter UN ticket" est isolee dans traiter_un_ticket() : ainsi la
couche de selection par lot (run_batch.py, etape 8) la reutilise telle quelle.
C'est le principe du brief §5 : un coeur unique, plusieurs facons de selectionner.

Usage (un ticket par son ID) :
    python create_jira.py 4927
"""

import argparse

import noyau
import review_ticket as revue   # revue interactive de l'etape 5


def apercu_description(adf: dict) -> None:
    """Affiche le contenu ADF en texte lisible, pour relecture avant envoi."""
    print("\n  -- DESCRIPTION qui sera envoyee a Jira --")
    for bloc in adf.get("content", []):
        if bloc["type"] == "paragraph":
            texte = "".join(n.get("text", "") for n in bloc.get("content", []))
            print(f"    | {texte}")
        elif bloc["type"] == "rule":
            print(f"    | {'-' * 50}")


def creer_issue(jira, base: str, fields: dict) -> str | None:
    """POST de creation a Jira. Renvoie la cle creee, ou None en cas d'echec.

    On NE coupe PAS le programme en cas d'erreur (pas de sys.exit) : en mode lot,
    un ticket fautif ne doit pas stopper le traitement des suivants.
    """
    print("\n  Creation en cours dans Jira...")
    rep = jira.post(f"{base}/rest/api/3/issue", json={"fields": fields}, timeout=30)

    if rep.status_code == 201:
        cle = rep.json().get("key")
        print(f"  [OK] Ticket cree : {cle}  ({base}/browse/{cle})")
        return cle

    print(f"  [X] Echec de creation (HTTP {rep.status_code}). Reponse Jira :")
    print(f"      {rep.text}")
    return None


def traiter_un_ticket(zd, base_zd, jira, base_jira, ticket: dict, options: list) -> str | None:
    """Pipeline complet pour UN ticket deja charge.

    Renvoie la cle Jira creee, ou None si le ticket est saute (deja converti,
    abandonne par toi, ou creation en echec). 'options' = liste des plateformes
    Jira, passee de l'exterieur pour ne la charger qu'une fois (utile en mode lot).
    """
    print("\n" + "=" * 70)
    print(f"Ticket Zendesk #{ticket['id']} : {ticket['subject']}")
    print("=" * 70)

    # Garde-fou anti-doublon (brief §3.3).
    if ticket["cle_jira_existante"]:
        print(f"  -> Deja converti ({ticket['cle_jira_existante']!r}). On saute.")
        noyau.journaliser(ticket["id"], "deja_converti", ticket["cle_jira_existante"])
        return None

    infos = noyau.detecter_type_et_assigne(ticket["tags"])
    match = noyau.matcher_plateforme(ticket["plateforme_zendesk"], options)
    compte = noyau.compte_jira_courant(jira, base_jira)

    print(f"\n  Type detecte : {infos['type']}")
    print(f"  Signale cote Zendesk par : {ticket['demandeur']}  (info, PAS le demandeur Jira)")
    print(f"  Demandeur Jira (rapporteur) : {compte['nom']}  (ton compte)")

    # Revue interactive (etape 5) -> 'fields' complet.
    fields, meta = revue.revue_interactive(ticket, infos, match, options)
    meta["rapporteur"] = compte["nom"]
    revue.afficher_recap(fields, meta)
    apercu_description(fields["description"])

    # Confirmation : c'est ici que l'ecriture devient reelle.
    print()
    if revue.demander_choix("Creer reellement ce ticket dans Jira ?",
                            ["Oui, creer maintenant", "Non, passer ce ticket"]) \
            != "Oui, creer maintenant":
        print("  Ticket passe. Rien cree.")
        noyau.journaliser(ticket["id"], "abandonne")
        return None

    # 1. Creation.
    cle = creer_issue(jira, base_jira, fields)
    if not cle:
        noyau.journaliser(ticket["id"], "echec_creation", detail="voir reponse Jira ci-dessus")
        return None  # echec : on n'enchaine ni PJ ni reecriture.

    # 2. Pieces jointes.
    print("\n  Transfert des pieces jointes...")
    nb_pj = noyau.transferer_pieces_jointes(zd=zd, base_zd=base_zd, jira=jira, base_jira=base_jira,
                                            ticket_id=ticket["id"], cle_jira=cle)

    # 3. Reecriture cote Zendesk (champ + tag).
    print("\n  Reecriture cote Zendesk...")
    noyau.ecrire_retour_zendesk(zd, base_zd, ticket["id"], cle)

    # 4. Trace dans le journal.
    noyau.journaliser(ticket["id"], "cree", cle, detail=f"{nb_pj} piece(s) jointe(s)")
    return cle


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit UN ticket Zendesk en ticket Jira.")
    parser.add_argument("ticket_id", help="ID du ticket Zendesk")
    args = parser.parse_args()

    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = noyau.session_jira()

    # On charge les options plateforme une fois, puis on traite le ticket.
    options = noyau.charger_options_plateforme(jira, base_jira)
    ticket = noyau.charger_ticket(zd, base_zd, args.ticket_id)

    cle = traiter_un_ticket(zd, base_zd, jira, base_jira, ticket, options)

    print("\n" + "=" * 70)
    print(f"Termine. {'Ticket cree : ' + cle if cle else 'Aucun ticket cree.'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
