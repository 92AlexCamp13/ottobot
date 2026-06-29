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
import sys

import requests

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


def creer_issue(jira, base: str, fields: dict) -> dict | None:
    """POST de creation a Jira. Renvoie {"cle": ..., "id": ...} cree, ou None.

    On renvoie AUSSI l'id numerique de l'issue (pas seulement la cle) car le
    connecteur Jira de Zendesk en a besoin pour creer le lien natif (lier_zendesk_jira).
    On NE coupe PAS le programme en cas d'erreur (pas de sys.exit) : en mode lot,
    un ticket fautif ne doit pas stopper le traitement des suivants.
    """
    print("\n  Creation en cours dans Jira...")
    rep = jira.post(f"{base}/rest/api/3/issue", json={"fields": fields}, timeout=30)

    if rep.status_code == 201:
        data = rep.json()
        cle = data.get("key")
        print(f"  [OK] Ticket cree : {cle}  ({base}/browse/{cle})")
        return {"cle": cle, "id": data.get("id")}

    print(f"  [X] Echec de creation (HTTP {rep.status_code}). Reponse Jira :")
    print(f"      {rep.text}")
    return None


def creer_depuis_fields(zd, base_zd, jira, base_jira, ticket: dict, fields: dict) -> dict:
    """Cree REELLEMENT le Jira a partir d'un 'fields' deja assemble, SANS interaction.

    C'est la sequence d'ecriture, isolee pour etre partagee (brief §4) :
      1. creation de l'issue,
      2. transfert des pieces jointes Zendesk -> Jira,
      3. reecriture cote Zendesk (champ 'Cle Jira liee' + retrait du tag),
      4. journalisation.

    Elle ne pose AUCUNE question : la validation humaine a deja eu lieu en amont
    (au clavier pour le CLI, par la soumission du formulaire pour le web). Renvoie
    {"cle": <str|None>, "nb_pj": <int>} ; 'cle' = None si la creation a echoue.
    """
    # 1. Creation.
    creation = creer_issue(jira, base_jira, fields)
    if not creation:
        noyau.journaliser(ticket["id"], "echec_creation", detail="voir reponse Jira ci-dessus")
        return {"cle": None, "nb_pj": 0, "writeback_ok": False}  # ni PJ ni reecriture.
    cle = creation["cle"]

    # 2. Reecriture cote Zendesk D'ABORD (champ 'Cle Jira liee' + tag).
    #    On la fait AVANT les pieces jointes : c'est l'ancre anti-doublon. Une fois
    #    le champ ecrit, meme si la suite echoue, un re-run sautera ce ticket au
    #    lieu d'en recreer un. On lit le retour pour journaliser honnetement.
    print("\n  Reecriture cote Zendesk...")
    writeback_ok = noyau.ecrire_retour_zendesk(zd, base_zd, ticket["id"], cle)

    # 3. Lien NATIF via le connecteur Jira de Zendesk (bidirectionnel, API legacy).
    #    Additif et non bloquant. Le champ 'Cle Jira liee' (ci-dessus) reste l'ancre
    #    anti-doublon ; ce lien-ci, c'est le confort d'affichage des deux cotes.
    print("\n  Lien via le connecteur Jira de Zendesk...")
    noyau.lier_zendesk_jira(zd, base_zd, ticket["id"], creation.get("id"), cle)

    # 4. Pieces jointes. Un echec reseau ici ne doit plus rien casser (le Jira
    #    existe, l'ancre anti-doublon est posee) : on isole et on continue.
    print("\n  Transfert des pieces jointes...")
    try:
        nb_pj = noyau.transferer_pieces_jointes(zd=zd, base_zd=base_zd, jira=jira, base_jira=base_jira,
                                                ticket_id=ticket["id"], cle_jira=cle)
    except requests.RequestException as erreur:
        print(f"  [!] Transfert des pieces jointes interrompu (reseau) : {erreur}")
        nb_pj = 0

    # 5. Trace dans le journal — statut HONNETE selon le writeback.
    #    'cree_writeback_ko' = le Jira est cree mais le champ Zendesk n'a PAS pu
    #    etre ecrit : ORPHELIN a corriger a la main (un re-run recreerait un doublon).
    statut = "cree" if writeback_ok else "cree_writeback_ko"
    noyau.journaliser(ticket["id"], statut, cle, detail=f"{nb_pj} piece(s) jointe(s)")
    return {"cle": cle, "nb_pj": nb_pj, "writeback_ok": writeback_ok}


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

    # Creation reelle (creation + PJ + reecriture + journal), via la fonction
    # partagee avec le web. C'est ici que l'ecriture devient effective.
    res = creer_depuis_fields(zd, base_zd, jira, base_jira, ticket, fields)
    return res["cle"]


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
    # On rattrape les erreurs previsibles du moteur (config, ticket introuvable,
    # API KO) pour afficher un message propre et sortir en code 1, sans trace brute.
    try:
        main()
    except noyau.ErreurOutil as erreur:
        print(f"  [X] {erreur}")
        sys.exit(1)
