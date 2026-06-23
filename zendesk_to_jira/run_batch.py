"""
run_batch.py — Traite EN LOT tous les tickets tagges 'to-jira' (ETAPE 8).

C'est la couche de SELECTION du brief (§5) : au lieu de donner un ID a la main,
on demande a Zendesk tous les tickets portant le tag declencheur, et on passe
chacun dans le MEME coeur (create_jira.traiter_un_ticket). Tu valides toujours
chaque ticket un par un avant creation.

L'anti-doublon est double :
  - on ne ramasse que les tickets tagges 'to-jira' ;
  - traiter_un_ticket saute ceux dont le champ 'Cle Jira liee' est deja rempli.

Usage :
    python run_batch.py
"""

import noyau
import create_jira


def chercher_tickets_a_traiter(zd, base: str) -> list:
    """Renvoie les IDs des tickets portant le tag declencheur.

    On utilise l'API de recherche Zendesk avec la requete 'type:ticket tags:to-jira'.
    On suit la pagination ('next_page') pour ne rien manquer si la liste est longue.
    """
    ids = []
    url = f"{base}/search.json"
    params = {"query": f"type:ticket tags:{noyau.TAG_DECLENCHEUR}"}

    while url:
        rep = zd.get(url, params=params, timeout=20)
        if rep.status_code != 200:
            print(f"  [X] Recherche Zendesk echouee (HTTP {rep.status_code}).")
            break
        data = rep.json()
        for resultat in data.get("results", []):
            if resultat.get("id"):
                ids.append(resultat["id"])
        # 'next_page' est une URL complete (ou None). Les params sont deja dedans.
        url, params = data.get("next_page"), None
    return ids


def main() -> None:
    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = noyau.session_jira()

    print("=" * 70)
    print(f"Traitement par lot des tickets tagges '{noyau.TAG_DECLENCHEUR}'")
    print("=" * 70)

    ids = chercher_tickets_a_traiter(zd, base_zd)
    if not ids:
        print(f"\n  Aucun ticket tagge '{noyau.TAG_DECLENCHEUR}'. Rien a faire.")
        return

    print(f"\n  {len(ids)} ticket(s) trouve(s) : {ids}")

    # Options plateforme chargees UNE seule fois pour tout le lot (gain d'appels).
    options = noyau.charger_options_plateforme(jira, base_jira)

    # On traite chaque ticket, en gardant une trace pour le bilan final.
    crees, sautes = [], []
    for ticket_id in ids:
        # Isolation : une erreur sur UN ticket (reseau, donnees inattendues...) ne
        # doit pas stopper tout le lot. On la logue et on passe au suivant.
        try:
            ticket = noyau.charger_ticket(zd, base_zd, ticket_id)
            cle = create_jira.traiter_un_ticket(zd, base_zd, jira, base_jira, ticket, options)
        except Exception as erreur:
            print(f"\n  [X] Erreur inattendue sur #{ticket_id} : {erreur}")
            print("      On continue avec les tickets suivants.")
            noyau.journaliser(ticket_id, "erreur", detail=str(erreur)[:200])
            sautes.append(ticket_id)
            continue

        if cle:
            crees.append((ticket_id, cle))
        else:
            sautes.append(ticket_id)

    # --- Bilan lisible -----------------------------------------------------
    print("\n" + "=" * 70)
    print("BILAN DU LOT")
    print("=" * 70)
    print(f"  Crees   ({len(crees)}) :")
    for tid, cle in crees:
        print(f"    - Zendesk #{tid} -> {cle}")
    print(f"  Sautes / abandonnes ({len(sautes)}) : {sautes or '(aucun)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
