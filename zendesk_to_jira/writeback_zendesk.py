"""
writeback_zendesk.py — Ecrit la cle Jira dans le ticket Zendesk + retire 'to-jira' (ETAPE 7).

Test autonome de la reecriture cote Zendesk, sur un ticket et une cle Jira donnes.
Une fois valide, cet appel sera ajoute a create_jira.py (creation -> PJ -> reecriture).

ATTENTION : ce script ECRIT dans Zendesk (renseigne le champ 'Cle Jira liee' et
retire le tag 'to-jira' s'il est present).

Usage :
    python writeback_zendesk.py 4927 BUGOTTO-1297
       (1er argument = id ticket Zendesk, 2e = cle Jira a inscrire)
"""

import argparse

import noyau


def main() -> None:
    parser = argparse.ArgumentParser(description="Reecrit la cle Jira dans le ticket Zendesk.")
    parser.add_argument("ticket_id", help="ID du ticket Zendesk")
    parser.add_argument("cle_jira", help="Cle Jira a inscrire (ex. BUGOTTO-1297)")
    args = parser.parse_args()

    zd, base_zd = noyau.session_zendesk()

    print("=" * 70)
    print(f"Reecriture Zendesk : ticket #{args.ticket_id} <- {args.cle_jira}")
    print("=" * 70 + "\n")

    noyau.ecrire_retour_zendesk(zd, base_zd, args.ticket_id, args.cle_jira)

    print("\n" + "=" * 70)
    print(f"Termine. Verifie avec : python fetch_ticket.py {args.ticket_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
