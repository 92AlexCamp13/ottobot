"""
transfer_attachments.py — Transfere les pieces jointes d'un ticket Zendesk vers une issue Jira.

ETAPE 6b (test autonome). Sert a valider le transfert de fichiers SANS recreer de
ticket Jira : on cible une issue deja existante (ex. celle creee a l'etape 6a).
Une fois la mecanique prouvee, elle sera appelee automatiquement par create_jira.py.

Usage :
    python transfer_attachments.py BUGOTTO-1297 4927
       (1er argument = cle Jira cible, 2e = id du ticket Zendesk source)
"""

import argparse

import noyau


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfere les PJ Zendesk -> Jira.")
    parser.add_argument("cle_jira", help="Cle de l'issue Jira cible (ex. BUGOTTO-1297)")
    parser.add_argument("ticket_id", help="ID du ticket Zendesk source")
    args = parser.parse_args()

    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = noyau.session_jira()

    print("=" * 70)
    print(f"Transfert des pieces jointes : Zendesk #{args.ticket_id} -> {args.cle_jira}")
    print("=" * 70)

    nb = noyau.transferer_pieces_jointes(zd, base_zd, jira, base_jira,
                                         args.ticket_id, args.cle_jira)

    print("\n" + "=" * 70)
    print(f"Termine : {nb} fichier(s) transfere(s).")
    print(f"Verifie sur {base_jira}/browse/{args.cle_jira}")
    print("=" * 70)


if __name__ == "__main__":
    main()
