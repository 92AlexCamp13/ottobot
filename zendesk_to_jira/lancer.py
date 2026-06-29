"""
lancer.py — Demarrage tout-en-un de l'interface web (brief §9, etape 7).

Objectif : ne plus toucher au terminal. Ce script :
  1. programme l'ouverture de ton navigateur sur l'app (apres un petit delai,
     le temps que le serveur soit pret) ;
  2. demarre le serveur web (uvicorn) sur 127.0.0.1.

Tu peux le lancer de deux facons :
  - en double-cliquant sur 'lancer.command' (le plus simple, voir ce fichier) ;
  - ou en ligne de commande :  .venv/bin/python lancer.py

Pour ARRETER : ferme la fenetre du terminal, ou fais Ctrl+C.

Note : ici PAS de '--reload' (c'est un lancement d'usage, pas de developpement).
Le serveur n'ecoute que sur 127.0.0.1 (ta machine) : inaccessible du reseau (brief §5).
"""

import os
import threading
import webbrowser

import uvicorn

# Hote toujours local (regle de securite §5). Port : 8000 par defaut, surchargeable
# via une variable WEB_PORT dans le .env si jamais 8000 est deja pris.
HOTE = "127.0.0.1"
PORT = int(os.getenv("WEB_PORT", "8000"))
URL = f"http://{HOTE}:{PORT}"


def ouvrir_navigateur() -> None:
    """Ouvre l'app dans le navigateur par defaut."""
    webbrowser.open(URL)


if __name__ == "__main__":
    print(f"Demarrage de l'outil Zendesk -> Jira sur {URL}")
    print("Ton navigateur va s'ouvrir tout seul dans un instant.")
    print("Pour arreter : ferme cette fenetre ou fais Ctrl+C.\n")

    # On ouvre le navigateur ~1,5 s apres, pour laisser au serveur le temps de
    # demarrer. threading.Timer execute la fonction en arriere-plan sans bloquer
    # le demarrage du serveur juste en dessous.
    threading.Timer(1.5, ouvrir_navigateur).start()

    # Demarre le serveur (appel bloquant : il tourne jusqu'a l'arret).
    uvicorn.run("web:app", host=HOTE, port=PORT)
