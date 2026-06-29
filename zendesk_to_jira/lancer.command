#!/bin/bash
# lancer.command — Double-clique ce fichier dans le Finder pour demarrer l'outil.
#
# Un fichier .command s'ouvre dans le Terminal et execute son contenu. On se
# place d'abord dans le dossier du script (pour que 'web:app' et le .venv soient
# trouvables, quel que soit l'endroit d'ou on double-clique), puis on lance le
# lanceur Python via l'interpreteur du venv (pas besoin d'activer le venv).
#
# Pour arreter le serveur : ferme cette fenetre du Terminal, ou fais Ctrl+C.

cd "$(dirname "$0")" || exit 1
exec .venv/bin/python lancer.py
