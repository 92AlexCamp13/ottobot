#!/bin/bash
# demarrer_serveur.sh — Lancement de l'app en PRODUCTION (Mac mini).
#
# Difference avec lancer.command (usage local) : pas de navigateur ouvert,
# pas de --reload, et surtout l'app ecoute UNIQUEMENT sur 127.0.0.1.
# L'exposition reseau (HTTPS) est faite par Tailscale (`tailscale serve`), qui
# proxifie vers 127.0.0.1:8000 -> l'app n'est JAMAIS joignable directement
# depuis le LAN ou Internet. Defense en profondeur : meme si Tailscale tombe,
# rien n'est expose.
#
# Les secrets (cle Anthropic, APP_SECRET_KEY, TOKEN_ENCRYPTION_KEY, identifiants
# Zendesk...) viennent du fichier .env, lu cote serveur uniquement (jamais envoye
# au navigateur).

cd "$(dirname "$0")" || exit 1
exec .venv/bin/uvicorn web:app \
  --host "${APP_HOST:-127.0.0.1}" \
  --port "${APP_PORT:-8000}"
