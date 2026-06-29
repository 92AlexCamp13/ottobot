"""
comptes.py — Stockage des comptes utilisateurs (v3 multi-utilisateur).

Petite base SQLite (lib standard, zero dependance reseau) qui retient, pour
chaque membre de l'equipe :
  - son identifiant + mot de passe (HACHE avec bcrypt, jamais en clair) ;
  - son token API Jira (CHIFFRE au repos avec Fernet) + son email Jira ;
  - son accountId / nom Jira (recuperes via /myself a la saisie du token) ;
  - s'il est admin (gere les comptes via la future page /admin).

Securite :
  - mots de passe : bcrypt (sale + lent -> resistant au brute force) ;
  - tokens Jira : chiffres avec une cle Fernet (TOKEN_ENCRYPTION_KEY, hors base) ;
  - le fichier .db lui-meme est gitignore.

Ce module ne fait QUE du stockage. La validation du token aupres de Jira
(/myself) et les routes web (login, /setup, /admin, /profil) viennent aux
etapes suivantes. Il leve noyau.ErreurOutil pour les erreurs previsibles
(deja attrapee proprement par le CLI et par le web).
"""

import os
import sqlite3
from contextlib import closing
from datetime import datetime

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

import noyau  # pour ErreurOutil (gestion d'erreur deja en place)


# Chemin de la base : configurable par APP_DB_PATH (utile en prod sur Railway,
# ou il faut pointer vers un VOLUME persistant). Par defaut : a cote du code.
CHEMIN_DB = os.getenv("APP_DB_PATH", "").strip() or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "comptes.db")


def _connexion() -> sqlite3.Connection:
    con = sqlite3.connect(CHEMIN_DB)
    con.row_factory = sqlite3.Row   # acceder aux colonnes par nom
    return con


def init_db() -> None:
    """Cree la table des utilisateurs si elle n'existe pas (idempotent)."""
    with closing(_connexion()) as con, con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                identifiant         TEXT UNIQUE NOT NULL,
                mot_de_passe_hache  TEXT NOT NULL,
                est_admin           INTEGER NOT NULL DEFAULT 0,
                jira_email          TEXT,
                jira_token_chiffre  TEXT,
                jira_account_id     TEXT,
                jira_nom            TEXT,
                cree_le             TEXT NOT NULL
            )
        """)


def _fernet() -> Fernet:
    """Outil de (de)chiffrement des tokens. Leve ErreurOutil si la cle manque/est invalide."""
    cle = os.getenv("TOKEN_ENCRYPTION_KEY", "").strip()
    if not cle:
        raise noyau.ErreurOutil(
            "TOKEN_ENCRYPTION_KEY absente du .env : impossible de chiffrer les tokens Jira.")
    try:
        return Fernet(cle.encode())
    except (ValueError, TypeError) as erreur:
        raise noyau.ErreurOutil("TOKEN_ENCRYPTION_KEY invalide (format Fernet attendu).") from erreur


# ============================================================================
# MOTS DE PASSE (bcrypt)
# ============================================================================

def _hacher(mot_de_passe: str) -> str:
    return bcrypt.hashpw(mot_de_passe.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verifier_hash(mot_de_passe: str, hache: str) -> bool:
    try:
        return bcrypt.checkpw(mot_de_passe.encode("utf-8"), hache.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ============================================================================
# COMPTES (creation, verification, gestion)
# ============================================================================

def compter_utilisateurs() -> int:
    """Nombre de comptes. Sert au bootstrap : si 0 -> on affiche la page /setup."""
    with closing(_connexion()) as con:
        return con.execute("SELECT COUNT(*) FROM utilisateurs").fetchone()[0]


def creer_utilisateur(identifiant: str, mot_de_passe: str, est_admin: bool = False) -> int:
    """Cree un compte (mot de passe hache). Renvoie son id. Leve si l'identifiant existe."""
    identifiant = (identifiant or "").strip()
    if not identifiant or not mot_de_passe:
        raise noyau.ErreurOutil("Identifiant et mot de passe sont obligatoires.")
    try:
        with closing(_connexion()) as con, con:
            cur = con.execute(
                "INSERT INTO utilisateurs (identifiant, mot_de_passe_hache, est_admin, cree_le) "
                "VALUES (?, ?, ?, ?)",
                (identifiant, _hacher(mot_de_passe), 1 if est_admin else 0,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError as erreur:
        raise noyau.ErreurOutil(f"L'identifiant « {identifiant} » existe deja.") from erreur


def verifier_identifiants(identifiant: str, mot_de_passe: str) -> dict | None:
    """Renvoie le compte (dict) si identifiant + mot de passe corrects, sinon None."""
    with closing(_connexion()) as con:
        ligne = con.execute("SELECT * FROM utilisateurs WHERE identifiant = ?",
                            ((identifiant or "").strip(),)).fetchone()
    if ligne and _verifier_hash(mot_de_passe, ligne["mot_de_passe_hache"]):
        return dict(ligne)
    return None


def get_utilisateur(user_id: int) -> dict | None:
    with closing(_connexion()) as con:
        ligne = con.execute("SELECT * FROM utilisateurs WHERE id = ?", (user_id,)).fetchone()
    return dict(ligne) if ligne else None


def lister_utilisateurs() -> list:
    """Liste des comptes (SANS le hash ni le token chiffre), pour la page /admin."""
    with closing(_connexion()) as con:
        lignes = con.execute(
            "SELECT id, identifiant, est_admin, jira_email, jira_nom, jira_account_id, cree_le "
            "FROM utilisateurs ORDER BY identifiant").fetchall()
    return [dict(ligne) for ligne in lignes]


def reinitialiser_mot_de_passe(user_id: int, nouveau: str) -> None:
    if not nouveau:
        raise noyau.ErreurOutil("Le nouveau mot de passe ne peut pas etre vide.")
    with closing(_connexion()) as con, con:
        con.execute("UPDATE utilisateurs SET mot_de_passe_hache = ? WHERE id = ?",
                    (_hacher(nouveau), user_id))


def supprimer_utilisateur(user_id: int) -> None:
    with closing(_connexion()) as con, con:
        con.execute("DELETE FROM utilisateurs WHERE id = ?", (user_id,))


# ============================================================================
# TOKEN JIRA PAR UTILISATEUR (chiffre au repos)
# ============================================================================

def definir_token_jira(user_id: int, jira_email: str, jira_token: str,
                       jira_account_id: str, jira_nom: str) -> None:
    """Stocke (chiffre) le token Jira de l'utilisateur + son identite Jira."""
    jeton_chiffre = _fernet().encrypt(jira_token.encode("utf-8")).decode("utf-8")
    with closing(_connexion()) as con, con:
        con.execute(
            "UPDATE utilisateurs SET jira_email = ?, jira_token_chiffre = ?, "
            "jira_account_id = ?, jira_nom = ? WHERE id = ?",
            (jira_email, jeton_chiffre, jira_account_id, jira_nom, user_id),
        )


def lire_token_jira(user_id: int) -> tuple[str, str] | None:
    """Renvoie (jira_email, jira_token EN CLAIR) de l'utilisateur, ou None si non configure.

    Dechiffre le token a la volee. Leve ErreurOutil si le token est illisible
    (typiquement : la cle TOKEN_ENCRYPTION_KEY a change) -> l'utilisateur devra
    re-saisir son token dans son profil.
    """
    u = get_utilisateur(user_id)
    if not u or not u.get("jira_token_chiffre"):
        return None
    try:
        token = _fernet().decrypt(u["jira_token_chiffre"].encode("utf-8")).decode("utf-8")
    except InvalidToken as erreur:
        raise noyau.ErreurOutil(
            "Token Jira illisible (la cle de chiffrement a-t-elle change ?). "
            "Re-saisis ton token dans ton profil.") from erreur
    return u["jira_email"], token
