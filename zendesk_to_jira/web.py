"""
web.py — L'interface WEB locale de l'outil Zendesk -> Jira (FastAPI).

C'est la "fine couche" du schema d'architecture (brief §4) : ce fichier recoit
les requetes du navigateur et appelle le moteur (noyau.py). Il ne contient
AUCUNE logique metier lui-meme.

------------------------------------------------------------------------------
ETAPE 3 (brief §9) : la page d'accueil GET / liste les tickets tagges 'to-jira'
(lecture seule, rien n'est cree). Un champ "mode test" permet d'ouvrir un ticket
par son ID. Les routes /ticket/{id} et .../creer viendront aux etapes 4 et 5.
------------------------------------------------------------------------------

POUR LANCER LE SERVEUR (depuis le dossier zendesk_to_jira/) :

    .venv/bin/uvicorn web:app --reload

Puis ouvre http://127.0.0.1:8000 dans ton navigateur. Ctrl+C pour arreter.

Le CLI (create_jira.py, run_batch.py...) continue de fonctionner a l'identique :
ce fichier ne fait qu'APPELER le moteur, il ne le modifie pas.
"""

import os

import requests

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import noyau
import create_jira
import comptes

# 'app' est l'application FastAPI : l'objet que uvicorn fait tourner.
app = FastAPI(title="Zendesk -> Jira (interface locale)")

# --- Sessions (login) -------------------------------------------------------
# SessionMiddleware pose un cookie SIGNE (avec APP_SECRET_KEY) : on y stocke
# uniquement l'id de l'utilisateur connecte (jamais le token ni le mot de passe).
# 'https_only' = cookie envoye seulement en HTTPS -> on l'active en prod
# (APP_ENV=prod) mais pas en local (sinon le cookie ne passerait pas en http).
app.add_middleware(
    SessionMiddleware,
    secret_key=noyau.get_required("APP_SECRET_KEY"),
    https_only=(os.getenv("APP_ENV", "").lower() == "prod"),
    same_site="lax",
    max_age=60 * 60 * 8,   # 8 heures
)

# La base de comptes existe des le demarrage (cree la table si besoin).
comptes.init_db()


def _injecter_utilisateur(request: Request) -> dict:
    """Context processor : rend l'utilisateur connecte dispo dans TOUS les templates
    (pour afficher son nom + le bouton Deconnexion dans le header)."""
    uid = request.session.get("user_id")
    return {"utilisateur": comptes.get_utilisateur(uid) if uid else None}


# Jinja2Templates dit a FastAPI ou trouver les fichiers HTML "a trous".
templates = Jinja2Templates(directory="templates",
                            context_processors=[_injecter_utilisateur])


# ----------------------------------------------------------------------------
# GESTIONNAIRE D'ERREUR (robustesse, brief §6)
# ----------------------------------------------------------------------------
# Quand une route leve noyau.ErreurOutil (config manquante, ticket introuvable,
# API KO...), FastAPI appelle CE gestionnaire au lieu de planter : il rend une
# page d'erreur lisible. Le serveur reste debout, et le navigateur ne voit
# jamais de trace technique brute. Un seul endroit pour toutes les routes.
@app.exception_handler(noyau.ErreurOutil)
def gerer_erreur_outil(request: Request, exc: noyau.ErreurOutil):
    return templates.TemplateResponse(
        request, "erreur.html", {"message": str(exc)}, status_code=400
    )


# Erreur RESEAU (Zendesk/Jira injoignable, timeout...) : on rend une page propre
# au lieu d'une 500 brute. 502 = "un service en amont n'a pas repondu".
@app.exception_handler(requests.RequestException)
def gerer_erreur_reseau(request: Request, exc: requests.RequestException):
    return templates.TemplateResponse(
        request, "erreur.html",
        {"message": "Un service externe (Zendesk, Jira) est injoignable pour le "
                    "moment. Réessaie dans quelques instants."},
        status_code=502,
    )


# ----------------------------------------------------------------------------
# AUTHENTIFICATION (v3 multi-utilisateur)
# ----------------------------------------------------------------------------
# Deux "erreurs" qui ne sont pas des erreurs mais des REDIRECTIONS :
#   - BesoinSetup : aucun compte n'existe encore -> page d'amorce /setup.
#   - NonConnecte : il faut etre connecte -> page /login.
# On les leve depuis la dependance 'requiert_connexion', et deux gestionnaires
# les transforment en redirection. Pratique : une seule regle pour toutes les
# routes protegees (il suffit d'ajouter Depends(requiert_connexion)).
class BesoinSetup(Exception):
    pass


class NonConnecte(Exception):
    pass


class BesoinProfil(Exception):
    # L'utilisateur est connecte mais n'a pas (encore) configure son token Jira.
    pass


@app.exception_handler(BesoinSetup)
def _vers_setup(request: Request, exc: BesoinSetup):
    return RedirectResponse(url="/setup", status_code=303)


@app.exception_handler(NonConnecte)
def _vers_login(request: Request, exc: NonConnecte):
    return RedirectResponse(url="/login", status_code=303)


@app.exception_handler(BesoinProfil)
def _vers_profil(request: Request, exc: BesoinProfil):
    return RedirectResponse(url="/profil", status_code=303)


def requiert_connexion(request: Request) -> dict:
    """Dependance : renvoie l'utilisateur connecte, ou redirige (setup/login).

    A mettre sur toute route protegee : `user: dict = Depends(requiert_connexion)`.
    """
    if comptes.compter_utilisateurs() == 0:
        raise BesoinSetup()
    uid = request.session.get("user_id")
    utilisateur = comptes.get_utilisateur(uid) if uid else None
    if utilisateur is None:
        raise NonConnecte()
    return utilisateur


def requiert_admin(request: Request) -> dict:
    """Comme requiert_connexion, mais exige le role admin (pour la future page /admin)."""
    utilisateur = requiert_connexion(request)
    if not utilisateur.get("est_admin"):
        raise noyau.ErreurOutil("Accès réservé aux administrateurs.")
    return utilisateur


def session_jira_utilisateur(user: dict) -> tuple:
    """Session Jira authentifiee avec le token de L'UTILISATEUR connecte.

    C'est le coeur du multi-utilisateur : la creation passe par CES identifiants,
    donc le ticket Jira a pour CREATEUR (et rapporteur) l'agent connecte -> il le
    retrouve dans son filtre « Tickets que j'ai creés ». Si l'utilisateur n'a pas
    encore configure son token, on le renvoie vers /profil (BesoinProfil).
    (Zendesk, lui, reste le profil partage du .env.)
    """
    creds = comptes.lire_token_jira(user["id"])
    if creds is None:
        raise BesoinProfil()
    email, token = creds
    return noyau.session_jira_pour(email, token)


# ----------------------------------------------------------------------------
# /setup  : amorce — creation du TOUT PREMIER compte (admin), si la base est vide
# ----------------------------------------------------------------------------
@app.get("/setup", response_class=HTMLResponse)
def setup_form(request: Request):
    # Si des comptes existent deja, l'amorce est terminee -> on renvoie au login.
    if comptes.compter_utilisateurs() > 0:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "setup.html", {})


@app.post("/setup", response_class=HTMLResponse)
def setup_creer(request: Request,
                identifiant: str = Form(...),
                mot_de_passe: str = Form(...)):
    if comptes.compter_utilisateurs() > 0:
        return RedirectResponse(url="/login", status_code=303)
    # Le premier compte est forcement admin.
    uid = comptes.creer_utilisateur(identifiant, mot_de_passe, est_admin=True)
    request.session["user_id"] = uid          # on le connecte directement
    return RedirectResponse(url="/", status_code=303)


# ----------------------------------------------------------------------------
# /login  et  /logout
# ----------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    if comptes.compter_utilisateurs() == 0:
        return RedirectResponse(url="/setup", status_code=303)
    return templates.TemplateResponse(request, "login.html", {})


@app.post("/login", response_class=HTMLResponse)
def login_verifier(request: Request,
                   identifiant: str = Form(...),
                   mot_de_passe: str = Form(...)):
    utilisateur = comptes.verifier_identifiants(identifiant, mot_de_passe)
    if utilisateur is None:
        # Message volontairement vague (ne pas reveler si l'identifiant existe).
        return templates.TemplateResponse(
            request, "login.html",
            {"erreur": "Identifiant ou mot de passe incorrect."}, status_code=401)
    request.session["user_id"] = utilisateur["id"]
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ----------------------------------------------------------------------------
# /profil  : chaque utilisateur enregistre SON token API Jira (chiffre)
# ----------------------------------------------------------------------------
# C'est ce token qui sera utilise pour creer les tickets sous son profil (etape 4).
# A la soumission, on VALIDE le token aupres de Jira (/myself) avant de le stocker.
@app.get("/profil", response_class=HTMLResponse)
def profil_form(request: Request, user: dict = Depends(requiert_connexion)):
    return templates.TemplateResponse(request, "profil.html", {
        "jira_ok": bool(user.get("jira_account_id")),
        "jira_email": user.get("jira_email"),
        "jira_nom": user.get("jira_nom"),
    })


@app.post("/profil", response_class=HTMLResponse)
def profil_enregistrer(request: Request,
                       jira_email: str = Form(...),
                       jira_token: str = Form(...),
                       user: dict = Depends(requiert_connexion)):
    identite = noyau.verifier_identite_jira(jira_email.strip(), jira_token.strip())
    if identite is None:
        return templates.TemplateResponse(request, "profil.html", {
            "erreur": "Identifiants Jira refusés : vérifie ton email et ton token API.",
            "jira_ok": bool(user.get("jira_account_id")),
            "jira_email": jira_email,
            "jira_nom": user.get("jira_nom"),
        }, status_code=400)
    comptes.definir_token_jira(user["id"], jira_email.strip(), jira_token.strip(),
                               identite["accountId"], identite["nom"])
    return templates.TemplateResponse(request, "profil.html", {
        "succes": True, "jira_ok": True,
        "jira_email": jira_email.strip(), "jira_nom": identite["nom"],
    })


# ----------------------------------------------------------------------------
# /admin  : gestion des comptes (reserve aux administrateurs)
# ----------------------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, admin: dict = Depends(requiert_admin)):
    return templates.TemplateResponse(request, "admin.html",
                                      {"comptes": comptes.lister_utilisateurs()})


@app.post("/admin/creer")
def admin_creer(request: Request,
                identifiant: str = Form(...),
                mot_de_passe: str = Form(...),
                est_admin: str = Form(None),
                admin: dict = Depends(requiert_admin)):
    comptes.creer_utilisateur(identifiant, mot_de_passe, est_admin=bool(est_admin))
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/reinitialiser")
def admin_reinitialiser(request: Request,
                        user_id: int = Form(...),
                        nouveau_mdp: str = Form(...),
                        admin: dict = Depends(requiert_admin)):
    comptes.reinitialiser_mot_de_passe(user_id, nouveau_mdp)
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/supprimer")
def admin_supprimer(request: Request,
                    user_id: int = Form(...),
                    admin: dict = Depends(requiert_admin)):
    # Garde-fous : ne pas se supprimer soi-meme, ni supprimer le dernier admin
    # (sinon on se verrouille hors de la gestion des comptes).
    if user_id == admin["id"]:
        raise noyau.ErreurOutil("Tu ne peux pas supprimer ton propre compte.")
    cible = comptes.get_utilisateur(user_id)
    if cible and cible.get("est_admin"):
        nb_admins = sum(1 for u in comptes.lister_utilisateurs() if u.get("est_admin"))
        if nb_admins <= 1:
            raise noyau.ErreurOutil("Impossible de supprimer le dernier administrateur.")
    comptes.supprimer_utilisateur(user_id)
    return RedirectResponse(url="/admin", status_code=303)


# ----------------------------------------------------------------------------
# GET /  : page d'accueil = liste des tickets tagges 'to-jira'
# ----------------------------------------------------------------------------
# Le parametre 'request: Request' est OBLIGATOIRE pour rendre un template Jinja2
# (FastAPI en a besoin pour construire la reponse). On ouvre une session Zendesk
# (les secrets restent cote serveur, jamais envoyes au navigateur), on demande au
# moteur la liste des tickets, et on la passe au template.
@app.get("/", response_class=HTMLResponse)
def accueil(request: Request, user: dict = Depends(requiert_connexion)):
    zd, base_zd = noyau.session_zendesk()
    tickets = noyau.lister_tickets_a_traiter(zd, base_zd)
    # TemplateResponse remplit accueil.html avec nos donnees et renvoie le HTML.
    # Signature recente de Starlette : la requete EN PREMIER, puis le nom du
    # template, puis le dict de donnees a injecter dans la page.
    return templates.TemplateResponse(
        request, "accueil.html", {"tickets": tickets}
    )


# ----------------------------------------------------------------------------
# GET /aller-ticket?id=...  : petit aiguillage pour le champ "mode test"
# ----------------------------------------------------------------------------
# Un formulaire HTML natif ne sait pas injecter une valeur directement dans un
# CHEMIN d'URL (/ticket/4927). On passe donc par cette route qui recoit l'id en
# parametre, puis REDIRIGE vers la "vraie" route /ticket/{id} (chemin propre).
@app.get("/aller-ticket")
def aller_ticket(id: str, user: dict = Depends(requiert_connexion)):
    return RedirectResponse(url=f"/ticket/{id.strip()}", status_code=303)


# ----------------------------------------------------------------------------
# GET /ticket/{id}  : affiche le ticket (lecture) + le formulaire de qualification
# ----------------------------------------------------------------------------
# On re-recupere le ticket a chaque fois depuis Zendesk (approche STATELESS du
# brief §5 : pas d'etat garde en memoire serveur entre deux requetes). On prepare
# tout ce dont le template a besoin, puis on rend ticket.html. RIEN n'est cree ici.
@app.get("/ticket/{ticket_id}", response_class=HTMLResponse)
def voir_ticket(request: Request, ticket_id: str,
                user: dict = Depends(requiert_connexion)):
    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = session_jira_utilisateur(user)   # identite Jira de l'agent connecte

    ticket = noyau.charger_ticket(zd, base_zd, ticket_id)

    # Garde-fou anti-doublon : si le champ "Cle Jira liee" est rempli, on bloque.
    if ticket["cle_jira_existante"]:
        return templates.TemplateResponse(request, "ticket.html", {
            "ticket": ticket,
            "deja_converti": True,
            "lien_jira": f"{base_jira}/browse/{ticket['cle_jira_existante']}",
        })

    # Briques du moteur : type/assigne (via tags), options plateforme, auto-match.
    infos = noyau.detecter_type_et_assigne(ticket["tags"])
    options_brutes = noyau.charger_options_plateforme(jira, base_jira)
    plateformes = sorted(
        ({"id": o.get("id"), "label": o.get("value") or o.get("name") or ""}
         for o in options_brutes),
        key=lambda p: p["label"].lower(),
    )
    match = noyau.matcher_plateforme(ticket["plateforme_zendesk"], options_brutes)
    pieces = noyau.lister_pieces_jointes(zd, base_zd, ticket_id)

    # Reformulation de la description en rapport de bug (via Claude). La fonction
    # ne leve jamais : elle renvoie le texte reformule, ou None si l'appel echoue
    # (cle absente, API down...). En cas de None, on NE bloque PAS : on retombe
    # sur le texte brut et 'reformulation_ok' declenche l'avertissement dans la
    # page. Le champ reste editable dans les deux cas.
    # On passe a l'IA le CONTEXTE connu de l'outil (type + plateforme/client) pour
    # une reformulation mieux ancree (et la mention explicite de la plateforme).
    plateforme = match["label"] if match else (ticket["plateforme_zendesk"] or "")
    contexte = f"Type de ticket : {infos['type']}"
    if plateforme:
        contexte += f"\nPlateforme / client concerné : {plateforme}"
    reformulation = noyau.reformuler_description(ticket["description"], ticket["subject"], contexte)
    reformulation_ok = reformulation is not None
    description_draft = reformulation if reformulation_ok else ticket["description"]

    return templates.TemplateResponse(request, "ticket.html", {
        "ticket": ticket,
        "deja_converti": False,
        "infos": infos,
        "priorites": list(noyau.PRIORITES),
        "assignes": noyau.assignes_possibles(),
        "plateformes": plateformes,
        "match": match,
        "pieces": pieces,
        "lien_zendesk": noyau.url_ticket_zendesk(base_zd, ticket_id),
        "description_draft": description_draft,
        "reformulation_ok": reformulation_ok,
    })


# ----------------------------------------------------------------------------
# POST /ticket/{id}/creer  : recoit le formulaire et CREE reellement le Jira
# ----------------------------------------------------------------------------
# Chaque champ du formulaire arrive via Form(...). Les champs conditionnels
# (systeme pour les apps ; onglet/bofo pour le web) ont une valeur par defaut
# None car ils ne sont pas tous presents selon le type.
#
# Le flux : on re-charge le ticket (stateless), on reconstruit les memes briques
# que le CLI (type, plateforme, choix), on assemble 'fields' via le MOTEUR
# (noyau.construire_fields), puis on cree via la fonction PARTAGEE avec le CLI
# (create_jira.creer_depuis_fields). web.py n'a aucune logique metier propre.
@app.post("/ticket/{ticket_id}/creer", response_class=HTMLResponse)
def creer_ticket(request: Request, ticket_id: str,
                 description: str = Form(...),
                 nom_prio: str = Form(...),
                 assigne_id: str = Form(...),
                 plateforme_id: str = Form(...),
                 systeme: str = Form(None),
                 onglet: str = Form(None),
                 bofo: str = Form(None),
                 user: dict = Depends(requiert_connexion)):
    zd, base_zd = noyau.session_zendesk()
    jira, base_jira = session_jira_utilisateur(user)   # identite Jira de l'agent connecte

    ticket = noyau.charger_ticket(zd, base_zd, ticket_id)

    # Garde-fou anti-doublon (re-verifie au moment de creer : approche stateless,
    # le ticket a pu etre converti entre l'affichage du formulaire et la soumission).
    if ticket["cle_jira_existante"]:
        return templates.TemplateResponse(request, "resultat.html", {
            "ok": False,
            "deja_converti": True,
            "ticket_id": ticket_id,
            "cle": ticket["cle_jira_existante"],
            "lien_jira": f"{base_jira}/browse/{ticket['cle_jira_existante']}",
        })

    infos = noyau.detecter_type_et_assigne(ticket["tags"])

    # Reconstruire la plateforme {id, label} a partir de l'id soumis.
    options_brutes = noyau.charger_options_plateforme(jira, base_jira)
    match = next(({"id": o.get("id"), "label": o.get("value") or o.get("name") or ""}
                  for o in options_brutes if o.get("id") == plateforme_id),
                 {"id": plateforme_id, "label": plateforme_id})

    # Retrouver le libelle de l'assigne (pour le recap).
    assigne_label = next((a["label"] for a in noyau.assignes_possibles()
                          if a["id"] == assigne_id), assigne_id)

    # Le dict 'choix' attendu par le moteur (memes cles que cote CLI), enrichi de
    # la description reformulee ET editee par l'utilisateur dans le textarea.
    choix = {"nom_prio": nom_prio, "assigne_id": assigne_id,
             "assigne_label": assigne_label, "description": description}
    if systeme:
        choix["systeme"] = systeme
    if onglet:
        choix["onglet"] = onglet
    if bofo:
        choix["bofo"] = bofo

    # Assemblage (moteur) puis creation reelle (fonction partagee avec le CLI).
    fields, meta = noyau.construire_fields(ticket, infos, match, choix)
    res = create_jira.creer_depuis_fields(zd, base_zd, jira, base_jira, ticket, fields)

    if not res["cle"]:
        return templates.TemplateResponse(request, "resultat.html", {
            "ok": False, "deja_converti": False, "ticket_id": ticket_id,
        })

    return templates.TemplateResponse(request, "resultat.html", {
        "ok": True,
        "ticket_id": ticket_id,
        "cle": res["cle"],
        "lien_jira": f"{base_jira}/browse/{res['cle']}",
        "resume": fields["summary"],
        "meta": meta,
        "nb_pj": res["nb_pj"],
        # Si False : le Jira est cree mais le champ 'Cle Jira liee' n'a PAS pu etre
        # ecrit cote Zendesk -> on previent (sinon un re-run creerait un doublon).
        "writeback_ok": res.get("writeback_ok", True),
    })
