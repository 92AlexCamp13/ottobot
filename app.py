import os
import hmac
import html
import streamlit as st
from openai import OpenAI
import chromadb

# ----------------------------
# Page config + style
# ----------------------------
st.set_page_config(page_title="Ottobot", page_icon="logo_OttoBot.png", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Orbitron:wght@900&display=swap');

*, body, .stApp {
  font-family: 'IBM Plex Sans', sans-serif !important;
}
[style*="Orbitron"] {
  font-family: 'Orbitron', sans-serif !important;
}
.stApp {
  background: #1a1625;
  color: #e8e4f0;
}
.block-container {
  padding-top: 2.5rem;
  max-width: 760px;
  margin: 0 auto;
}
h1, h2, h3 {
  font-weight: 600;
  letter-spacing: -0.02em;
  color: #f0ecf8;
}
.kpi {
  border: 1px solid rgba(167, 139, 250, 0.2);
  border-radius: 12px;
  padding: 14px 16px;
  background: rgba(167, 139, 250, 0.05);
}
.sourcebox {
  border: 1px solid rgba(167, 139, 250, 0.15);
  border-radius: 10px;
  padding: 10px 14px;
  background: rgba(255, 255, 255, 0.03);
  margin-bottom: 10px;
  font-size: 0.92rem;
  color: #c4bdd4;
}
.smallmuted { color: #9b93ab; font-size: 0.9rem; }
.stTextInput input, .stChatInput textarea {
  font-family: 'IBM Plex Sans', sans-serif !important;
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(167, 139, 250, 0.25) !important;
  border-radius: 10px !important;
  color: #e8e4f0 !important;
}
.stButton button {
  font-family: 'IBM Plex Sans', sans-serif !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  background: rgba(167, 139, 250, 0.15) !important;
  border: 1px solid rgba(167, 139, 250, 0.3) !important;
  color: #c4b5fd !important;
}
.stButton button:hover {
  background: rgba(167, 139, 250, 0.25) !important;
}
section[data-testid="stSidebar"] {
  background: #13101e !important;
  border-right: 1px solid rgba(167, 139, 250, 0.15);
}
.stChatMessage {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(167, 139, 250, 0.12) !important;
  border-radius: 12px !important;
}
.stChatMessage [data-testid="chatAvatarIcon-user"],
.stChatMessage [data-testid="chatAvatarIcon-assistant"],
.stChatMessage img,
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  display: none !important;
}
.stChatInput:focus-within {
  border-color: rgba(167, 139, 250, 0.4) !important;
  box-shadow: none !important;
}
.streamlit-expanderHeader,
[data-testid="stExpander"] summary,
[data-testid="stExpanderToggleIcon"] {
  color: #c4b5fd !important;
  font-weight: 500 !important;
}
[data-testid="stExpanderToggleIcon"] svg { display: inline !important; }
button[data-testid="stBaseButton-minimal"] { color: #c4b5fd !important; }
hr { border-color: rgba(167, 139, 250, 0.15); }
.stChatInput {
  max-width: 760px !important;
  margin: 0 auto !important;
  background: #1a1625 !important;
}
.stChatInput > div { background: #1a1625 !important; }
[data-testid="stBottomBlockContainer"] { background: #1a1625 !important; }
header[data-testid="stHeader"] { background: #1a1625 !important; }
footer { display: none !important; }
/* Masquer bouton collapse sidebar */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
button[kind="header"] {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
a { color: #a78bfa !important; }
a:hover { color: #c4b5fd !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1.5rem 0;">
      <div style="font-size:56px;font-weight:700;letter-spacing:-0.02em;color:#f0ecf8;font-family:'Orbitron', sans-serif;">Ottobot</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# OpenAI
# ----------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY introuvable.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Chroma
# ----------------------------
PROJECT_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(PROJECT_DIR, "chroma_db")
COLLECTION_NAME = "kbase"

if "chroma" not in st.session_state:
    st.session_state["chroma"] = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    return st.session_state["chroma"].get_or_create_collection(name=COLLECTION_NAME)

def clear_collection():
    try:
        st.session_state["chroma"].delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

def collection_has_data(col) -> bool:
    try:
        return col.count() > 0
    except Exception:
        return False

# ----------------------------
# Helpers
# ----------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap doit être inférieur à chunk_size")
    text = " ".join(text.split()).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else start + 1
    return chunks

@st.cache_data(show_spinner=False)
def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return resp.data[0].embedding

def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        raise ValueError("Aucun texte valide à embedder.")
    resp = client.embeddings.create(model="text-embedding-3-small", input=cleaned)
    return [x.embedding for x in resp.data]

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    admin_token = os.environ.get("OTTOBOT_ADMIN_TOKEN", "")
    is_admin_url = st.query_params.get("admin", "") == "true"

    if is_admin_url:
        st.markdown("**Parametres Admin**")
        entered_password = st.text_input("Mot de passe admin", type="password")
        is_admin = bool(admin_token) and hmac.compare_digest(entered_password, admin_token)
        if entered_password and not is_admin:
            st.caption("Mot de passe incorrect.")
    else:
        is_admin = False

    top_k = 3

    st.divider()
    st.markdown("**Historique**")
    if st.session_state.get("chat"):
        questions = [m for m in st.session_state["chat"] if m["role"] == "user"]
        for q in questions:
            st.markdown(
                f"<div style='border:1px solid rgba(167,139,250,0.2);border-radius:8px;"
                f"padding:8px 10px;margin-bottom:8px;font-size:0.85rem;color:#c4b5fd;'>"
                f"<b>{q['content'][:50]}...</b></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Aucune question posée pour l'instant.")

# ----------------------------
# Chroma collection
# ----------------------------
collection = get_collection()

st.divider()

# ----------------------------
# Espace Admin
# ----------------------------
if is_admin:
    st.subheader("Espace Admin")

    if collection_has_data(collection):
        st.markdown(f"**Chunks indexés :** {collection.count()}")

        # Liste des tutoriels — sans expander pour éviter les bugs
        results = collection.get(include=["metadatas"])
        sources = {}
        for meta in results["metadatas"]:
            title = meta.get("source", "Inconnu")
            url = meta.get("url", "")
            if title not in sources:
                sources[title] = url

        st.markdown(f"**Tutoriels indexés ({len(sources)}) :**")
        tutos_html = "".join([
            f"<div class='sourcebox'>"
            f"<a href='{url}' target='_blank'>{title}</a>"
            f"</div>"
            if url else
            f"<div class='sourcebox'>{title}</div>"
            for title, url in sorted(sources.items())
        ])
        st.markdown(tutos_html, unsafe_allow_html=True)

    else:
        st.warning("Base vide. Lancez scraper.py pour indexer les tutoriels.")

    st.divider()

    # Recherche de test
    st.markdown("**Tester une recherche dans la base**")
    test_query = st.text_input("Entrez un mot-clé ou une question...")
    if test_query and collection_has_data(collection):
        q_emb = embed_query(test_query)
        test_results = collection.query(query_embeddings=[q_emb], n_results=3)
        test_docs = test_results.get("documents", [[]])[0]
        test_metas = test_results.get("metadatas", [[]])[0]
        st.markdown("**Top 3 extraits trouvés :**")
        for i, (doc, meta) in enumerate(zip(test_docs, test_metas), 1):
            src = meta.get("source", "?")
            url = meta.get("url", "")
            st.markdown(
                f"<div class='sourcebox'><b>Extrait {i} — {src}</b>"
                f"{'<br/><a href=' + url + ' target=_blank>Voir le tutoriel</a>' if url else ''}"
                f"<br/><small>{doc[:200]}...</small></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Zone dangereuse
    st.markdown("**Zone dangereuse**")
    if st.button("Vider et réinitialiser la base", type="secondary"):
        clear_collection()
        collection = get_collection()
        st.warning("Base vidée. Relancez scraper.py pour réindexer.")

    st.divider()

# ----------------------------
# Accroche + questions suggérées
# ----------------------------
st.markdown(
    "<p style='text-align:center;color:#9b93ab;font-size:14px;'>"
    "Une question sur Otto ? Obtenez la réponse en 2 clics</p>",
    unsafe_allow_html=True,
)

if not st.session_state.get("chat"):
    col1, col2, col3 = st.columns(3)
    questions_suggerees = [
        "Comment créer ma clé SSH ?",
        "Comment configurer un bloc de A à Z ?",
        "Comment créer une fiche contenu/série ?",
    ]
    for col, question in zip([col1, col2, col3], questions_suggerees):
        with col:
            if st.button(question, use_container_width=True):
                st.session_state["pending_prompt"] = question

st.divider()

# ----------------------------
# Chat
# ----------------------------
if "chat" not in st.session_state:
    st.session_state["chat"] = []

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

_pending = st.session_state.pop("pending_prompt", None)
prompt = st.chat_input("Posez votre question...") or _pending

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:
    st.session_state["chat"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    collection = get_collection()
    if not collection_has_data(collection):
        with st.chat_message("assistant"):
            st.warning("Je n'ai pas de base indexée. Lancez scraper.py d'abord.")
    else:
        with st.spinner("Recherche des passages pertinents..."):
            q_emb = embed_query(prompt)
            results = collection.query(query_embeddings=[q_emb], n_results=top_k)
            contexts = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

        context = "\n\n---\n\n".join(contexts)

        system = (
            "Tu es Ottobot, un assistant support pour Otto Academy by VodFactory. "
            "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
            "Si le contexte ne contient pas l'information, dis : "
            "\"Je ne trouve pas cette information dans les tutoriels.\" "
            "Réponse concise, utile, en français. Sans emojis."
        )

        with st.spinner("Génération de la réponse..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"CONTEXTE:\n{context}\n\nQUESTION:\n{prompt}"}
                ],
            )
            answer = resp.choices[0].message.content

        st.session_state["chat"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

            if contexts:
                with st.expander("Sources"):
                    for i, (c, meta) in enumerate(zip(contexts, metadatas), 1):
                        src = (meta or {}).get("source", "?")
                        url = (meta or {}).get("url", "")
                        label = f"Extrait {i} — {src}"
                        safe_c = html.escape(c)
                        st.markdown(
                            f"<div class='sourcebox'><b>{label}</b><br/>{safe_c}</div>",
                            unsafe_allow_html=True,
                        )
                        if url:
                            st.markdown(f"[Voir le tutoriel]({url})")