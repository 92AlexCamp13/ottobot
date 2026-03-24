import os
import hmac
import html
import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Ottobot", page_icon="logo_OttoBot.png", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
body, .stApp { 
  font-family: 'Inter', sans-serif; 
}
.stApp { 
  background: #FFFFFF; 
  color: #1a1a2e; 
}
.block-container { 
  padding-top: 2.5rem; 
  max-width: 760px; 
  margin: 0 auto; 
}

/* Titres */
h1, h2, h3 { 
  font-weight: 600; 
  letter-spacing: -0.02em; 
  color: #1a1a2e; 
}

/* Box des sources */
.sourcebox {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px 16px;
  background: #f9fafb;
  margin-bottom: 10px;
  font-size: 0.92rem;
  color: #4b5563;
  max-width: 100%;
  overflow-wrap: break-word;
  word-break: break-word;
  transition: border-color 0.2s ease;
}
.sourcebox:hover {
  border-color: #6366f1;
}

/* Inputs */
.stTextInput input, .stChatInput textarea {
  font-family: 'Inter', sans-serif !important;
  background: #ffffff !important;
  border: 2px solid #e5e7eb !important;
  border-radius: 12px !important;
  color: #1a1a2e !important;
  min-height: 50px !important;
}
.stTextInput input:focus, .stChatInput textarea:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Boutons */
.stButton button {
  font-family: 'Inter', sans-serif !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  background: #6366f1 !important;
  border: none !important;
  color: #ffffff !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.2s ease !important;
}
.stButton button:hover { 
  background: #4f46e5 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

/* Messages du chat */
.stChatMessage {
  background: #f9fafb !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 16px !important;
  max-width: 100% !important;
  overflow: hidden !important;
  margin-bottom: 1rem !important;
}
.stChatMessage > div,
.stChatMessage p,
.stChatMessage li,
.stChatMessage span {
  max-width: 100% !important;
  overflow-wrap: break-word !important;
  word-break: break-word !important;
  color: #374151 !important;
}
.stChatMessage pre, .stChatMessage code {
  white-space: pre-wrap !important;
  overflow-x: auto !important;
  max-width: 100% !important;
  background: #1a1a2e !important;
  color: #e5e7eb !important;
  border-radius: 8px !important;
}

/* Cacher les avatars */
.stChatMessage [data-testid="chatAvatarIcon-user"],
.stChatMessage [data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { 
  display: none !important; 
}

/* Input du chat focus */
.stChatInput:focus-within { 
  border-color: #6366f1 !important; 
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important; 
}

/* Séparateurs */
hr { 
  border-color: #e5e7eb; 
}

/* Footer et sidebar cachés */
footer { display: none !important; }
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { 
  display: none !important; 
  width: 0 !important; 
  visibility: hidden !important; 
}

/* Liens */
a { 
  color: #6366f1 !important; 
  text-decoration: none !important;
}
a:hover { 
  color: #4f46e5 !important; 
  text-decoration: underline !important;
}

/* Header */
header[data-testid="stHeader"] { 
  background: #ffffff !important;
  border-bottom: 1px solid #e5e7eb !important;
}

/* Footer/Chat input */
.stChatInput,
.stChatInput > div {
  background: #ffffff !important;
  max-width: 100% !important;
  width: 760px !important;
  margin: 0 auto !important;
}

[data-testid="stBottomBlockContainer"],
.st-emotion-cache-hzygls,
.st-emotion-cache-6shykm,
.eht7o1d3,
.eht7o1d7,
[data-testid="stBottom"] {
  background: #ffffff !important;
  border-top: 1px solid #e5e7eb !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  padding: 12px 0 !important;
  min-height: auto !important;
  max-height: 80px !important;
}

/* Spinner/Loading */
.stSpinner > div {
  border-top-color: #6366f1 !important;
}

/* Warnings et alertes */
.stAlert {
  border-radius: 12px !important;
  border: none !important;
}
</style>
""", unsafe_allow_html=True)

# Header avec logo Otto Academy
st.markdown("""
<div style="text-align:center;padding:2rem 0 1.5rem 0;">
  <img src="https://cdn.prod.website-files.com/5d1b4c09d7f0159a77c39cb1/63612273c638609bffb6246c_otto-academy_logo.png" 
       alt="Otto Academy" 
       style="height:60px;margin-bottom:0.5rem;">
  <div style="font-size:32px;font-weight:700;letter-spacing:-0.02em;color:#1a1a2e;margin-top:0.5rem;">
    Ottobot
  </div>
  <p style="color:#6b7280;font-size:14px;margin-top:0.25rem;">
    Votre assistant intelligent Otto Academy
  </p>
</div>
""", unsafe_allow_html=True)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY introuvable.")
    st.stop()

client = OpenAI(api_key=api_key)

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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap doit être inférieur à chunk_size")
    text = " ".join(text.split()).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
        if start + overlap >= len(text):
            break
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

admin_token = os.environ.get("OTTOBOT_ADMIN_TOKEN", "")
is_admin_url = st.query_params.get("admin", "") == "true"

if is_admin_url:
    entered_password = st.text_input("Mot de passe admin", type="password")
    is_admin = bool(admin_token) and hmac.compare_digest(entered_password, admin_token)
    if entered_password and not is_admin:
        st.caption("Mot de passe incorrect.")
else:
    is_admin = False

top_k = 3
collection = get_collection()

if is_admin:
    st.divider()
    st.subheader("Espace Admin")
    if collection_has_data(collection):
        st.markdown(f"**Chunks indexés :** {collection.count()}")
        results = collection.get(include=["metadatas"])
        sources = {}
        for meta in results["metadatas"]:
            title = meta.get("source", "Inconnu")
            url = meta.get("url", "")
            if title not in sources:
                sources[title] = url
        st.markdown(f"**Tutoriels indexés ({len(sources)}) :**")
        tutos_html = "".join([
            f"<div class='sourcebox'><a href='{url}' target='_blank'>{title}</a></div>"
            if url else f"<div class='sourcebox'>{title}</div>"
            for title, url in sorted(sources.items())
        ])
        st.markdown(tutos_html, unsafe_allow_html=True)
    else:
        st.warning("Base vide. Lancez scraper.py pour indexer les tutoriels.")
    st.divider()
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
                f"{'<br/><a href=' + url + ' target=_blank>Voir</a>' if url else ''}"
                f"<br/><small>{doc[:200]}...</small></div>",
                unsafe_allow_html=True,
            )
    st.divider()
    st.markdown("**Zone dangereuse**")
    if st.button("Vider et réinitialiser la base", type="secondary"):
        clear_collection()
        collection = get_collection()
        st.warning("Base vidée. Relancez scraper.py pour réindexer.")
    st.divider()

st.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:14px;'>"
    "Une question sur Otto ? Obtenez la réponse en 2 clics</p>",
    unsafe_allow_html=True,
)

if not st.session_state.get("chat"):
    col1, col2, col3 = st.columns(3)
    questions_suggerees = [
        "Comment générer ma clé SSH ?",
        "Comment configurer un bloc de A à Z ?",
        "Comment créer une fiche contenu ?",
    ]
    for i, (col, question) in enumerate(zip([col1, col2, col3], questions_suggerees)):
        with col:
            if st.button(question, use_container_width=True, key=f"btn_q{i}"):
                st.session_state["pending_prompt"] = question
                st.rerun()

st.divider()

if "chat" not in st.session_state:
    st.session_state["chat"] = []

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

_pending = st.session_state.get("pending_prompt")
if _pending:
    st.session_state["pending_prompt"] = None

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
            "Tu es Ottobot, l'assistant intelligent d'Otto Academy by VodFactory. "
            "Tu aides les utilisateurs à comprendre et utiliser la plateforme Otto. "
            "Tu as accès à des extraits de tutoriels comme contexte. "
            "Utilise ce contexte pour construire une réponse claire, pédagogique et bienveillante. "
            "Reformule toujours avec tes propres mots — ne recopie jamais le texte tel quel. "
            "Si la question comporte plusieurs étapes, structure ta réponse en étapes numérotées. "
            "Si le contexte ne contient pas l'information, dis : "
            "'Je ne trouve pas cette information dans les tutoriels. "
            "N'hésitez pas à contacter le support VodFactory.' "
            "Réponds toujours en français, sans emojis, de façon concise et utile."
        )
        with st.spinner("Génération de la réponse..."):
            messages = [{"role": "system", "content": system}]
            recent_history = st.session_state["chat"][-6:]
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({
                "role": "user",
                "content": f"CONTEXTE (extraits de tutoriels):\n{context}\n\nQUESTION:\n{prompt}"
            })
            resp = client.chat.completions.create(model="gpt-4o", messages=messages)
            answer = resp.choices[0].message.content
        st.session_state["chat"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)