import os
import time
import chromadb
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# --- Config ---
INDEX_URL = "https://www.vodfactory.com/otto-academy/les-tutoriels"
BASE_URL = "https://www.vodfactory.com"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "kbase"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
OVERLAP = 150

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; OttobotScraper/1.0)"
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
chroma = chromadb.CloudClient(
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
    api_key=os.environ["CHROMA_API_KEY"],
)

try:
    chroma.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = chroma.get_or_create_collection(COLLECTION_NAME)


# --- Étape 1 : Récupérer toutes les URLs des tutoriels ---
def get_tutorial_urls() -> list[str]:
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    EXCLUDE = ["/otto-academy/home", "/otto-academy/les-tutoriels"]
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ("/otto-academy-tutoriels/" in href or "/otto-academy/" in href) and not any(e in href for e in EXCLUDE):
            full_url = href if href.startswith("http") else BASE_URL + href
            if full_url not in urls:
                urls.append(full_url)

    return urls


# --- Étape 2 : Scraper une page de tutoriel ---
def scrape(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Titre
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else url

    # Contenu principal — on essaie plusieurs sélecteurs courants
    content = (
        soup.find("article") or
        soup.find("main") or
        soup.find(class_="post-content") or
        soup.find(class_="entry-content") or
        soup.find(class_="content") or
        soup.body
    )

    # On retire les éléments inutiles (nav, footer, scripts...)
    for tag in content.find_all(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    text = content.get_text(separator=" ", strip=True) if content else ""
    return {"url": url, "title": title, "text": text}


# --- Étape 3 : Découper en chunks ---
def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split()).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start = end - OVERLAP if end - OVERLAP > start else start + 1
    return chunks


# --- Étape 4 : Embeddings par batch ---
def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [x.embedding for x in resp.data]


# --- Programme principal ---
if __name__ == "__main__":
    print("🔍 Récupération des URLs...")
    urls = get_tutorial_urls()
    print(f"   {len(urls)} tutoriels trouvés :\n")
    for u in urls:
        print(f"   · {u}")

    if not urls:
        print("\n⚠️  Aucune URL trouvée. Voir note ci-dessous.")
        exit(1)

    print("\n📄 Scraping des tutoriels...")
    all_docs, all_metas, all_ids = [], [], []
    chunk_idx = 0

    for url in urls:
        print(f"  → {url}")
        try:
            page = scrape(url)
            chunks = chunk_text(page["text"])
            if not chunks:
                print(f"     ⚠️  Aucun texte extrait")
                continue
            for chunk in chunks:
                all_docs.append(chunk)
                all_metas.append({"source": page["title"], "url": url})
                all_ids.append(f"chunk_{chunk_idx}")
                chunk_idx += 1
            print(f"     ✅ {len(chunks)} chunks")
        except Exception as e:
            print(f"     ❌ Erreur : {e}")

        time.sleep(0.8)  # Respectueux envers le serveur

    if not all_docs:
        print("\n❌ Aucun chunk indexé. Vérifiez les sélecteurs CSS.")
        exit(1)

    print(f"\n⚡ {len(all_docs)} chunks à indexer. Embeddings en cours...")

    BATCH = 100
    for i in range(0, len(all_docs), BATCH):
        b_docs  = all_docs[i:i+BATCH]
        b_metas = all_metas[i:i+BATCH]
        b_ids   = all_ids[i:i+BATCH]
        b_embs  = embed_batch(b_docs)
        collection.add(ids=b_ids, documents=b_docs,
                       embeddings=b_embs, metadatas=b_metas)
        print(f"   Indexé {min(i+BATCH, len(all_docs))}/{len(all_docs)}")

    print("\n✅ Indexation terminée ! Vous pouvez lancer app.py")
