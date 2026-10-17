import os

# MUST come before any ML library import so weights land on D:, not C:
os.environ['HF_HOME'] = 'D:/AI_Models_Cache'
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import json

import chromadb
import requests
import streamlit as st

# ----------------------------- Configuration -----------------------------
VIDEO_PATH = "sample_video.mp4"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "video_frames"

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You help people find moments in a video. You are given timestamped "
    "descriptions of frames, produced by an automatic image captioner. The "
    "captioner describes what it sees (e.g. 'a woman holding a baby') and rarely "
    "uses labels like 'mother' or 'child'. So match by MEANING, not exact words: "
    "infer relationships and synonyms when the visual evidence reasonably "
    "supports it, and mark such matches as 'likely'.\n\n"
    "Rules:\n"
    "1. Use only the provided descriptions. Do not invent details.\n"
    "2. Start with a one-sentence direct answer.\n"
    "3. Then list the relevant moments as bullets in the form "
    "'- mm:ss: what is visible'.\n"
    "4. If nothing plausibly matches, say so briefly and mention the closest "
    "moment instead.\n"
    "5. Never answer just 'the description does not mention X' because exact "
    "words are missing."
)
# -------------------------------------------------------------------------


class OllamaError(Exception):
    """Raised when the local LLM can't be reached or returns an error."""


# ------------------------------ Retrieval --------------------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def search(collection, query, n_results):
    """Return a list of dicts: timestamp, caption, distance (best first)."""
    n_results = min(n_results, collection.count())
    res = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"timestamp": int(meta["timestamp"]), "caption": doc, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def fmt_time(seconds):
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ------------------------------ Generation -------------------------------
def build_messages(question, results):
    """Build the chat prompt. Context is sorted chronologically so the model
    can reason about what happened before/after what."""
    ordered = sorted(results, key=lambda r: r["timestamp"])
    context = "\n".join(f"[{fmt_time(r['timestamp'])}] {r['caption']}" for r in ordered)
    user_msg = (
        f"Frame descriptions:\n{context}\n\n"
        f"User request (may be a keyword search or a question): {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def stream_answer(model, messages):
    """Yield answer text chunks from a local Ollama server."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    try:
        # (connect timeout, read timeout) - first call may load the model
        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(5, 300)
        ) as resp:
            if resp.status_code == 404:
                raise OllamaError(
                    f"Model `{model}` not found. Run `ollama pull {model}` in a terminal."
                )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                yield chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            "Can't reach Ollama at localhost:11434. Install it from ollama.com "
            "and make sure it's running."
        )
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}")


def render_answer(query, results, model, n_results):
    """Show the LLM answer. Cached per (query, model, k) so that clicking a
    different match doesn't trigger a re-generation."""
    key = (query, model, n_results)
    with st.container(border=True):
        st.subheader("💬 Answer")
        if st.session_state.get("answer_key") == key:
            st.markdown(st.session_state.answer)
            return
        try:
            with st.spinner(f"Asking {model}..."):
                answer = st.write_stream(stream_answer(model, build_messages(query, results)))
            st.session_state.answer_key = key
            st.session_state.answer = answer
        except OllamaError as e:
            st.error(str(e))


# --------------------------------- UI ------------------------------------
def main():
    st.set_page_config(page_title="Temporal Video RAG", page_icon="🎬", layout="wide")
    st.title("🎬 Temporal Video RAG")
    st.caption("Search the video or ask questions about it, then jump to the moment.")

    with st.sidebar:
        st.header("Settings")
        generate = st.toggle("Generate answer with local LLM", value=True)
        model = st.text_input("Ollama model", DEFAULT_MODEL, disabled=not generate)
        n_results = st.slider("Frames to retrieve", 3, 15, 8)

    # --- Pre-flight checks ---
    if not os.path.isfile(VIDEO_PATH):
        st.error(f"Video file `{VIDEO_PATH}` not found in the project folder.")
        st.stop()

    try:
        collection = get_collection()
    except Exception:
        st.error("Vector database not found. Run `python ingest.py` first.")
        st.stop()

    if collection.count() == 0:
        st.warning("The database is empty. Run `python ingest.py` first.")
        st.stop()

    # --- Search ---
    query = st.text_input(
        "Search or ask a question",
        placeholder="e.g. red car  —  or:  when does someone enter the room?",
    )

    results = []
    if query:
        try:
            results = search(collection, query, n_results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # --- Generated answer ---
    if results and generate:
        render_answer(query, results, model, n_results)

    # --- Video + matches ---
    col_video, col_results = st.columns([3, 2])

    start = 0  # default to the top hit; user can switch to other matches
    with col_results:
        if results:
            st.subheader("Matches")
            labels = [
                f"{fmt_time(r['timestamp'])}  (score {1 - r['distance']:.2f})"
                for r in results
            ]
            choice = st.radio("Jump to", labels, index=0, label_visibility="collapsed")
            selected = results[labels.index(choice)]
            start = selected["timestamp"]
            st.info(selected["caption"])
            with st.expander("All retrieved frames"):
                for r in sorted(results, key=lambda r: r["timestamp"]):
                    st.markdown(f"**{fmt_time(r['timestamp'])}**: {r['caption']}")
        elif query:
            st.write("No matches found.")
        else:
            st.write("Enter a query to find a moment in the video.")

    with col_video:
        st.video(VIDEO_PATH, start_time=start)
        if results:
            st.caption(f"Playing from {fmt_time(start)}")


if __name__ == "__main__":
    main()
