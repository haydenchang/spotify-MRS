import os, json, joblib, re
from pathlib import Path
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers.util import normalize_embeddings

from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title = "Song Recommender (ML + NLP)", layout = "wide")

st.set_page_config(page_title="Spotify Recommender (ML + NLP)", layout="wide")

CWD = Path(os.getcwd())
ROOT = Path(os.path.abspath(os.path.join(CWD, ".."))) if (CWD.name in {"app", "scripts", "EDA"}) else CWD

DATA_CSV   = ROOT / "data" / "clean_data" / "spotify_features_with_info.csv"
MODELS_DIR = ROOT / "models"
ART_DIR    = ROOT / "artifacts"
EMB_DIR    = ART_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

SCALER_P   = MODELS_DIR / "scaler.joblib"
EMB_NPY_P  = EMB_DIR / "title_embeddings_minilm.npy"
EMB_META_P = EMB_DIR / "embeddings_meta.json"

def normalize_title(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    replacements = {
        "–": "-", "—": "-", "’": "'", "“": '"', "”": '"',
        " (feat.": " feat ", "(feat.": " feat ", " feat.": " feat ",
        " (live)": " live", "(live)": " live"
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return " ".join(s.split())

def _tok(s: str):
    return re.findall(r"[a-z0-9]+", s.lower()) if isinstance(s, str) else []

ANTONYMS = {
    "happy": {"sad", "melancholy", "gloomy", "downbeat"},
    "sad": {"happy", "cheerful", "uplifting"},
    "chill": {"hype", "intense", "aggressive"},
    "upbeat": {"downbeat"},
    "calm": {"angry", "aggressive"},
    "dark": {"bright", "uplifting", "happy"},
    "bright": {"dark", "gloomy"},
}

@st.cache_resource(show_spinner=False)
def load_df_and_scaler():
    df = pd.read_csv(DATA_CSV)
    scaler = joblib.load(SCALER_P)

    name_col = "track_name" if "track_name" in df.columns else None
    artist_col = "artist_name" if "artist_name" in df.columns else None

    exclude = {"popularity", "popular_flag", "track_id"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X_audio = df[num_cols].copy()
    X_audio_scaled = scaler.transform(X_audio)

    titles_norm = df[name_col].fillna("").apply(normalize_title) if name_col else pd.Series([""] * len(df))
    artists_norm = df[artist_col].fillna("").apply(normalize_title) if artist_col else pd.Series([""] * len(df))
    df["_title_norm"] = titles_norm
    df["_artist_norm"] = artists_norm

    return df, scaler, name_col, artist_col, num_cols, X_audio_scaled

@st.cache_resource(show_spinner=False)
def load_encoder(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def build_or_load_embeddings(df, name_col):
    if EMB_NPY_P.exists() and EMB_META_P.exists():
        E_text = np.load(EMB_NPY_P)
        with open(EMB_META_P, "r") as f:
            meta = json.load(f)
        return E_text, meta

    encoder = load_encoder()
    titles = df[name_col].fillna("").apply(normalize_title).tolist() if name_col else ["" for _ in range(len(df))]
    E_text = encoder.encode(titles, normalize_embeddings = True, show_progress_bar = True)
    E_text = np.asarray(E_text)
    np.save(EMB_NPY_P, E_text)
    meta = {"model": "all-MiniLM-L6-v2", "normalized": True, "shape": E_text.shape}
    with open(EMB_META_P, "w") as f:
        json.dump(meta, f, indent = 2)
    return E_text, meta

@st.cache_data(show_spinner=False)
def pca_project(X, n_components = 2, random_state = 42):
    pca = PCA(n_components = n_components, random_state = random_state)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    return Z, var

def _keyword_bonus_and_penalty(query_tokens, title_tokens):
    qt = set(query_tokens)
    tt = set(title_tokens)
    overlap = qt & tt
    bonus = min(len(overlap), 4) * 0.02
    penalty = 0.0
    for q in qt:
        if q in ANTONYMS and (ANTONYMS[q] & tt):
            penalty += 0.04
    return bonus, penalty, sorted(list(overlap))[:4]

def search_text(df, titles_norm, encoder, E_text, query: str, top_k: int = 10, w_cos: float = 0.8, w_kw: float = 0.2, w_pen: float = 0.2):
    q_norm = normalize_title(query)
    q_vec = encoder.encode([q_norm], normalize_embeddings = True)
    cos = cosine_similarity(q_vec, E_text)[0]
    q_tokens = _tok(q_norm)
    prelim_k = max(top_k * 5, 50)
    prelim_index = np.argpartition(-cos, range(min(prelim_k, len(cos))))[:prelim_k]

    bonuses = np.zeros_like(cos)
    penalties = np.zeros_like(cos)
    overlaps = [""] * len(cos)

    for i in prelim_index:
        t_tokens = _tok(titles_norm.iloc[i])
        b, p, ov = _keyword_bonus_and_penalty(q_tokens, t_tokens)
        bonuses[i] = b
        penalties[i] = p
        overlaps[i] = ", ".join(ov)

    final = (w_cos * cos) - (w_pen * penalties) + (w_kw * bonuses)

    top_index = np.argpartition(-final, range(min(top_k, len(final))))[:top_k]
    top_index = top_index[np.argsort(-final[top_index])]

    cols = [c for c in ["track_name", "artist_name", "popularity"] if c in df.columns]
    out = df.iloc[top_index][cols].copy()
    out.insert(0, "score", final[top_index].round(4))
    out.insert(1, "similarity", cos[top_index].round(4))
    out.insert(2, "kw_bonus", bonuses[top_index].round(4))
    out.insert(3, "kw_penalty", penalties[top_index].round(4))
    out["why_matched_tokens"] = [overlaps[i] for i in top_index]
    return out.reset_index(drop=True)

def get_track_index(df, title_part: str, artist_part: str = ""):
    t = normalize_title(title_part)
    a = normalize_title(artist_part)
    mask = df["_title_norm"].str.contains(t, case = False, regex = False)
    if a:
        mask &= df["_artist_norm"].str.contains(a, case = False, regex = False)
    index = np.where(mask.values)[0]
    return int(index[0]) if len(index) else None

def topk_neighbors_fused(df, X_fused, seed_index: int, top_k: int = 10):
    sims = cosine_similarity(X_fused[seed_index : seed_index + 1], X_fused)[0]
    sims[seed_index] = -np.inf
    top_index = np.argpartition(-sims, range(min(top_k, len(sims))))[:top_k]
    top_index = top_index[np.argsort(-sims[top_index])]
    cols = [c for c in ["track_name", "artist_name", "popularity"] if c in df.columns]
    out = df.iloc[top_index][cols].copy()
    out.insert(0, "similarity", sims[top_index].round(4))
    return out.reset_index(drop=True), top_index

st.title("Song Recommender - ML + NLP")

df, scaler, name_col, artist_col, num_cols, X_audio_scaled = load_df_and_scaler()
encoder = load_encoder()
E_text, meta = build_or_load_embeddings(df, name_col)

X_fused = np.hstack([E_text, X_audio_scaled])
tab1, tab2 = st.tabs(["Search (NLP)", "Similar by Track"])

with tab1:
    st.subheader("Semantic Search (type what you want)")
    q = st.text_input("Describe the vibe (e.g., 'chill upbeat', 'happy songs for studying')", "")
    colA, colB, colC = st.columns(3)
    with colA:
        top_k = st.slider("Top K", 5, 30, 10)
    with colB:
        w_kw = st.slider("Keyword bonus weight", 0.0, 1.0, 0.2, 0.05)
    with colC:
        w_pen = st.slider("Antonym penalty weight", 0.0, 1.0, 0.2, 0.05)
    st.caption("Hybrid score = 0.8 x cosine - w_pen x penalty + w_kw x bonus")

    if st.button("Search", type = "primary", use_container_width = True):
        if not q.strip():
            st.warning("Enter a query to search.")
        else:
            results = search_text(df, df["_title_norm"], encoder, E_text, q, top_k = top_k, w_kw = w_kw, w_pen = w_pen)
            st.dataframe(results, use_container_width = True)

with tab2:
    st.subheader("Find similar songs to a seed track (fused: text + audio)")
    col1, col2 = st.columns(2)
    with col1:
        t_in = st.text_input("Track (partial ok)", "Ghost Town")
    with col2:
        a_in = st.text_input("Artist (optional)", "Kanye")
    st.markdown("**Preference Weights (audio)** - bump features you care about")
    c1, c2, c3 = st.columns(3)
    with c1:
        w_dance = st.slider("danceability x", 0.5, 2.0, 1.0, 0.05)
    with c2:
        w_val = st.slider("valence x", 0.5, 2.0, 1.0, 0.05)
    with c3:
        w_energy = st.slider("energy x", 0.5, 2.0, 1.0, 0.05)
    audio_weights = np.ones(X_audio_scaled.shape[1], dtype = float)
    if "danceability" in num_cols:
        audio_weights[num_cols.index("danceability")] = w_dance
    if "valence" in num_cols:
        audio_weights[num_cols.index("valence")] = w_val
    if "energy" in num_cols:
        audio_weights[num_cols.index("energy")] = w_energy

    X_fused_weighted = np.hstack([E_text, X_audio_scaled * audio_weights])

    colL, colR = st.columns([1, 1])
    with colL:
        show_labels = st.checkbox("Label top-5 neighbors on PCA", value = True)
    with colR:
        topk = st.slider("Neighbors (K)", 5, 30, 10, 1)

    if st.button("Find Similar", type = "primary", use_container_width = True):
        seed_index = get_track_index(df, t_in, a_in)
        if seed_index is None:
            st.error("Seed track not found. Try a shorter fragment or remove the artist filter.")
        else:
            st.success(f"Seed: {df.loc[seed_index, name_col]} - {df.loc[seed_index, artist_col]}")
            results, neighbor_index = topk_neighbors_fused(df, X_fused_weighted, seed_index, top_k = 10)
            st.session_state["seed_index"] = seed_index
            st.session_state["neighbor_index"] = neighbor_index
            st.session_state["X_fused_weighted_shape"] = X_fused_weighted.shape
            st.dataframe(results, use_container_width = True)
    if "seed_index" in st.session_state and "neighbor_index" in st.session_state:
        if st.button("Show PCA map", type = "secondary", use_container_width = True):
            seed_index = st.session_state["seed_index"]
            neighbor_index = st.session_state["neighbor_index"]
            Z, var = pca_project(X_fused_weighted, n_components = 2, random_state = 42)
            fig = plt.figure(figsize = (7.5, 6))
            plt.scatter(Z[:, 0], Z[:, 1], s = 6, alpha = 0.12, label = "All tracks")
            plt.scatter(Z[neighbor_index, 0], Z[neighbor_index, 1], s = 28, alpha = 0.9, label = "Neighbors")
            plt.scatter(Z[seed_index, 0], Z[seed_index, 1], s = 120, marker = "*", label = "Seed")

            if show_labels:
                top5 = neighbor_index[:5]
                for i in top5:
                    label = f"{df.loc[i, name_col]} - {df.loc[i, artist_col]}"
                    plt.text(Z[i, 0] + 0.02, Z[i, 1] + 0.02, label[:40], fontsize = 8)

            plt.title(f"PCA (weighted fused space) - Seed & Neighbors\nPC1 {var[0]*100:.1f}% • PC2 {var[1]*100:.1f}%")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend(loc = "best")
            plt.tight_layout
            st.pyplot(fig)
st.caption("Tip: This app uses sentence embeddings for semantic search and fuses them with scaled audio features for similarity.")