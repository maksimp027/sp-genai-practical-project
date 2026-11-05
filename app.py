"""
Streamlit Web Application for Multimodal RAG on The Batch (DeepLearning.AI).
Enables semantic search, multimodal retrieval of articles and images,
generative answering with citations, and evaluation metrics visualization.
"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.indexer import MultimodalIndex
from src.rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="The Batch — Multimodal RAG",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, academic look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .article-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.4rem;
    }
    .badge-score {
        background-color: #DCFCE7;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_index():
    return MultimodalIndex(db_path=Path("data/chroma_db"))


index = get_index()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG Settings")

    backend_choice = st.selectbox(
        "LLM Backend",
        ["Gemini", "Ollama (Local)", "OpenAI", "Extractive Fallback"],
        index=0,
        help="Select the generation provider. Gemini supports native multimodal vision.",
    )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if backend_choice == "Gemini":
        gemini_api_input = st.text_input(
            "Gemini API Key",
            value=gemini_key,
            type="password",
            help="Enter key or set GEMINI_API_KEY in .env",
        )
        gemini_model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
        )
    elif backend_choice == "Ollama (Local)":
        ollama_url = st.text_input("Ollama Host", value="http://localhost:11434")
        ollama_model = st.selectbox(
            "Local Model",
            ["qwen2.5-coder:7b", "gemma2:9b", "llama3.2:3b", "deepseek-coder-v2:16b"],
            index=0,
        )
    elif backend_choice == "OpenAI":
        openai_api_input = st.text_input(
            "OpenAI API Key",
            value=openai_key,
            type="password",
            help="Enter key or set OPENAI_API_KEY in .env",
        )
        openai_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

    top_k = st.slider("Retrieved Articles (Top-K)", min_value=2, max_value=8, value=4)

    st.markdown("---")
    st.subheader("📚 Dataset Stats")
    articles_file = Path("data/the_batch_articles.json")
    if articles_file.exists():
        with open(articles_file, "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
        total_arts = len(raw_articles)
        images_dir = Path("data/images")
        total_imgs = len(list(images_dir.glob("*.*"))) if images_dir.exists() else 0
        st.write(f"• **Articles:** {total_arts}")
        st.write(f"• **Images Cached:** {total_imgs}")
        st.write(f"• **Vector Docs:** {index.text_col.count()}")
    else:
        st.warning("No articles scraped yet. Run `python -m src.scraper`.")

    if st.button("🔄 Rebuild Vector Index"):
        with st.spinner("Rebuilding ChromaDB index..."):
            index.index_data(reset=True)
            st.success("Vector index successfully rebuilt!")
            st.rerun()

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-header">The Batch — Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal retrieval and synthesis system across DeepLearning.AI news & research articles</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Q&A & Retrieval", "🖼️ Visual Media Search", "📊 System Evaluation"])

# TAB 1: Q&A and Retrieval
with tab1:
    col_q, col_btn = st.columns([5, 1])

    # Sample query chips
    sample_queries = [
        "How are video models being used to steer robots?",
        "What hardware devices are developers using for local AI models?",
        "What are the recent updates to Anthropic's Claude family?",
        "How did DeepSeek Flash compare against Pro?",
    ]

    selected_sample = st.selectbox("Quick query examples:", ["-- Choose a sample question --"] + sample_queries)
    default_text = "" if selected_sample.startswith("--") else selected_sample

    with col_q:
        user_query = st.text_input("Enter your question or search query:", value=default_text, placeholder="e.g. What are the latest developments in AI agents?")
    with col_btn:
        st.write("")
        st.write("")
        submit = st.button("Search & Answer", type="primary", use_container_width=True)

    if submit and user_query.strip():
        with st.spinner("Retrieving multimodal context and generating answer..."):
            # Setup RAG engine with chosen settings
            active_backend = "extractive"
            g_key = gemini_key
            o_key = openai_key

            if backend_choice == "Gemini":
                active_backend = "gemini"
                g_key = gemini_api_input if "gemini_api_input" in locals() and gemini_api_input else gemini_key
            elif backend_choice == "Ollama (Local)":
                active_backend = "ollama"
            elif backend_choice == "OpenAI":
                active_backend = "openai"
                o_key = openai_api_input if "openai_api_input" in locals() and openai_api_input else openai_key
            else:
                active_backend = "extractive"

            engine = RAGEngine(
                index=index,
                gemini_api_key=g_key,
                openai_api_key=o_key,
                ollama_model=ollama_model if backend_choice == "Ollama (Local)" else "qwen2.5-coder:7b",
            )

            result = engine.answer_query(
                query=user_query,
                top_k=top_k,
                backend=active_backend,
                gemini_model=gemini_model if backend_choice == "Gemini" else "gemini-2.5-flash",
                openai_model=openai_model if backend_choice == "OpenAI" else "gpt-4o-mini",
            )

        # 1. Synthesized Response Card
        st.subheader("💡 Synthesized Answer")
        st.markdown(result["answer"])

        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Model:** {result['model_used']}")
        c2.caption(f"**Retrieval:** {result['retrieval_time'] * 1000:.1f} ms")
        c3.caption(f"**Generation:** {result['generation_time']:.2f} s")

        st.markdown("---")

        # 2. Associated Images and Retrieved Articles
        st.subheader("🖼️ Associated Visual Media")
        gallery = result.get("gallery_images", [])

        if gallery:
            img_cols = st.columns(min(len(gallery), 3))
            for i, img in enumerate(gallery[:3]):
                col = img_cols[i % len(img_cols)]
                with col:
                    local_p = img.get("local_path")
                    if local_p and Path(local_p).exists():
                        st.image(local_p, caption=img.get("title", "Article figure"), use_container_width=True)
                        if img.get("url"):
                            st.caption(f"[Read Article]({img['url']})")
        else:
            st.info("No specific figures associated with this query context.")

        st.subheader(f"📑 Retrieved Source Articles ({len(result.get('articles', []))})")
        for i, art in enumerate(result.get("articles", []), start=1):
            with st.expander(f"[{i}] {art['title']} — (Score: {art['best_score']:.2f})", expanded=(i == 1)):
                st.markdown(f"**Date:** {art.get('date', 'N/A')} | **Tags:** `{art.get('tags', 'N/A')}`")
                st.markdown(f"**Source URL:** [{art.get('url')}]({art.get('url')})")

                # Chunks
                for c_idx, c_text in enumerate(art.get("chunks", [])):
                    st.markdown(f"> {c_text}")

                # Article thumbnail if present
                if art.get("local_images"):
                    p0 = art["local_images"][0]
                    if Path(p0).exists():
                        st.image(p0, width=320, caption=f"Figure: {art['title']}")

# TAB 2: Visual Media Search
with tab2:
    st.subheader("Explore Visual Database")
    st.markdown("Search directly for article figures, infographics, and illustrations based on visual concepts.")

    v_query = st.text_input("Search visual media by topic or keyword:", placeholder="e.g. robot arm, neural network architecture, data center")
    v_top_k = st.slider("Max images to display", 2, 8, 4, key="v_slider")

    if st.button("Search Media", key="v_btn") or v_query:
        query_to_use = v_query.strip() or "machine learning"
        img_results = index.search_images(query_to_use, top_k=v_top_k)

        if img_results:
            cols = st.columns(2)
            for idx, item in enumerate(img_results):
                with cols[idx % 2]:
                    lp = item.get("local_path")
                    if lp and Path(lp).exists():
                        st.image(lp, use_container_width=True)
                        st.markdown(f"**{item.get('title')}**")
                        st.caption(f"Score: `{item.get('score', 0):.2f}` | [View Article]({item.get('url')})")
        else:
            st.warning("No media found matching this query.")

# TAB 3: System Evaluation
with tab3:
    st.subheader("System Evaluation & Benchmark Metrics")
    st.markdown("Automated evaluation of retrieval quality, Mean Reciprocal Rank (MRR), and multimodal grounding.")

    eval_json_path = Path("data/eval_results.json")
    chart_path = Path("data/eval_metrics.png")

    c_run, c_space = st.columns([2, 5])
    with c_run:
        run_eval = st.button("🚀 Run Live Benchmark Evaluation", type="primary")

    if run_eval:
        from src.evaluate import evaluate_retrieval, plot_metrics
        with st.spinner("Executing retrieval benchmark over test suite..."):
            report = evaluate_retrieval(index, top_k=5)
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            plot_metrics(report, output_path=chart_path)
            st.success("Benchmark completed!")

    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Hit@1", f"{eval_data['hit_rate_at_1'] * 100:.1f}%")
        m2.metric("Hit@3", f"{eval_data['hit_rate_at_3'] * 100:.1f}%")
        m3.metric("Hit@5", f"{eval_data['hit_rate_at_5'] * 100:.1f}%")
        m4.metric("MRR", f"{eval_data['mean_reciprocal_rank_mrr']:.3f}")
        m5.metric("Avg Latency", f"{eval_data['avg_retrieval_latency_ms']:.1f} ms")

        if chart_path.exists():
            st.image(str(chart_path), caption="Retrieval Evaluation Summary", use_container_width=True)

        st.markdown("### Benchmark Query Results")
        query_details = eval_data.get("query_details", [])
        for q in query_details:
            hit_color = "🟢" if q["hit_at_1"] else ("🟡" if q["hit_at_5"] else "🔴")
            with st.expander(f"{hit_color} {q['query']} (MRR: {q['reciprocal_rank']})"):
                st.write(f"**Category:** {q.get('category')}")
                st.write(f"**First match rank:** {q.get('first_match_rank')}")
                st.write(f"**Expected targets:** `{q.get('target_slugs')}`")
                st.write(f"**Retrieved:** `{q.get('retrieved_slugs')}`")
                st.write(f"**Latency:** {q.get('latency_ms')} ms")
    else:
        st.info("Evaluation has not been run yet. Click 'Run Live Benchmark Evaluation' above.")
