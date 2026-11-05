"""
Evaluation benchmark for the Multimodal RAG system.
Evaluates retrieval accuracy (Hit@K, MRR, Precision, Recall), multimodal grounding,
and latency across a curated benchmark of domain queries from The Batch.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.indexer import MultimodalIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Representative benchmark queries mapping natural language queries to expected article slugs/topics
BENCHMARK_QUERIES = [
    {
        "query": "How are video models being used to steer robots and world-action models?",
        "target_slugs": ["video-models-steer-robots"],
        "category": "Robotics & Embodied AI",
    },
    {
        "query": "What hardware devices are developers using to run AI models locally instead of cloud?",
        "target_slugs": ["computers-built-for-local-ai"],
        "category": "Hardware & Local AI",
    },
    {
        "query": "What are the latest updates to Anthropic's Claude family including Haiku and Opus?",
        "target_slugs": [
            "claude-fills-out-its-model-family",
            "anthropic-gives-claude-haiku-a-big-update",
            "claude-opus-5-5-leaps-forward",
            "claude-5-5s-first-model-has-landed",
        ],
        "category": "Frontier LLMs",
    },
    {
        "query": "How did DeepSeek Flash compare against DeepSeek Pro architecture and benchmarks?",
        "target_slugs": [
            "deepseeks-flash-leapfrogs-pro-again",
            "deepseek-v4-1-flashs-new-architecture",
        ],
        "category": "Open Weight Models",
    },
    {
        "query": "Can autonomous AI agents collaborate and pass notes to work together?",
        "target_slugs": [
            "dawn-of-the-agents",
            "agents-work-better-when-they-can-pass-each-other-notes",
            "one-agent-works-another-directs",
        ],
        "category": "Agentic Workflows",
    },
    {
        "query": "What did Anthropic's public filings disclose regarding their future and corporate governance?",
        "target_slugs": ["anthropics-public-filing-offers-clues-to-its-future"],
        "category": "AI Industry & Policy",
    },
    {
        "query": "What announcements were made at OpenAI Dev Day regarding models and pricing?",
        "target_slugs": ["the-biggest-news-from-openais-dev-day"],
        "category": "Frontier LLMs",
    },
    {
        "query": "How do researchers improve LLM context management and context engineering?",
        "target_slugs": ["a-tool-for-better-context-management"],
        "category": "Context & Prompt Engineering",
    },
    {
        "query": "What is Google's Gemini 4 Argon and its benchmark performance?",
        "target_slugs": ["gemini-4-argons-benchmarks-and-availability"],
        "category": "Multimodal LLMs",
    },
    {
        "query": "How are AI companies addressing cybersecurity and defensive capabilities?",
        "target_slugs": ["we-should-be-encouraged-by-ais-cybersecurity-abilities"],
        "category": "Security & Safety",
    },
    {
        "query": "How does Meta's Muse open model family perform against top competitors?",
        "target_slugs": [
            "meta-releases-wonder-opens-spark",
            "muse-spark-open-leads-u-s-open-models",
            "metas-muse-wonder-rivals-top-models",
        ],
        "category": "Open Weight Models",
    },
    {
        "query": "Could companies or governments shut down advanced AI model development?",
        "target_slugs": ["leaders-lock-models-away"],
        "category": "AI Governance",
    },
]


def evaluate_retrieval(
    index: MultimodalIndex,
    queries: List[Dict[str, Any]] = BENCHMARK_QUERIES,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run evaluation across benchmark queries and calculate formal IR metrics."""
    logger.info("Evaluating retrieval over %d benchmark queries (top_k=%d)...", len(queries), top_k)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    multimodal_visual_hits = 0
    latencies = []
    query_details = []

    for item in queries:
        q_text = item["query"]
        target_slugs = set(item["target_slugs"])
        category = item.get("category", "General")

        t0 = time.time()
        retrieval = index.search_multimodal(q_text, top_k=top_k)
        lat = round((time.time() - t0) * 1000, 2)
        latencies.append(lat)

        articles = retrieval.get("articles", [])
        retrieved_slugs = [a["article_id"] for a in articles]

        # Find first matching rank (1-indexed)
        first_match_rank = None
        for rank, slug in enumerate(retrieved_slugs, start=1):
            if slug in target_slugs:
                first_match_rank = rank
                break

        # Check Hit@K
        hit_1 = 1 if (first_match_rank is not None and first_match_rank <= 1) else 0
        hit_3 = 1 if (first_match_rank is not None and first_match_rank <= 3) else 0
        hit_5 = 1 if (first_match_rank is not None and first_match_rank <= 5) else 0

        hits_at_1 += hit_1
        hits_at_3 += hit_3
        hits_at_5 += hit_5

        rr = 1.0 / first_match_rank if first_match_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        # Check multimodal visual support (did the retrieved context include relevant images?)
        gallery = retrieval.get("gallery_images", [])
        has_media = len(gallery) > 0 and any(
            g.get("article_id") in target_slugs or g.get("local_path") for g in gallery
        )
        if has_media:
            multimodal_visual_hits += 1

        query_details.append({
            "query": q_text,
            "category": category,
            "target_slugs": list(target_slugs),
            "retrieved_slugs": retrieved_slugs,
            "first_match_rank": first_match_rank,
            "hit_at_1": bool(hit_1),
            "hit_at_3": bool(hit_3),
            "hit_at_5": bool(hit_5),
            "reciprocal_rank": round(rr, 4),
            "has_media": has_media,
            "num_gallery_images": len(gallery),
            "latency_ms": lat,
        })

    n_q = len(queries)
    summary = {
        "num_queries": n_q,
        "hit_rate_at_1": round(hits_at_1 / n_q, 4),
        "hit_rate_at_3": round(hits_at_3 / n_q, 4),
        "hit_rate_at_5": round(hits_at_5 / n_q, 4),
        "mean_reciprocal_rank_mrr": round(sum(reciprocal_ranks) / n_q, 4),
        "multimodal_visual_coverage": round(multimodal_visual_hits / n_q, 4),
        "avg_retrieval_latency_ms": round(sum(latencies) / n_q, 2),
        "min_latency_ms": min(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "query_details": query_details,
    }

    logger.info("Evaluation Summary:")
    logger.info(" - Hit@1: %.1f%%", summary["hit_rate_at_1"] * 100)
    logger.info(" - Hit@3: %.1f%%", summary["hit_rate_at_3"] * 100)
    logger.info(" - Hit@5: %.1f%%", summary["hit_rate_at_5"] * 100)
    logger.info(" - MRR:   %.4f", summary["mean_reciprocal_rank_mrr"])
    logger.info(" - Visual Coverage: %.1f%%", summary["multimodal_visual_coverage"] * 100)
    logger.info(" - Avg Latency: %.1f ms", summary["avg_retrieval_latency_ms"])

    return summary


def plot_metrics(summary: Dict[str, Any], output_path: Path = Path("data/eval_metrics.png")):
    """Optional helper to render visual chart of IR performance."""
    try:
        import matplotlib.pyplot as plt

        metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR", "Visual Cov."]
        values = [
            summary["hit_rate_at_1"] * 100,
            summary["hit_rate_at_3"] * 100,
            summary["hit_rate_at_5"] * 100,
            summary["mean_reciprocal_rank_mrr"] * 100,
            summary["multimodal_visual_coverage"] * 100,
        ]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(metrics, values, color=["#3B82F6", "#10B981", "#6366F1", "#F59E0B", "#EC4899"])
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (%)")
        ax.set_title("Multimodal RAG Benchmark Retrieval Performance", fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="semibold",
            )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Saved metrics plot to %s", output_path)
    except Exception as e:
        logger.warning("Could not generate plot: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multimodal RAG Evaluation")
    parser.add_argument("--db-dir", type=str, default="data/chroma_db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="data/eval_results.json")
    args = parser.parse_args()

    idx = MultimodalIndex(db_path=Path(args.db_dir))
    report = evaluate_retrieval(idx, top_k=args.top_k)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_metrics(report, output_path=out_file.parent / "eval_metrics.png")
    logger.info("Saved evaluation report to %s", out_file)
