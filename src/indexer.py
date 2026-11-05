"""
Multimodal database creation and indexing pipeline.
Processes ingested The Batch articles and images, creates embeddings,
and stores them in a persistent ChromaDB vector store.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/the_batch_articles.json")
DEFAULT_DB_DIR = Path("data/chroma_db")
TEXT_COLLECTION_NAME = "the_batch_articles"
IMAGE_COLLECTION_NAME = "the_batch_images"


def get_default_embedding_fn():
    """Default ONNX-based sentence transformer (all-MiniLM-L6-v2) packaged with ChromaDB."""
    return embedding_functions.DefaultEmbeddingFunction()


def chunk_article(
    article: Dict[str, Any],
    max_chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split an article into overlapping chunks suitable for semantic search.
    Preserves article metadata and prepends the headline for contextual grounding.
    """
    content = article.get("content", "").strip()
    title = article.get("title", "Untitled")

    if not content:
        content = article.get("excerpt", "")

    chunks = []
    # If article is relatively concise, keep it whole
    if len(content) <= max_chunk_chars:
        chunks.append({
            "text": f"Title: {title}\n\n{content}",
            "chunk_index": 0,
            "raw_text": content,
        })
        return chunks

    # Split into paragraphs and group them
    paragraphs = article.get("paragraphs", [])
    if not paragraphs:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    current_chunk = []
    current_len = 0
    chunk_idx = 0

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len > max_chunk_chars and current_chunk:
            chunk_body = "\n\n".join(current_chunk)
            chunks.append({
                "text": f"Title: {title}\n\n{chunk_body}",
                "chunk_index": chunk_idx,
                "raw_text": chunk_body,
            })
            chunk_idx += 1

            # Retain small trailing overlap if possible
            if len(current_chunk[-1]) <= overlap_chars:
                current_chunk = [current_chunk[-1], p]
                current_len = len(current_chunk[0]) + p_len
            else:
                current_chunk = [p]
                current_len = p_len
        else:
            current_chunk.append(p)
            current_len += p_len

    if current_chunk:
        chunk_body = "\n\n".join(current_chunk)
        chunks.append({
            "text": f"Title: {title}\n\n{chunk_body}",
            "chunk_index": chunk_idx,
            "raw_text": chunk_body,
        })

    return chunks


class MultimodalIndex:
    """Manages ChromaDB vector collections for textual and visual data."""

    def __init__(self, db_path: Path = DEFAULT_DB_DIR):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.embed_fn = get_default_embedding_fn()

        self.text_col = self.client.get_or_create_collection(
            name=TEXT_COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"description": "Text chunks of The Batch news articles"},
        )

        self.image_col = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"description": "Visual media metadata and contexts from The Batch"},
        )

    def index_data(self, data_path: Path = DEFAULT_DATA_PATH, reset: bool = False):
        """Index all articles and associated media into the vector store."""
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}. Run scraper first.")

        with open(data_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        logger.info("Loaded %d articles from %s", len(articles), data_path)

        if reset:
            try:
                self.client.delete_collection(TEXT_COLLECTION_NAME)
                self.client.delete_collection(IMAGE_COLLECTION_NAME)
            except Exception:
                pass
            self.text_col = self.client.create_collection(
                name=TEXT_COLLECTION_NAME,
                embedding_function=self.embed_fn,
            )
            self.image_col = self.client.create_collection(
                name=IMAGE_COLLECTION_NAME,
                embedding_function=self.embed_fn,
            )

        text_docs, text_metas, text_ids = [], [], []
        image_docs, image_metas, image_ids = [], [], []

        for article in articles:
            art_id = article.get("id", "")
            title = article.get("title", "")
            date_str = article.get("date", "")
            url = article.get("url", "")
            excerpt = article.get("excerpt", "")
            tags = ", ".join(article.get("tags", []))
            local_imgs = article.get("local_images", [])
            img_urls = article.get("image_urls", [])

            # Index text chunks
            chunks = chunk_article(article)
            for c in chunks:
                doc_id = f"{art_id}_c{c['chunk_index']}"
                primary_img = local_imgs[0] if local_imgs else ""

                text_docs.append(c["text"])
                text_ids.append(doc_id)
                text_metas.append({
                    "article_id": art_id,
                    "title": title,
                    "date": date_str,
                    "url": url,
                    "tags": tags,
                    "chunk_index": c["chunk_index"],
                    "primary_image": primary_img,
                    "local_images": json.dumps(local_imgs),
                    "image_urls": json.dumps(img_urls[:3]),
                })

            # Index associated images with multimodal context
            for i, local_img in enumerate(local_imgs):
                img_id = f"img_{art_id}_{i}"
                remote_url = img_urls[i] if i < len(img_urls) else ""
                # Context combines headline, excerpt, and article focus for cross-modal search
                context = (
                    f"Illustration for AI news article: '{title}'. "
                    f"Summary: {excerpt}. "
                    f"Topics: {tags}."
                )
                image_docs.append(context)
                image_ids.append(img_id)
                image_metas.append({
                    "article_id": art_id,
                    "title": title,
                    "date": date_str,
                    "url": url,
                    "local_path": local_img,
                    "remote_url": remote_url,
                    "image_index": i,
                })

        # Upsert in batches
        BATCH_SIZE = 100
        logger.info("Upserting %d text chunks...", len(text_docs))
        for i in range(0, len(text_docs), BATCH_SIZE):
            self.text_col.upsert(
                documents=text_docs[i:i + BATCH_SIZE],
                metadatas=text_metas[i:i + BATCH_SIZE],
                ids=text_ids[i:i + BATCH_SIZE],
            )

        logger.info("Upserting %d image entries...", len(image_docs))
        for i in range(0, len(image_docs), BATCH_SIZE):
            self.image_col.upsert(
                documents=image_docs[i:i + BATCH_SIZE],
                metadatas=image_metas[i:i + BATCH_SIZE],
                ids=image_ids[i:i + BATCH_SIZE],
            )

        logger.info(
            "Indexing complete. Database has %d text chunks and %d images.",
            self.text_col.count(),
            self.image_col.count(),
        )

    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant article text chunks."""
        results = self.text_col.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results or not results["documents"]:
            return hits

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for doc, meta, dist, hit_id in zip(docs, metas, distances, ids):
            # Convert cosine distance to approximate similarity score
            sim = max(0.0, 1.0 - dist)
            local_imgs = json.loads(meta.get("local_images", "[]"))
            img_urls = json.loads(meta.get("image_urls", "[]"))

            hits.append({
                "id": hit_id,
                "score": round(sim, 4),
                "distance": round(dist, 4),
                "text": doc,
                "title": meta.get("title", ""),
                "date": meta.get("date", ""),
                "url": meta.get("url", ""),
                "tags": meta.get("tags", ""),
                "article_id": meta.get("article_id", ""),
                "primary_image": meta.get("primary_image", ""),
                "local_images": local_imgs,
                "image_urls": img_urls,
            })
        return hits

    def search_images(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve most relevant visual media based on query context."""
        results = self.image_col.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results or not results["documents"]:
            return hits

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for doc, meta, dist, hit_id in zip(docs, metas, distances, ids):
            sim = max(0.0, 1.0 - dist)
            hits.append({
                "id": hit_id,
                "score": round(sim, 4),
                "distance": round(dist, 4),
                "context": doc,
                "title": meta.get("title", ""),
                "date": meta.get("date", ""),
                "url": meta.get("url", ""),
                "local_path": meta.get("local_path", ""),
                "remote_url": meta.get("remote_url", ""),
                "article_id": meta.get("article_id", ""),
            })
        return hits

    def search_multimodal(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Unified multimodal retrieval: combines textual passages and ranked images,
        deduplicating articles and aggregating associated media.
        """
        text_hits = self.search_text(query, top_k=top_k)
        image_hits = self.search_images(query, top_k=top_k)

        # Collect and deduplicate unique articles
        seen_articles = {}
        for h in text_hits:
            art_id = h["article_id"]
            if art_id not in seen_articles:
                seen_articles[art_id] = {
                    "article_id": art_id,
                    "title": h["title"],
                    "date": h["date"],
                    "url": h["url"],
                    "tags": h["tags"],
                    "primary_image": h["primary_image"],
                    "local_images": h["local_images"],
                    "image_urls": h["image_urls"],
                    "chunks": [h["text"]],
                    "best_score": h["score"],
                }
            else:
                seen_articles[art_id]["chunks"].append(h["text"])
                if h["score"] > seen_articles[art_id]["best_score"]:
                    seen_articles[art_id]["best_score"] = h["score"]

        # Collect gallery images with verified existence
        gallery = []
        seen_img_paths = set()
        for img in image_hits:
            lp = img.get("local_path")
            if lp and Path(lp).exists() and lp not in seen_img_paths:
                seen_img_paths.add(lp)
                gallery.append(img)

        # If some articles have images not in gallery, also attach them
        for art in seen_articles.values():
            for p in art["local_images"]:
                if p and Path(p).exists() and p not in seen_img_paths and len(gallery) < 6:
                    seen_img_paths.add(p)
                    gallery.append({
                        "id": f"art_{art['article_id']}",
                        "score": art["best_score"],
                        "title": art["title"],
                        "date": art["date"],
                        "url": art["url"],
                        "local_path": p,
                        "remote_url": art["image_urls"][0] if art["image_urls"] else "",
                        "context": f"Image from article: {art['title']}",
                    })

        return {
            "query": query,
            "text_hits": text_hits,
            "articles": list(seen_articles.values()),
            "gallery_images": gallery,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index The Batch articles into ChromaDB")
    parser.add_argument("--data", type=str, default="data/the_batch_articles.json")
    parser.add_argument("--db-dir", type=str, default="data/chroma_db")
    parser.add_argument("--reset", action="store_true", help="Recreate collections")
    args = parser.parse_args()

    index = MultimodalIndex(db_path=Path(args.db_dir))
    index.index_data(data_path=Path(args.data), reset=args.reset)
