"""
Article and media ingestion pipeline for The Batch (DeepLearning.AI).
Fetches recent news articles, extracts text content, metadata, and associated images.
"""

import argparse
import html
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.deeplearning.ai"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_IMAGES_DIR = DEFAULT_OUTPUT_DIR / "images"

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


def request_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch URL contents with browser-like user agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def get_the_batch_urls(max_urls: int = 100) -> List[str]:
    """Extract article URLs from the main sitemap."""
    logger.info("Fetching sitemap from %s", SITEMAP_URL)
    content = request_url(SITEMAP_URL)
    if not content:
        logger.error("Could not load sitemap")
        return []

    # Extract all the-batch URLs
    raw_urls = re.findall(r"<loc>(https://www\.deeplearning\.ai/the-batch/[^<]+)</loc>", content)
    if not raw_urls:
        raw_urls = re.findall(r"https://www\.deeplearning\.ai/the-batch/[^<\"\s]+", content)

    # Filter out system/listing pages
    excluded_keywords = ["/about", "/tag/", "/category/", "/author/", "/page/"]
    candidate_urls = []
    seen = set()

    for url in raw_urls:
        url = url.strip()
        if any(kw in url for kw in excluded_keywords):
            continue
        if url == f"{BASE_URL}/the-batch" or url == f"{BASE_URL}/the-batch/":
            continue
        if url not in seen:
            seen.add(url)
            candidate_urls.append(url)

    logger.info("Discovered %d candidate article URLs in sitemap", len(candidate_urls))
    return candidate_urls[:max_urls]


def clean_text(raw_html: str) -> str:
    """Remove HTML tags and unescape entities."""
    no_tags = re.sub(r"<[^>]+>", " ", raw_html)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def parse_article(url: str, html_text: str) -> Optional[Dict]:
    """Parse title, published date, tags, body paragraphs, and image links."""
    # Title
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, re.DOTALL)
    if title_match:
        title = clean_text(title_match.group(1))
    else:
        title_meta = re.search(r"<title>(.*?)</title>", html_text)
        title = clean_text(title_meta.group(1)) if title_meta else "Untitled"

    # Remove site suffix
    title = re.sub(r"\s*\|\s*The Batch.*$", "", title).strip()

    # Excerpt / meta description
    desc_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html_text)
    excerpt = clean_text(desc_match.group(1)) if desc_match else ""

    # Publish date
    date_match = re.search(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b)", html_text)
    date_str = date_match.group(1) if date_match else ""

    # Tags
    tags = list(set(re.findall(r"/the-batch/tag/([a-zA-Z0-9_-]+)", html_text)))

    # Images from DeepLearning.AI CDN (charonhub)
    raw_images = re.findall(r"https%3A%2F%2Fcharonhub\.deeplearning\.ai%2Fcontent%2Fimages%2F[^&\"\'\s]+", html_text)
    decoded_images = [urllib.parse.unquote(img) for img in raw_images]

    # Also catch unescaped charonhub images
    direct_images = re.findall(r'https://charonhub\.deeplearning\.ai/content/images/[^"\'\s)]+', html_text)
    all_images = []
    seen_imgs = set()
    for img in decoded_images + direct_images:
        # Ignore tiny icons, badges, avatars if any
        if img not in seen_imgs and not any(skip in img.lower() for skip in ["avatar", "icon", "logo", "badge"]):
            seen_imgs.add(img)
            all_images.append(img)

    # Paragraphs extraction
    # The Batch articles contain narrative paragraphs under main content
    raw_paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, re.DOTALL)
    cleaned_paragraphs = []
    boilerplate_snippets = [
        "enroll in", "new course", "deeplearning.ai pro", "cookie", "privacy policy",
        "subscribe to the batch", "andrew's letter", "for business overview", "all rights reserved"
    ]

    for p in raw_paragraphs:
        txt = clean_text(p)
        if len(txt) < 35:
            continue
        txt_lower = txt.lower()
        if any(b in txt_lower for b in boilerplate_snippets):
            continue
        cleaned_paragraphs.append(txt)

    if not cleaned_paragraphs and not excerpt:
        return None

    full_text = "\n\n".join(cleaned_paragraphs)
    if not full_text:
        full_text = excerpt

    slug = url.rstrip("/").split("/")[-1]

    return {
        "id": slug,
        "url": url,
        "title": title,
        "date": date_str,
        "excerpt": excerpt,
        "tags": tags,
        "content": full_text,
        "paragraphs": cleaned_paragraphs,
        "image_urls": all_images,
    }


def download_image(img_url: str, dest_dir: Path, article_id: str, img_idx: int) -> Optional[str]:
    """Download image to local cache directory and return local file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Determine extension
    ext = ".jpg"
    if ".webp" in img_url.lower():
        ext = ".webp"
    elif ".png" in img_url.lower():
        ext = ".png"

    filename = f"{article_id}_{img_idx}{ext}"
    local_path = dest_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > 1024:  # At least 1KB
                with open(local_path, "wb") as f:
                    f.write(data)
                return str(local_path)
    except Exception as e:
        logger.debug("Could not download image %s: %s", img_url, e)

    return None


def scrape_batch_articles(limit: int = 50, output_dir: Path = DEFAULT_OUTPUT_DIR) -> List[Dict]:
    """Main crawler: fetches URLs, parses contents, downloads images, and stores JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    urls = get_the_batch_urls(max_urls=limit * 2)
    articles = []

    logger.info("Starting ingestion of up to %d articles...", limit)

    for idx, url in enumerate(urls):
        if len(articles) >= limit:
            break

        html_text = request_url(url)
        if not html_text:
            continue

        item = parse_article(url, html_text)
        if not item or len(item["content"]) < 100:
            continue

        # Download associated images
        local_image_paths = []
        for img_i, img_url in enumerate(item["image_urls"][:3]):
            lp = download_image(img_url, images_dir, item["id"], img_i)
            if lp:
                local_image_paths.append(lp)

        item["local_images"] = local_image_paths
        articles.append(item)

        logger.info(
            "[%d/%d] Ingested: '%s' (%d chars, %d images)",
            len(articles),
            limit,
            item["title"][:50],
            len(item["content"]),
            len(local_image_paths),
        )
        time.sleep(0.3)  # Polite crawling delay

    out_file = output_dir / "the_batch_articles.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    logger.info("Successfully saved %d articles to %s", len(articles), out_file)
    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape articles and images from The Batch")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of articles to ingest")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory path")
    args = parser.parse_args()

    scrape_batch_articles(limit=args.limit, output_dir=Path(args.output_dir))
