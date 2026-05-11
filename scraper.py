"""
SHL Catalog Scraper
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Saves to catalog.json for use by the recommender.
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
CATALOG_FILE = Path("catalog.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def type_types_to_str(codes: list[str]) -> str:
    return ", ".join(TYPE_MAP.get(c, c) for c in codes)


def _parse_page(html: str, page_url: str) -> list[dict]:
    """Parse one catalog page and return list of assessment dicts."""
    soup = BeautifulSoup(html, "lxml")
    results = []

    # The catalog table has rows for each assessment
    table = soup.find("table")
    if not table:
        return results

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Cell 0: assessment name + link
        name_cell = cells[0]
        link_tag = name_cell.find("a")
        if not link_tag:
            continue
        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        if href.startswith("/"):
            url = BASE_URL + href
        elif href.startswith("http"):
            url = href
        else:
            url = page_url

        # Cell 1: Remote Testing (Yes/No, indicated by icon)
        remote_cell = cells[1]
        remote = bool(remote_cell.find(class_=lambda c: c and "yes" in c.lower())) if remote_cell else False

        # Cell 2: Adaptive/IRT
        adaptive_cell = cells[2]
        adaptive = bool(adaptive_cell.find(class_=lambda c: c and "yes" in c.lower())) if adaptive_cell else False

        # Cell 3: Test Type (letter codes)
        type_cell = cells[3]
        type_text = type_cell.get_text(strip=True)
        test_types = [ch for ch in type_text if ch in TYPE_MAP]

        results.append({
            "name": name,
            "url": url,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type": type_types_to_str(test_types),
            "test_type_codes": test_types,
            "description": "",  # filled by detail fetch
            "job_levels": [],
            "languages": [],
            "duration_minutes": None,
        })

    return results


def _fetch_detail(client: httpx.Client, item: dict) -> dict:
    """Fetch individual assessment detail page and enrich item."""
    try:
        resp = client.get(item["url"], timeout=15)
        if resp.status_code != 200:
            return item
        soup = BeautifulSoup(resp.text, "lxml")

        # Description: first substantial paragraph in main content
        main = soup.find("main") or soup.find("article") or soup.body
        if main:
            paras = main.find_all("p")
            for p in paras:
                text = p.get_text(strip=True)
                if len(text) > 60:
                    item["description"] = text
                    break

        # Look for job levels, duration in page text
        page_text = soup.get_text(" ", strip=True)

        # Duration
        dur_match = re.search(
            r"(?:Approximate Completion Time|Duration)[^\d]*(\d+)[^m]*minutes?",
            page_text, re.IGNORECASE
        )
        if dur_match:
            item["duration_minutes"] = int(dur_match.group(1))

        # Job levels listed
        levels = ["Director", "Entry-Level", "Executive", "Front Line Manager",
                  "General Population", "Graduate", "Manager", "Mid-Professional",
                  "Professional Individual Contributor", "Supervisor"]
        item["job_levels"] = [lvl for lvl in levels if lvl in page_text]

    except Exception as e:
        logger.warning(f"Detail fetch failed for {item['name']}: {e}")

    return item


def scrape_catalog(max_pages: int = 30) -> list[dict]:
    """Scrape all Individual Test Solutions from SHL catalog."""
    all_items: list[dict] = []
    seen_names: set[str] = set()

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        start = 0
        page_size = 12  # SHL default

        for _ in range(max_pages):
            url = (
                f"{CATALOG_URL}?action_doFilteringForm=Search"
                f"&f=1&type=1&start={start}"
            )
            logger.info(f"Fetching catalog page start={start}: {url}")
            try:
                resp = client.get(url, timeout=20)
                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code} for {url}")
                    break
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                break

            items = _parse_page(resp.text, url)
            if not items:
                logger.info("No items found on page, stopping.")
                break

            new_items = [i for i in items if i["name"] not in seen_names]
            if not new_items:
                logger.info("No new items, stopping.")
                break

            for item in new_items:
                seen_names.add(item["name"])
                enriched = _fetch_detail(client, item)
                all_items.append(enriched)
                time.sleep(0.3)

            # Check for next page link
            soup = BeautifulSoup(resp.text, "lxml")
            next_link = soup.find("a", string=lambda t: t and "next" in t.lower())
            if not next_link:
                # also check pagination
                pagination = soup.find("nav", class_=lambda c: c and "pagination" in (c or "").lower())
                if not pagination:
                    break

            start += page_size
            time.sleep(0.5)

    logger.info(f"Scraped {len(all_items)} assessments total.")
    return all_items


def load_or_scrape_catalog() -> list[dict]:
    """Load catalog from disk if fresh, otherwise scrape."""
    if CATALOG_FILE.exists():
        try:
            with open(CATALOG_FILE) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 10:
                logger.info(f"Loaded {len(data)} assessments from {CATALOG_FILE}")
                return data
        except Exception:
            pass

    logger.info("Scraping SHL catalog...")
    items = scrape_catalog()

    if not items:
        logger.warning("Scraping returned 0 items, using fallback catalog.")
        items = get_fallback_catalog()

    with open(CATALOG_FILE, "w") as f:
        json.dump(items, f, indent=2)
    logger.info(f"Saved {len(items)} assessments to {CATALOG_FILE}")
    return items


def get_fallback_catalog() -> list[dict]:
    """
    Comprehensive fallback catalog of SHL Individual Test Solutions.
    Covers all major assessment types: A, B, C, D, E, K, P, S.
    URLs are from the official SHL product catalog.
    """
    return [
        # ── COGNITIVE / ABILITY (A) ──────────────────────────────────────────
        {
            "name": "Verify - Numerical Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/verify-numerical-reasoning/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to make correct decisions or inferences from numerical or statistical data.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 25,
        },
        {
            "name": "Verify - Verbal Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/verify-verbal-reasoning/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to understand and evaluate the logic of various kinds of arguments in verbal form.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 19,
        },
        {
            "name": "Verify - Inductive Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/verify-inductive-reasoning/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to draw inferences and understand the relationships between various concepts independent of acquired knowledge.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Entry-Level"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Dutch"],
            "duration_minutes": 25,
        },
        {
            "name": "Verify - Deductive Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/verify-deductive-reasoning/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to draw logical conclusions from information provided, identify strengths and weaknesses of arguments, and complete scenarios using incomplete information.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Dutch", "Russian"],
            "duration_minutes": 18,
        },
        {
            "name": "Verify G+ (General Ability)",
            "url": "https://www.shl.com/products/product-catalog/view/verify-g-plus-general-ability/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Adaptive measure of general cognitive ability combining numerical, inductive, and deductive reasoning. Adaptive/IRT-based for precision across all job levels.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Director", "Executive"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 36,
        },
        {
            "name": "Verify - Verbal Ability (Next Generation)",
            "url": "https://www.shl.com/products/product-catalog/view/verify-verbal-ability-next-generation/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Adaptive verbal ability test measuring reading comprehension, tone interpretation, main idea identification, and author intent. Appropriate for all job levels.",
            "job_levels": ["General Population", "Graduate", "Executive", "Director", "Entry-Level", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor", "Front Line Manager"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 18,
        },
        {
            "name": "SHL Verify Interactive – Numerical Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Mobile-first interactive numerical reasoning test with drag-and-drop functionality. Adaptive IRT-based scoring.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "English (USA)", "German", "French", "Dutch", "Spanish"],
            "duration_minutes": 18,
        },
        {
            "name": "SHL Verify Interactive – Inductive Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-inductive-reasoning/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Mobile-first interactive inductive reasoning assessment with adaptive question difficulty.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Entry-Level"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 18,
        },
        {
            "name": "SHL Verify Interactive – Deductive Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-deductive-reasoning/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Mobile-first interactive deductive reasoning test. Available in 30+ languages with adaptive IRT scoring.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Japanese", "Chinese Simplified"],
            "duration_minutes": 18,
        },
        {
            "name": "Numerical Ability - Next Generation",
            "url": "https://www.shl.com/products/product-catalog/view/numerical-ability-next-generation/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Adaptive numerical ability test designed for a wide range of job levels. Assesses ability to work with numerical data in tables and graphs.",
            "job_levels": ["Entry-Level", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 20,
        },
        {
            "name": "Calculation",
            "url": "https://www.shl.com/products/product-catalog/view/calculation/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures ability to perform basic arithmetic calculations including addition, subtraction, multiplication, and division.",
            "job_levels": ["Entry-Level", "Front Line Manager", "Supervisor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 12,
        },
        {
            "name": "Checking",
            "url": "https://www.shl.com/products/product-catalog/view/checking/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to identify errors and discrepancies in lists of numbers and names. Relevant for clerical and data-entry roles.",
            "job_levels": ["Entry-Level", "Front Line Manager", "Supervisor", "General Population"],
            "languages": ["English (USA)", "English International", "French"],
            "duration_minutes": 7,
        },
        {
            "name": "Reading Comprehension - Next Generation",
            "url": "https://www.shl.com/products/product-catalog/view/reading-comprehension-next-generation/",
            "remote_testing": True, "adaptive_irt": True,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Adaptive reading comprehension test suitable for roles requiring written communication skills. Covers main idea, detail recall, inference, and vocabulary.",
            "job_levels": ["Entry-Level", "General Population", "Graduate", "Supervisor", "Front Line Manager"],
            "languages": ["English (USA)", "English International", "Spanish", "French"],
            "duration_minutes": 14,
        },
        {
            "name": "Spatial Reasoning",
            "url": "https://www.shl.com/products/product-catalog/view/spatial-reasoning/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures the ability to visualize and mentally manipulate two- and three-dimensional shapes. Relevant for engineering and technical roles.",
            "job_levels": ["Entry-Level", "Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 15,
        },
        {
            "name": "Mechanical Comprehension",
            "url": "https://www.shl.com/products/product-catalog/view/mechanical-comprehension/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Measures knowledge and understanding of mechanical, physical, and electrical concepts. Ideal for technical, engineering, and manufacturing roles.",
            "job_levels": ["Entry-Level", "Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "Multitasking Ability",
            "url": "https://www.shl.com/products/product-catalog/view/multitasking-ability/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Ability & Aptitude", "test_type_codes": ["A"],
            "description": "Face-valid split-screen simulation measuring the ability to work on multiple tasks simultaneously while maintaining efficiency. Relevant for contact center and operational roles.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "Manager", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)"],
            "duration_minutes": 10,
        },

        # ── PERSONALITY & BEHAVIOR (P) ────────────────────────────────────────
        {
            "name": "OPQ32r (Occupational Personality Questionnaire)",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "The OPQ32r is SHL's flagship personality questionnaire, measuring 32 dimensions of personality relevant to work performance. Used for selection, development, and coaching across all job levels.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Chinese Simplified", "Japanese", "Dutch", "Russian", "Arabic", "Portuguese"],
            "duration_minutes": 25,
        },
        {
            "name": "OPQ32 (Occupational Personality Questionnaire)",
            "url": "https://www.shl.com/products/product-catalog/view/opq32/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Comprehensive occupational personality questionnaire measuring 32 personality scales related to work. Provides detailed profile for selection and development.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Chinese Simplified"],
            "duration_minutes": 40,
        },
        {
            "name": "Motivation Questionnaire (MQ)",
            "url": "https://www.shl.com/products/product-catalog/view/mq/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Measures 18 dimensions of motivation at work, identifying what energizes individuals and what may lead to disengagement. Used for selection, career development, and coaching.",
            "job_levels": ["Director", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish", "Dutch"],
            "duration_minutes": 25,
        },
        {
            "name": "Customer Contact Styles Questionnaire (CCSQ7.2)",
            "url": "https://www.shl.com/products/product-catalog/view/customer-contact-styles-questionnaire/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Measures personality characteristics relevant to roles involving regular contact with customers, including service orientation, empathy, and communication style.",
            "job_levels": ["Entry-Level", "Front Line Manager", "General Population", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 20,
        },
        {
            "name": "Sales Solution Questionnaire (SSQ8)",
            "url": "https://www.shl.com/products/product-catalog/view/sales-solution-questionnaire/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Personality questionnaire designed for sales roles. Measures traits linked to effective prospecting, relationship building, closing, and account management.",
            "job_levels": ["Entry-Level", "Front Line Manager", "General Population", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 25,
        },
        {
            "name": "General Safety Attitude Questionnaire (GSA7)",
            "url": "https://www.shl.com/products/product-catalog/view/general-safety-attitude-questionnaire/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Measures safety-relevant attitudes and behavioral tendencies. Used in manufacturing, industrial, and safety-critical environments.",
            "job_levels": ["Entry-Level", "Front Line Manager", "General Population", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "Spanish"],
            "duration_minutes": 15,
        },
        {
            "name": "Hogan Personality Inventory (HPI)",
            "url": "https://www.shl.com/products/product-catalog/view/hogan-personality-inventory/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Personality & Behavior", "test_type_codes": ["P"],
            "description": "Measures normal personality characteristics linked to career success and organizational fit. Seven primary scales including Adjustment, Ambition, Sociability.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Director", "Executive"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 15,
        },

        # ── BIODATA & SITUATIONAL JUDGEMENT (B) ──────────────────────────────
        {
            "name": "Workplace English Language Test (WELT)",
            "url": "https://www.shl.com/products/product-catalog/view/workplace-english-language-test/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Biodata & Situational Judgement", "test_type_codes": ["B"],
            "description": "Assesses English language proficiency in workplace contexts. Covers listening, reading, and writing components relevant to professional settings.",
            "job_levels": ["Entry-Level", "Graduate", "General Population", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 35,
        },
        {
            "name": "Graduate 8.0 (Situational Judgement)",
            "url": "https://www.shl.com/products/product-catalog/view/graduate-8-situational-judgement/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Biodata & Situational Judgement", "test_type_codes": ["B"],
            "description": "Situational judgement test for graduate candidates. Presents realistic work scenarios and asks candidates to select the most effective response.",
            "job_levels": ["Graduate", "Entry-Level"],
            "languages": ["English International", "English (USA)", "French", "German"],
            "duration_minutes": 30,
        },
        {
            "name": "Manager SJT (Situational Judgement Test)",
            "url": "https://www.shl.com/products/product-catalog/view/manager-sjt/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Biodata & Situational Judgement", "test_type_codes": ["B"],
            "description": "Situational judgement test measuring judgment and decision-making relevant to managerial roles.",
            "job_levels": ["Front Line Manager", "Manager", "Supervisor"],
            "languages": ["English International", "English (USA)"],
            "duration_minutes": 25,
        },

        # ── KNOWLEDGE & SKILLS (K) ────────────────────────────────────────────
        {
            "name": "Java (New)",
            "url": "https://www.shl.com/products/product-catalog/view/java-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses knowledge and skills in Java programming including OOP, data structures, exception handling, and core Java APIs. For software developer roles.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "Java 8 (New)",
            "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses Java 8 skills including lambda expressions, streams, functional interfaces, and new Date/Time API. For software developer roles.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/products/product-catalog/view/python-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses Python programming knowledge covering data structures, functions, OOP, file handling, and standard library usage.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "SQL (New)",
            "url": "https://www.shl.com/products/product-catalog/view/sql-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests knowledge of SQL including SELECT queries, JOINs, GROUP BY, subqueries, and database manipulation. For data analyst and developer roles.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate", "Entry-Level"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "C++ (New)",
            "url": "https://www.shl.com/products/product-catalog/view/c-plus-plus-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests C++ programming knowledge including classes, templates, STL, memory management, and modern C++ features.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "C# (New)",
            "url": "https://www.shl.com/products/product-catalog/view/c-sharp-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses C# programming knowledge including .NET framework, LINQ, async/await, and object-oriented principles.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "JavaScript (New)",
            "url": "https://www.shl.com/products/product-catalog/view/javascript-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests JavaScript knowledge including ES6+, closures, async programming, DOM manipulation, and common patterns.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate", "Entry-Level"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "HTML/CSS (New)",
            "url": "https://www.shl.com/products/product-catalog/view/html-css-new/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses HTML5 and CSS3 knowledge including semantic markup, responsive design, flexbox/grid, and accessibility.",
            "job_levels": ["Mid-Professional", "Professional Individual Contributor", "Graduate", "Entry-Level"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 30,
        },
        {
            "name": "Automata Pro (Coding Simulation)",
            "url": "https://www.shl.com/products/product-catalog/view/automata-pro/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K", "S"],
            "description": "Coding simulation requiring candidates to write actual working code in a realistic IDE environment. Supports multiple languages. Measures practical coding ability beyond syntax knowledge.",
            "job_levels": ["Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 60,
        },
        {
            "name": "Automata (Coding Simulation)",
            "url": "https://www.shl.com/products/product-catalog/view/automata/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K", "S"],
            "description": "Foundational coding simulation assessing ability to write code to solve real problems. Evaluates problem decomposition, code quality, and functional correctness.",
            "job_levels": ["Entry-Level", "Graduate", "Mid-Professional"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 45,
        },
        {
            "name": "Entry Level Sales 7.1",
            "url": "https://www.shl.com/products/product-catalog/view/entry-level-sales-7-1/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses knowledge and aptitude for entry-level sales positions. Covers prospecting, customer interaction, objection handling, and closing.",
            "job_levels": ["Entry-Level", "General Population"],
            "languages": ["English (USA)", "English International", "Spanish"],
            "duration_minutes": 30,
        },
        {
            "name": "Basic Computer Literacy Skills (BCLS7.1)",
            "url": "https://www.shl.com/products/product-catalog/view/basic-computer-literacy-skills/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Assesses fundamental computer skills including file management, word processing, spreadsheets, internet use, and email.",
            "job_levels": ["Entry-Level", "Front Line Manager", "General Population", "Supervisor"],
            "languages": ["English (USA)", "English International", "French", "Spanish"],
            "duration_minutes": 20,
        },
        {
            "name": "Microsoft Word",
            "url": "https://www.shl.com/products/product-catalog/view/microsoft-word/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests practical Microsoft Word skills including document formatting, styles, mail merge, and collaboration features.",
            "job_levels": ["Entry-Level", "General Population", "Supervisor"],
            "languages": ["English (USA)"],
            "duration_minutes": 30,
        },
        {
            "name": "Microsoft Excel",
            "url": "https://www.shl.com/products/product-catalog/view/microsoft-excel/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests practical Microsoft Excel skills including formulas, pivot tables, data analysis, and chart creation.",
            "job_levels": ["Entry-Level", "General Population", "Supervisor", "Mid-Professional"],
            "languages": ["English (USA)"],
            "duration_minutes": 30,
        },
        {
            "name": "Data Entry Speed & Accuracy",
            "url": "https://www.shl.com/products/product-catalog/view/data-entry-speed-accuracy/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Measures data entry speed (keystrokes per hour) and accuracy. Relevant for administrative, data processing, and back-office roles.",
            "job_levels": ["Entry-Level", "General Population", "Supervisor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 7,
        },
        {
            "name": "Accounting & Finance knowledge test",
            "url": "https://www.shl.com/products/product-catalog/view/accounting-finance/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Knowledge & Skills", "test_type_codes": ["K"],
            "description": "Tests knowledge of accounting principles, financial statements, and financial analysis. For finance and accounting roles.",
            "job_levels": ["Entry-Level", "Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English (USA)", "English International"],
            "duration_minutes": 25,
        },

        # ── COMPETENCIES (C) ──────────────────────────────────────────────────
        {
            "name": "Universal Competency Framework (UCF)",
            "url": "https://www.shl.com/products/assessments/behavioral-assessments/universal-competency-framework/",
            "remote_testing": False, "adaptive_irt": False,
            "test_type": "Competencies", "test_type_codes": ["C"],
            "description": "SHL's science-based framework covering 8 Great Competencies and 20 behavioral dimensions. Used to structure job analysis, assessment, and development conversations.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": None,
        },

        # ── DEVELOPMENT & 360 (D) ──────────────────────────────────────────────
        {
            "name": "360-Degree Feedback",
            "url": "https://www.shl.com/products/360/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Development & 360", "test_type_codes": ["D"],
            "description": "Multi-rater 360-degree feedback tool providing data visualizations that clarify performance and understand potential. Covers competencies aligned to UCF.",
            "job_levels": ["Director", "Executive", "Front Line Manager", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English International", "English (USA)", "French", "German", "Spanish"],
            "duration_minutes": 20,
        },

        # ── SIMULATIONS / EXERCISES (S, E) ────────────────────────────────────
        {
            "name": "Virtual Assessment and Development Centers",
            "url": "https://www.shl.com/products/product-catalog/view/virtual-assessment-and-development-centers/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Simulations", "test_type_codes": ["S", "E"],
            "description": "Comprehensive virtual assessment center platform combining exercises, simulations, and interviews to evaluate candidates at manager and leadership levels.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English (USA)", "English International", "Italian", "German", "Portuguese", "Spanish", "French", "Dutch", "Japanese", "Chinese Simplified", "Arabic"],
            "duration_minutes": None,
        },
        {
            "name": "Contact Center – SVAR Simulation",
            "url": "https://www.shl.com/products/product-catalog/view/contact-center-svar/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Simulations", "test_type_codes": ["S"],
            "description": "Realistic contact center simulation assessing how candidates handle customer calls, multitask, and apply company procedures in a simulated call environment.",
            "job_levels": ["Entry-Level", "Front Line Manager", "General Population", "Supervisor"],
            "languages": ["English (USA)", "English International", "Spanish"],
            "duration_minutes": 15,
        },
        {
            "name": "Smart Interview (AI Video Interview)",
            "url": "https://www.shl.com/products/video-interviews/",
            "remote_testing": True, "adaptive_irt": False,
            "test_type": "Simulations", "test_type_codes": ["S"],
            "description": "AI-powered video interview platform supporting on-demand and live structured interviews. Provides scoring and candidate ranking.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"],
            "languages": ["English (USA)", "English International", "French", "German", "Spanish", "Chinese Simplified", "Japanese"],
            "duration_minutes": 20,
        },
    ]
