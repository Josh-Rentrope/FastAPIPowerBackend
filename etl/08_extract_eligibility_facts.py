#!/usr/bin/env python3
"""Derive screening facts from the free-text criteria, deterministically.

## Where this sits

`trial_eligibility` holds what the API gave us: sex, age bounds, healthy
volunteers, and the raw criteria text. Age and sex are already discrete, so most
screening works without any of this.

This stage covers the residue — the parts of a protocol that are only expressed
in prose: prior therapy, biomarkers, pregnancy, organ function, and so on. It
emits rows into `trial_eligibility_facts`, each with the sentence it came from
and how confident the rule is.

## Why rules first

The instruction behind this design was to use a model to *build a parser*, not to
run one per query. These regex rules are that parser's first, transparent tier:
they are free, instantaneous, reproducible, and auditable — you can point at a
fact and see the exact clause it came from. A classifier that generalises better
plugs in at the same seam (`method` column) without changing any consumer.

Facts are deliberately conservative: a rule that is not confident stays silent
rather than guessing. A missing fact means "ask a human or run the classifier";
a wrong fact means bad screening.

Usage:
    uv run python etl/08_extract_eligibility_facts.py
    uv run python etl/08_extract_eligibility_facts.py --limit 5000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import psycopg2
from psycopg2.extras import execute_values

METHOD = "rule/1"

# ── section detection ───────────────────────────────────────────────────────
#
# Criteria text is conventionally "Inclusion Criteria: ... Exclusion Criteria:
# ...". Polarity matters: "prior chemotherapy" in an inclusion list is a
# requirement, in an exclusion list it is a disqualifier. Getting this backwards
# inverts the screening, so the heading is tracked explicitly.

_INC = re.compile(r"^\s*(inclusion(\s+criteria)?|key inclusion)", re.I)
_EXC = re.compile(r"^\s*(exclusion(\s+criteria)?|key exclusion)", re.I)

_SENT_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")


def sentences_with_polarity(text: str) -> list[tuple[str, str]]:
    """Yield (sentence, 'require'|'exclude') pairs."""
    out: list[tuple[str, str]] = []
    polarity = "require"
    for line in (text or "").splitlines() or []:
        if _INC.search(line):
            polarity = "require"
            continue
        if _EXC.search(line):
            polarity = "exclude"
            continue
        for s in _SENT_SPLIT.split(line):
            s = s.strip(" \t*-•·")
            if len(s) >= 12:
                out.append((s, polarity))
    return out


# ── rules ───────────────────────────────────────────────────────────────────
#
# (fact_key, fact_value, pattern, confidence). The first matching pattern per
# key/value pair wins; a sentence may produce several facts.

RULES: list[tuple[str, str, re.Pattern, float]] = [
    ("pregnancy", "pregnancy", re.compile(r"\b(pregnan\w*|lactat\w*|breast-?feed\w*|nursing)\b", re.I), 0.9),

    ("prior_therapy", "chemotherapy", re.compile(r"\bchemo(therapy|therapeutic)\b", re.I), 0.8),
    ("prior_therapy", "immunotherapy", re.compile(r"\bimmunotherapy\b", re.I), 0.8),
    ("prior_therapy", "checkpoint_inhibitor",
     re.compile(r"\b(checkpoint inhibitor|anti[- ]pd[- ]?1|anti[- ]pd[- ]?l1|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab)\b", re.I), 0.85),
    ("prior_therapy", "anti_egfr", re.compile(r"\b(cetuximab|panitumumab|anti[- ]egfr)\b", re.I), 0.85),
    ("prior_therapy", "anti_tgfb", re.compile(r"\b(anti[- ]tgf|tgf-?[βb]eta?)\b", re.I), 0.85),
    ("prior_therapy", "radiotherapy", re.compile(r"\b(radiotherapy|radiation therapy)\b", re.I), 0.8),
    ("prior_therapy", "surgery", re.compile(r"\b(surgical resection|prior surgery)\b", re.I), 0.7),

    ("biomarker", "pd_l1", re.compile(r"\bpd[- ]?l1\b", re.I), 0.9),
    ("biomarker", "her2", re.compile(r"\bher2\b", re.I), 0.9),
    ("biomarker", "egfr_mutation", re.compile(r"\begfr\b", re.I), 0.7),
    ("biomarker", "brca", re.compile(r"\bbrca\s?[12]?\b", re.I), 0.9),
    ("biomarker", "msi", re.compile(r"\b(msi[- ]?(high|h)|microsatellite instability)\b", re.I), 0.9),
    ("biomarker", "hpv", re.compile(r"\bhpv\b", re.I), 0.85),
    ("biomarker", "kras", re.compile(r"\bkras\b", re.I), 0.9),
    ("biomarker", "alk", re.compile(r"\balk\b", re.I), 0.7),
    ("biomarker", "ros1", re.compile(r"\bros1\b", re.I), 0.9),
    ("biomarker", "braf", re.compile(r"\bbraf\b", re.I), 0.9),
    ("biomarker", "tmb", re.compile(r"\b(tmb|tumou?r mutational burden)\b", re.I), 0.85),

    ("performance_status", "ecog", re.compile(r"\becog\b", re.I), 0.8),
    ("prognosis", "life_expectancy", re.compile(r"\blife expectancy\b", re.I), 0.85),
    ("organ_function", "required", re.compile(r"\b(adequate (organ|hepatic|renal|bone marrow)|organ function)\b", re.I), 0.75),
    ("transplant", "transplant", re.compile(r"\b(organ transplant|stem cell transplant|allogeneic)\b", re.I), 0.8),
    ("infection", "hiv", re.compile(r"\bhiv\b", re.I), 0.85),
    ("infection", "hepatitis_b", re.compile(r"\bhepatitis b\b", re.I), 0.85),
    ("infection", "hepatitis_c", re.compile(r"\bhepatitis c\b", re.I), 0.85),
    ("cns", "metastases", re.compile(r"\b(brain metast|leptomeningeal|carcinomatous meningitis|cord compression)\b", re.I), 0.9),
    ("other_malignancy", "recent", re.compile(r"\b(other malignancy|second malignan|prior malignan)\b", re.I), 0.8),
    ("autoimmune", "active", re.compile(r"\bautoimmune\b", re.I), 0.75),
    ("vaccine", "live", re.compile(r"\blive (attenuated )?vaccine\b", re.I), 0.85),
]


def extract(text: str) -> list[tuple[str, str, str, float, str]]:
    """Return (fact_key, fact_value, polarity, confidence, evidence)."""
    found: dict[tuple[str, str], tuple[str, float, str]] = {}
    for sentence, polarity in sentences_with_polarity(text):
        for key, value, rx, conf in RULES:
            if (key, value) in found:
                continue
            m = rx.search(sentence)
            if m:
                found[(key, value)] = (polarity, conf, sentence[:400])
    return [(k, v, p, c, ev) for (k, v), (p, c, ev) in found.items()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--limit", type=int, help="only process N studies (development)")
    p.add_argument("--batch", type=int, default=5000)
    args = p.parse_args()

    conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    conn.autocommit = False
    cur = conn.cursor()

    # Only rows whose facts are missing or from an older extractor. Re-running
    # after a rule change re-derives everything without a refetch.
    sql = """
        SELECT te.nct_number, te.criteria_text
        FROM trial_eligibility te
        WHERE te.criteria_text IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM trial_eligibility_facts f
              WHERE f.nct_number = te.nct_number AND f.method = %s)
        ORDER BY te.nct_number
    """
    params: list = [METHOD]
    if args.limit:
        sql += " LIMIT %s"
        params.append(args.limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    print(f"studies to process: {len(rows):,}")

    t0 = time.time()
    buf = []
    total_facts = 0
    processed = 0
    for nct, text in rows:
        for key, value, polarity, conf, evidence in extract(text):
            buf.append((nct, key, value, polarity, conf, METHOD, evidence))
        processed += 1
        if len(buf) >= args.batch:
            execute_values(cur, """
                INSERT INTO trial_eligibility_facts
                    (nct_number, fact_key, fact_value, polarity, confidence, method, evidence)
                VALUES %s
                ON CONFLICT (nct_number, fact_key, fact_value) DO UPDATE SET
                    polarity = EXCLUDED.polarity,
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method,
                    evidence = EXCLUDED.evidence
            """, buf, page_size=1000)
            conn.commit()
            total_facts += len(buf)
            buf.clear()
            if processed % 50000 == 0:
                print(f"  {processed:,} studies, {total_facts:,} facts, "
                      f"{time.time()-t0:.0f}s")
    if buf:
        execute_values(cur, """
            INSERT INTO trial_eligibility_facts
                (nct_number, fact_key, fact_value, polarity, confidence, method, evidence)
            VALUES %s
            ON CONFLICT (nct_number, fact_key, fact_value) DO UPDATE SET
                polarity = EXCLUDED.polarity,
                confidence = EXCLUDED.confidence,
                method = EXCLUDED.method,
                evidence = EXCLUDED.evidence
        """, buf, page_size=1000)
        conn.commit()
        total_facts += len(buf)

    print(f"\nprocessed {processed:,} studies -> {total_facts:,} facts in {time.time()-t0:.0f}s")

    print("\n-- most common facts --")
    cur.execute("""
        SELECT fact_key, fact_value, polarity, count(*)
        FROM trial_eligibility_facts
        GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 20
    """)
    for k, v, pol, n in cur.fetchall():
        print(f"   {k:<20} {v:<20} {pol:<8} {n:>8,}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
