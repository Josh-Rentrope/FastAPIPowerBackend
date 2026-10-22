"""Choose the next screening question by information gain, not by the model.

## The problem

A person may match thousands of studies. Asking every criterion of every protocol
is untenable, and asking a fixed questionnaire wastes the user's attention on
questions that eliminate nothing. What is wanted is twenty questions: each one
chosen because it splits the remaining candidate set as evenly as possible.

That is a decision-tree split, and it is arithmetic over the candidate pool. No
model is involved, so it costs nothing, cannot hallucinate, and gives the same
answer twice.

## What is structured and what is not

Age, sex, and healthy-volunteer status come from the API as discrete values, so
they are **form fields** — collected once, applied as filters, never asked
conversationally. That is deliberate: they are also the questions every
demographic form already asks, so asking them in chat would be redundant.

The conversational questions come from `trial_eligibility_facts`: prior therapy,
biomarkers, pregnancy, infection status. Each fact carries a polarity — a trial
either *requires* it or *excludes* it — which is what makes it a split.

## Scoring

For a candidate question, trials partition into three groups: those that require
the attribute, those that exclude it, and those that are silent. The score is the
entropy of that partition: a question that puts 90% of trials in one bucket
teaches almost nothing, one that splits them evenly teaches a lot.

A question with no discrimination (one group holds everything, or answering
would empty the pool) is skipped outright rather than asked and wasted.
"""

from __future__ import annotations

import math
from typing import Iterable

# How many candidates to examine when scoring. The split statistics stabilise long
# before this, and bounding it keeps the endpoint interactive on a cold pool.
SAMPLE_LIMIT = 20000

# Questions that should be a form, never a chat turn.
FORM_FIELDS = ("age", "sex", "healthy_volunteer")

# A fact must appear on at least this share of candidates to be worth asking.
# Below it, the question cannot move the needle.
MIN_SUPPORT = 0.05

# A question must actually teach something. Entropy is bounded by log2(3) ~= 1.58
# for a three-way split; 0.5 bits is roughly "no bucket above 80%".
MIN_GAIN = 0.5

# Once the pool is this small, stop. At this size the shortlist is readable, and
# another question costs the reader more attention than it saves.
STOP_AT = 40

# A hard ceiling on fact questions. Entropy will happily keep splitting a large
# pool, but attention is the scarce resource, not the search space.
MAX_QUESTIONS = 8

# Facts phrased in a way a person can actually answer. Anything not listed is
# still scored, but questions are generated from this mapping so the assistant
# never asks something incomprehensible.
QUESTION_TEXT = {
    ("prior_therapy", "chemotherapy"): "Have you had chemotherapy before?",
    ("prior_therapy", "immunotherapy"): "Have you had immunotherapy before?",
    ("prior_therapy", "checkpoint_inhibitor"):
        "Have you had an immune checkpoint inhibitor, such as pembrolizumab or nivolumab?",
    ("prior_therapy", "anti_egfr"): "Have you had an anti-EGFR antibody, such as cetuximab?",
    ("prior_therapy", "anti_tgfb"): "Have you had a treatment targeting TGF-beta?",
    ("prior_therapy", "radiotherapy"): "Have you had radiation therapy?",
    ("biomarker", "pd_l1"): "Do you know your PD-L1 status?",
    ("biomarker", "her2"): "Do you know your HER2 status?",
    ("biomarker", "msi"): "Do you know whether your tumour is MSI-high?",
    ("biomarker", "hpv"): "Do you know your HPV status?",
    ("biomarker", "brca"): "Do you know whether you carry a BRCA mutation?",
    ("infection", "hiv"): "Do you have HIV?",
    ("infection", "hepatitis_b"): "Do you have hepatitis B?",
    ("infection", "hepatitis_c"): "Do you have hepatitis C?",
    ("pregnancy", "pregnancy"): "Is there any chance you are pregnant or breastfeeding?",
    ("autoimmune", "active"): "Do you have an autoimmune condition needing treatment?",
    ("transplant", "transplant"): "Have you had an organ or stem-cell transplant?",
    ("cns", "metastases"): "Do you have metastases in the brain or spine?",
    ("other_malignancy", "recent"): "Have you had another cancer in the last few years?",
    # Condition-specific facts that were in the data but unphrased, so the
    # segmenter could score them and never ask them. These are what make the
    # questions feel like they are about this reader's disease.
    ("performance_status", "ecog"):
        "Do you know your ECOG performance status? It runs from 0 (fully active) "
        "to 4 (fully disabled).",
    ("prognosis", "life_expectancy"):
        "Has your care team estimated how long your disease might stay stable?",
    ("organ_function", "required"):
        "Do you know whether your liver and kidney function meet standard trial "
        "requirements?",
    ("prior_therapy", "surgery"): "Have you had surgery for this condition?",
    ("vaccine", "live"): "Have you had a live vaccine in the last few weeks?",
    ("biomarker", "egfr_mutation"):
        "Do you know whether your tumour has an EGFR mutation?",
    ("biomarker", "alk"):
        "Do you know whether your tumour has an ALK rearrangement?",
    ("biomarker", "braf"):
        "Do you know whether your tumour has a BRAF mutation?",
    ("biomarker", "kras"):
        "Do you know whether your tumour has a KRAS mutation?",
    ("biomarker", "ros1"):
        "Do you know whether your tumour has a ROS1 fusion?",
    ("biomarker", "tmb"):
        "Do you know your tumour mutational burden (TMB)?",
}


# Not every question can be asked of everyone, and entropy alone does not know
# that. Pregnancy is the clearest case: on a mixed pool it splits beautifully --
# hundreds of trials exclude it, hundreds require a negative test -- so it wins on
# information and would be the first thing anyone is asked, including men. That is
# both wrong and strange. Applicability is checked before a question is a
# candidate at all.
#
# Cost is separate and covers questions that are applicable but unpleasant or
# hard to answer: sensitive topics, and biomarkers most people have never been
# told. Score is gain/cost, so a cheap question wins a close race.

COST = {
    "pregnancy": 2.0,
    "infection": 2.0,      # HIV/hepatitis: sensitive, and often unknown to the patient
    "biomarker": 1.8,      # most patients do not know their marker status
    "other_malignancy": 1.4,
    "transplant": 1.3,
    "performance_status": 1.6,
    "prognosis": 1.8,
    "organ_function": 2.4,  # a blood test, not something a patient is usually told
    "vaccine": 1.3,
}


def applicable(fact_key: str, fact_value: str, answers: dict) -> bool:
    """Whether this question can meaningfully be asked given the answers so far."""
    # A healthy volunteer has no tumour to characterise. Asking about BRAF or
    # ECOG would be nonsense, however well it splits the pool.
    if (answers.get("healthy_volunteer") is True
            and fact_key in ("biomarker", "performance_status", "prognosis",
                             "cns", "other_malignancy", "transplant")):
        return False

    if fact_key == "pregnancy":
        sex = (answers.get("sex") or "").lower()
        if sex != "female":
            # Without a known sex we cannot justify asking, and with "male" it is
            # simply not applicable.
            return False
        age = answers.get("age")
        if isinstance(age, int):
            if age >= 55:
                return False
        else:
            # With a band, only skip when the whole band is past the cutoff.
            age_min = answers.get("age_min")
            if isinstance(age_min, int) and age_min >= 55:
                return False
    if fact_key == "prior_therapy" and fact_value == "checkpoint_inhibitor":
        # Only worth asking someone who has had some systemic therapy.
        facts = answers.get("facts") or {}
        if isinstance(facts, dict) and facts.get("prior_therapy:immunotherapy") is False:
            return False
    return True


def entropy(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _where(condition: str | None, status: str | None, answers: dict,
           phase: str | None = None) -> tuple[str, dict]:
    """Candidate filter: filters first, structured fields, then answered facts."""
    # No eligibility gate here. The pool is LEFT-joined, so a study the registry
    # has no eligibility rows for still counts -- exactly as it does in /studies.
    # Gating it out made the shortlist size disagree with the result count shown
    # after applying the answers, which reads as a bug because it is one.
    clauses: list[str] = []
    params: dict = {}

    if condition:
        # Same predicate as /studies' `q`, deliberately. The shortlist count is
        # shown to the reader and then has to match the result count after they
        # apply it; searching only `conditions` here made those two numbers
        # disagree by a few hundred, which reads as a bug because it is one.
        clauses.append("(pd.study_title ILIKE %(cond)s OR pd.conditions ILIKE %(cond)s)")
        params["cond"] = f"%{condition}%"
    if status:
        wanted = [x.strip().upper().replace(" ", "_")
                  for x in status.split(",") if x.strip()]
        if wanted:
            clauses.append("pd.status = ANY(%(status)s)")
            params["status"] = wanted
    if phase:
        # Same token-membership rule as /studies: EARLY_PHASE1 is not PHASE1.
        wanted = [x.strip().upper().replace(" ", "").replace("_", "")
                  for x in phase.split(",") if x.strip()]
        if wanted:
            clauses.append("""
                EXISTS (
                    SELECT 1 FROM unnest(string_to_array(pd.phase, '|')) AS ph
                    WHERE upper(replace(replace(ph, ' ', ''), '_', '')) = ANY(%(phases)s)
                )
            """)
            params["phases"] = wanted

    age = answers.get("age")
    if isinstance(age, int):
        clauses.append("(te.min_age_years IS NULL OR te.min_age_years <= %(age)s)")
        clauses.append("(te.max_age_years IS NULL OR te.max_age_years >= %(age)s)")
        params["age"] = age
    else:
        # Same overlap rule as /studies, so the count shown while asking is the
        # count the list will show after applying.
        age_min = answers.get("age_min")
        age_max = answers.get("age_max")
        if isinstance(age_min, int):
            clauses.append("(te.max_age_years IS NULL OR te.max_age_years >= %(age_min)s)")
            params["age_min"] = age_min
        if isinstance(age_max, int):
            clauses.append("(te.min_age_years IS NULL OR te.min_age_years <= %(age_max)s)")
            params["age_max"] = age_max

    sex = (answers.get("sex") or "").upper()
    if sex in ("FEMALE", "MALE"):
        clauses.append("(te.sex IS NULL OR te.sex = 'ALL' OR te.sex = %(sex)s)")
        params["sex"] = sex

    healthy = answers.get("healthy_volunteer")
    if healthy is True:
        clauses.append("te.healthy_volunteers = true")
    elif healthy is False:
        clauses.append("coalesce(te.healthy_volunteers, false) = false")

    return " AND ".join(clauses), params


def _answered_fact_filter(answers: dict) -> list[tuple[str, str, bool]]:
    """Facts the user has answered, as (key, value, has_it).

    Answers arrive keyed `"prior_therapy:chemotherapy": true`, so one dict carries
    both the dimension and the value.
    """
    out: list[tuple[str, str, bool]] = []
    facts = answers.get("facts")
    if isinstance(facts, dict):
        for k, v in facts.items():
            if not isinstance(v, bool):
                continue
            if ":" in k:
                fk, fv = k.split(":", 1)
            else:
                fk, fv = k, "any"
            out.append((fk, fv, v))
    return out


def next_question(cur, condition: str | None, status: str | None,
                  answers: dict, phase: str | None = None) -> dict:
    """Return the highest-gain question, with the counts that justify it."""
    where, params = _where(condition, status, answers, phase)

    # Apply facts already answered: a trial that requires a fact the user lacks,
    # or excludes a fact the user has, drops out.
    #
    # Each fact needs its OWN parameter names. Reusing `fk`/`fv` meant the second
    # answer overwrote the first, so the pool grew after an answer and the same
    # question came back -- a loop the UI could only hide.
    answered = _answered_fact_filter(answers)
    for i, (fk, fv, has_it) in enumerate(answered):
        polarity = "exclude" if has_it else "require"
        where += f"""
            AND NOT EXISTS (
                SELECT 1 FROM trial_eligibility_facts f
                WHERE f.nct_number = te.nct_number
                  AND f.fact_key = %(fk{i})s AND f.fact_value = %(fv{i})s
                  AND f.polarity = '{polarity}')
        """
        params[f"fk{i}"] = fk
        params[f"fv{i}"] = fv

    cur.execute(f"""
        SELECT count(*) FROM (
            SELECT pd.nct_number
            FROM processed_data pd
            LEFT JOIN trial_eligibility te ON te.nct_number = pd.nct_number
            WHERE {where or "TRUE"}
            LIMIT {SAMPLE_LIMIT}
        ) x
    """, params)
    total = cur.fetchone()[0]
    if total == 0:
        return {"total": 0, "next_question": None,
                "message": "No studies match those filters yet."}
    if total <= STOP_AT:
        return {"total": total, "next_question": None,
                "message": (f"Only {total:,} studies remain. That is already a "
                            "readable shortlist, and another question would cost "
                            "more attention than it saves.")}
    if len(answered) >= MAX_QUESTIONS:
        return {"total": total, "next_question": None,
                "message": (f"You have answered {len(answered)} questions and "
                            f"{total:,} studies remain. Here is the shortlist; "
                            "answering more would give diminishing returns.")}

    # Count, within the surviving pool, how each candidate fact partitions it.
    cur.execute(f"""
        WITH pool AS (
            SELECT pd.nct_number
            FROM processed_data pd
            LEFT JOIN trial_eligibility te ON te.nct_number = pd.nct_number
            WHERE {where or "TRUE"}
            LIMIT {SAMPLE_LIMIT}
        ),
        dist AS (
            SELECT f.fact_key, f.fact_value, f.polarity, count(DISTINCT f.nct_number) AS n
            FROM trial_eligibility_facts f
            JOIN pool ON pool.nct_number = f.nct_number
            GROUP BY 1, 2, 3
        )
        SELECT fact_key, fact_value,
               sum(n) FILTER (WHERE polarity = 'require') AS requires,
               sum(n) FILTER (WHERE polarity = 'exclude') AS excludes
        FROM dist
        GROUP BY 1, 2
        HAVING sum(n) >= %(min_support_rows)s
        ORDER BY 1, 2
    """, {**params, "min_support_rows": max(5, int(total * MIN_SUPPORT))})
    rows = cur.fetchall()

    # Never ask something already answered, whatever the counts say. The SQL
    # filter above should have removed it; this is the guard that keeps a future
    # change to that filter from turning the conversation into a loop.
    answered_keys = {f"{fk}:{fv}" for fk, fv, _ in answered}

    best = None
    for fact_key, fact_value, requires, excludes in rows:
        if f"{fact_key}:{fact_value}" in answered_keys:
            continue
        requires = int(requires or 0)
        excludes = int(excludes or 0)
        silent = max(0, total - requires - excludes)
        parts = [requires, excludes, silent]

        # A question whose answers all land in one bucket teaches nothing.
        non_empty = sum(1 for x in parts if x > 0)
        if non_empty < 2:
            continue
        # Answering must leave something on both sides of the interesting split.
        if requires == 0 and excludes == 0:
            continue

        # A question that cannot be asked of this person is not a candidate,
        # however well it would split the pool.
        if not applicable(fact_key, fact_value, answers):
            continue

        gain = entropy(parts)
        if gain < MIN_GAIN:
            continue
        question = QUESTION_TEXT.get((fact_key, fact_value))
        # Unmapped facts are still scored (they show how much is learnable) but
        # are never surfaced as a question, because we cannot phrase them.
        if not question:
            continue

        score = gain / COST.get(fact_key, 1.0)

        if best is None or score > best["_score"]:
            best = {
                "_score": score,
                "gain": round(gain, 3),
                "cost": COST.get(fact_key, 1.0),
                "question": question,
                "fact_key": fact_key,
                "fact_value": fact_value,
                "key": f"{fact_key}:{fact_value}",
                "split": {"requires": requires, "excludes": excludes, "silent": silent},
                "eliminates_if_yes": excludes,
                "eliminates_if_no": requires,
            }

    if best is None:
        return {"total": total, "next_question": None,
                "message": ("No further question meaningfully narrows these "
                            f"{total:,} studies. Show the shortlist.")}

    best.pop("_score", None)
    return {
        "total": total,
        "next_question": best,
        "note": ("Sorted by how evenly the question splits the remaining "
                 "candidates. Answering it is expected to remove "
                 f"{best['eliminates_if_yes']:,} studies if yes and "
                 f"{best['eliminates_if_no']:,} if no."),
    }
