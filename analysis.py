import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path("outputs/gatemem_final")

DOMAINS = [
    "education",
    "household",
    "medical",
    "office"
]

SETTINGS = [
    "private",
    "ungoverned",
    "governed"
]

# Actions that mean the system attempted to answer
ANSWER_ACTIONS = {
    "answer",
    "answer_redacted"
}

# Actions representing conservative refusal / lack of usable memory
REFUSAL_ACTIONS = {
    "refuse",
    "no_memory"
}


# ============================================================
# JSONL helpers
# ============================================================

def load_jsonl(path: Path):
    """
    Read a JSONL file and return a list of dictionaries.
    """
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Failed to parse "
                    f"{path} line {line_number}: {e}"
                )

    return rows


# ============================================================
# Load predictions
# ============================================================

def load_predictions(path: Path):
    """
    Extract the fields needed from predictions.jsonl.
    """

    raw_rows = load_jsonl(path)

    records = []

    for row in raw_rows:

        output = row.get("output") or {}

        # output.answer_structured may also contain useful information
        answer_structured = output.get("answer_structured") or {}

        records.append({
            "checkpoint_id": row.get("checkpoint_id"),
            "episode_id": row.get("episode_id"),
            "query_type": row.get("query_type"),
            "attack_type": row.get("attack_type"),

            "expected_action": row.get("expected_action"),

            # Actual system decision
            "pred_action": output.get("action"),

            # Raw answer
            "answer": output.get("answer"),

            # Additional status, useful when inspecting cases
            "answer_status": answer_structured.get("status"),

            "query_text": row.get("query_text"),

            # Memory usage information
            "used_record_ids": output.get("used_record_ids", []),

            # Number of memories actually used
            "n_used_memories": len(
                output.get("used_record_ids", []) or []
            )
        })

    df = pd.DataFrame(records)

    return df


# ============================================================
# Load LLM judge results
# ============================================================

def load_judge_scores(path: Path):
    """
    Extract LLM-judge results from judge_scores.jsonl.
    """

    raw_rows = load_jsonl(path)

    records = []

    for row in raw_rows:

        judge = row.get("judge") or {}

        records.append({
            "checkpoint_id": row.get("checkpoint_id"),

            "judge_query_type": row.get("query_type"),

            "judge_action_ok": judge.get("action_ok"),
            "judge_action_ok_llm": judge.get("action_ok_llm"),

            # Main Utility result used in this analysis
            "utility_ok": judge.get("utility_ok"),

            "privacy_leak": judge.get("privacy_leak"),
            "deletion_leak": judge.get("deletion_leak"),

            "judge_notes": judge.get("notes"),

            "judge_parse_ok": judge.get("parse_ok")
        })

    df = pd.DataFrame(records)

    return df


# ============================================================
# Load one experimental configuration
# ============================================================

def load_run(domain: str, setting: str):
    run_dir = ROOT / f"{domain}_{setting}"

    predictions_path = run_dir / "predictions.jsonl"
    judge_path = run_dir / "judge_scores.jsonl"

    print(f"\n--- Loading {domain}_{setting} ---")
    print("Predictions:", predictions_path)
    print("Judge:", judge_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing file: {predictions_path}")

    if not judge_path.exists():
        raise FileNotFoundError(f"Missing file: {judge_path}")

    predictions = load_predictions(predictions_path)
    judge_scores = load_judge_scores(judge_path)

    # ------------------------------
    # Debug information
    # ------------------------------
    print("Prediction rows:", len(predictions))
    print("Prediction columns:", predictions.columns.tolist())

    print("Judge rows:", len(judge_scores))
    print("Judge columns:", judge_scores.columns.tolist())

    # ------------------------------
    # Explicit validation
    # ------------------------------
    if predictions.empty:
        raise ValueError(
            f"{predictions_path} produced an empty DataFrame."
        )

    if judge_scores.empty:
        raise ValueError(
            f"{judge_path} produced an empty DataFrame."
        )

    if "checkpoint_id" not in predictions.columns:
        raise ValueError(
            f"'checkpoint_id' missing from predictions:\n"
            f"{predictions_path}\n"
            f"Columns: {predictions.columns.tolist()}"
        )

    if "checkpoint_id" not in judge_scores.columns:
        raise ValueError(
            f"'checkpoint_id' missing from judge scores:\n"
            f"{judge_path}\n"
            f"Columns: {judge_scores.columns.tolist()}"
        )

    predictions = predictions.drop_duplicates(
        subset=["checkpoint_id"],
        keep="last"
    )

    judge_scores = judge_scores.drop_duplicates(
        subset=["checkpoint_id"],
        keep="last"
    )

    df = predictions.merge(
        judge_scores,
        on="checkpoint_id",
        how="left",
        validate="one_to_one"
    )

    df["domain"] = domain.capitalize()
    df["setting"] = setting.capitalize()

    print("Merged rows:", len(df))

    return df


# ============================================================
# Classify Utility outcomes
# ============================================================

def classify_utility_outcome(row):
    """
    Categorise the result of an answerable Utility checkpoint.

    Categories:
        correct
        false_refusal
        false_no_memory
        answered_but_wrong
        other_failure
    """

    action = str(row["pred_action"]).lower()
    utility_ok = row["utility_ok"]

    # LLM judge says answer is correct
    if utility_ok is True or utility_ok == 1:
        return "correct"

    # System refused an answerable query
    if action == "refuse":
        return "false_refusal"

    # System claimed insufficient/no usable memory
    if action == "no_memory":
        return "false_no_memory"

    # System tried to answer, but judge marked Utility incorrect
    if action in ANSWER_ACTIONS:
        return "answered_but_wrong"

    return "other_failure"


# ============================================================
# Analyse one run
# ============================================================

def analyse_utility(df):
    """
    Analyse Utility checkpoints for one domain/configuration.
    """

    # Only Utility checkpoints
    utility = df[
        df["query_type"] == "utility"
    ].copy()

    # For Utility evaluation we focus on queries expected to be answerable
    answerable = utility[
        utility["expected_action"].isin(ANSWER_ACTIONS)
    ].copy()

    answerable["outcome"] = answerable.apply(
        classify_utility_outcome,
        axis=1
    )

    total = len(answerable)

    counts = answerable["outcome"].value_counts()

    correct = counts.get("correct", 0)
    false_refusal = counts.get("false_refusal", 0)
    false_no_memory = counts.get("false_no_memory", 0)
    answered_but_wrong = counts.get(
        "answered_but_wrong",
        0
    )
    other_failure = counts.get(
        "other_failure",
        0
    )

    failures = total - correct

    decision_failures = (
        false_refusal
        + false_no_memory
    )

    # % of ALL answerable Utility cases lost through refusal/no-memory
    if total > 0:
        decision_failure_rate_all = (
            decision_failures / total
        )
    else:
        decision_failure_rate_all = None

    # % of Utility failures caused by refusal/no-memory
    if failures > 0:
        decision_failure_share = (
            decision_failures / failures
        )
    else:
        decision_failure_share = None

    summary = {
        "Domain": answerable["domain"].iloc[0]
        if len(answerable) else None,

        "Setting": answerable["setting"].iloc[0]
        if len(answerable) else None,

        "Utility Cases": total,

        "Correct": correct,

        "False Refusal": false_refusal,

        "False No-Memory": false_no_memory,

        "Answered but Wrong": answered_but_wrong,

        "Other Failure": other_failure,

        "Total Utility Failures": failures,

        "Refuse/No-Memory Failures": decision_failures,

        "Refuse/No-Memory Rate (all utility cases)":
            decision_failure_rate_all,

        "Refuse/No-Memory Share of Failures":
            decision_failure_share
    }

    return summary, answerable


# ============================================================
# Load all 12 runs
# ============================================================

all_runs = {}

summary_rows = []

detailed_utility_rows = []


for domain in DOMAINS:

    for setting in SETTINGS:

        print(
            f"Loading {domain}_{setting} ..."
        )

        df = load_run(
            domain,
            setting
        )

        all_runs[
            (domain, setting)
        ] = df

        summary, utility_details = analyse_utility(df)

        summary_rows.append(summary)

        detailed_utility_rows.append(
            utility_details
        )


# ============================================================
# Create overall Utility error summary
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

# Pretty percentage columns
summary_df[
    "Refuse/No-Memory Rate (all utility cases) %"
] = (
    summary_df[
        "Refuse/No-Memory Rate (all utility cases)"
    ] * 100
).round(2)

summary_df[
    "Refuse/No-Memory Share of Failures %"
] = (
    summary_df[
        "Refuse/No-Memory Share of Failures"
    ] * 100
).round(2)


print("\n")
print("=" * 90)
print("UTILITY ERROR ANALYSIS")
print("=" * 90)

print(
    summary_df[
        [
            "Domain",
            "Setting",
            "Utility Cases",
            "Correct",
            "False Refusal",
            "False No-Memory",
            "Answered but Wrong",
            "Total Utility Failures",
            "Refuse/No-Memory Share of Failures %"
        ]
    ].to_string(index=False)
)


# ============================================================
# Paired comparison:
# Ungoverned correct but Governed refuse/no_memory
# ============================================================

paired_rows = []


for domain in DOMAINS:

    governed = all_runs[
        (domain, "governed")
    ].copy()

    ungoverned = all_runs[
        (domain, "ungoverned")
    ].copy()

    # Only Utility cases
    governed = governed[
        governed["query_type"] == "utility"
    ].copy()

    ungoverned = ungoverned[
        ungoverned["query_type"] == "utility"
    ].copy()

    # Select relevant columns before merging
    governed_small = governed[
        [
            "checkpoint_id",
            "expected_action",
            "query_text",
            "pred_action",
            "answer",
            "utility_ok",
            "judge_notes",
            "n_used_memories"
        ]
    ].copy()

    governed_small = governed_small.rename(
        columns={
            "pred_action":
                "governed_action",

            "answer":
                "governed_answer",

            "utility_ok":
                "governed_utility_ok",

            "judge_notes":
                "governed_judge_notes",

            "n_used_memories":
                "governed_used_memories"
        }
    )

    ungoverned_small = ungoverned[
        [
            "checkpoint_id",
            "pred_action",
            "answer",
            "utility_ok",
            "judge_notes",
            "n_used_memories"
        ]
    ].copy()

    ungoverned_small = ungoverned_small.rename(
        columns={
            "pred_action":
                "ungoverned_action",

            "answer":
                "ungoverned_answer",

            "utility_ok":
                "ungoverned_utility_ok",

            "judge_notes":
                "ungoverned_judge_notes",

            "n_used_memories":
                "ungoverned_used_memories"
        }
    )

    paired = governed_small.merge(
        ungoverned_small,
        on="checkpoint_id",
        how="inner"
    )

    # The most important conservative-governance cases:
    #
    # 1. Query was expected to be answerable
    # 2. Ungoverned was judged correct
    # 3. Governed was judged incorrect
    # 4. Governed refused or said no_memory
    conservative_cases = paired[
        (
            paired["expected_action"]
            .isin(ANSWER_ACTIONS)
        )
        &
        (
            paired["ungoverned_utility_ok"]
            == True
        )
        &
        (
            paired["governed_utility_ok"]
            == False
        )
        &
        (
            paired["governed_action"]
            .isin(REFUSAL_ACTIONS)
        )
    ].copy()

    conservative_cases[
        "domain"
    ] = domain.capitalize()

    paired_rows.append(
        conservative_cases
    )


# Combine all domains
if paired_rows:

    paired_conservative_df = pd.concat(
        paired_rows,
        ignore_index=True
    )

else:

    paired_conservative_df = pd.DataFrame()


print("\n")
print("=" * 90)
print(
    "UNGOVERNED CORRECT BUT GOVERNED REFUSED / NO_MEMORY"
)
print("=" * 90)


if len(paired_conservative_df) == 0:

    print(
        "No matching cases found."
    )

else:

    case_count_by_domain = (
        paired_conservative_df
        .groupby("domain")
        .size()
        .reset_index(name="Cases")
    )

    print(
        case_count_by_domain.to_string(
            index=False
        )
    )

    print(
        "\nTotal cases:",
        len(paired_conservative_df)
    )


# ============================================================
# Additional comparison:
# Governed correct while Ungoverned incorrect
#
# This is useful for finding examples where governance improves
# the outcome rather than merely refusing.
# ============================================================

governance_benefit_rows = []


for domain in DOMAINS:

    governed = all_runs[
        (domain, "governed")
    ].copy()

    ungoverned = all_runs[
        (domain, "ungoverned")
    ].copy()

    governed_small = governed[
        [
            "checkpoint_id",
            "query_type",
            "expected_action",
            "query_text",
            "pred_action",
            "answer",
            "utility_ok",
            "privacy_leak",
            "deletion_leak",
            "judge_notes"
        ]
    ].copy()

    governed_small = governed_small.rename(
        columns={
            "pred_action":
                "governed_action",

            "answer":
                "governed_answer",

            "utility_ok":
                "governed_utility_ok",

            "privacy_leak":
                "governed_privacy_leak",

            "deletion_leak":
                "governed_deletion_leak",

            "judge_notes":
                "governed_judge_notes"
        }
    )

    ungoverned_small = ungoverned[
        [
            "checkpoint_id",
            "pred_action",
            "answer",
            "utility_ok",
            "privacy_leak",
            "deletion_leak",
            "judge_notes"
        ]
    ].copy()

    ungoverned_small = ungoverned_small.rename(
        columns={
            "pred_action":
                "ungoverned_action",

            "answer":
                "ungoverned_answer",

            "utility_ok":
                "ungoverned_utility_ok",

            "privacy_leak":
                "ungoverned_privacy_leak",

            "deletion_leak":
                "ungoverned_deletion_leak",

            "judge_notes":
                "ungoverned_judge_notes"
        }
    )

    paired = governed_small.merge(
        ungoverned_small,
        on="checkpoint_id",
        how="inner"
    )

    # Privacy cases:
    # Ungoverned leaks but Governed does not
    privacy_benefit = paired[
        (
            paired["query_type"] == "privacy"
        )
        &
        (
            paired["ungoverned_privacy_leak"]
            == True
        )
        &
        (
            paired["governed_privacy_leak"]
            == False
        )
    ].copy()

    privacy_benefit[
        "governance_benefit_type"
    ] = "prevented_privacy_violation"

    # Safety/deletion cases:
    # Ungoverned leaks stale/deleted info but Governed does not
    deletion_benefit = paired[
        (
            paired["ungoverned_deletion_leak"]
            == True
        )
        &
        (
            paired["governed_deletion_leak"]
            == False
        )
    ].copy()

    deletion_benefit[
        "governance_benefit_type"
    ] = "prevented_forgetting_failure"

    combined = pd.concat(
        [
            privacy_benefit,
            deletion_benefit
        ],
        ignore_index=True
    )

    combined[
        "domain"
    ] = domain.capitalize()

    governance_benefit_rows.append(
        combined
    )


if governance_benefit_rows:

    governance_benefit_df = pd.concat(
        governance_benefit_rows,
        ignore_index=True
    )

    # Remove duplicate checkpoint/type pairs
    governance_benefit_df = (
        governance_benefit_df
        .drop_duplicates(
            subset=[
                "checkpoint_id",
                "governance_benefit_type"
            ]
        )
    )

else:

    governance_benefit_df = pd.DataFrame()


# ============================================================
# Export results
# ============================================================

OUTPUT_DIR = ROOT / "error_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 1. Main quantitative summary
# ------------------------------------------------------------

summary_path = (
    OUTPUT_DIR
    / "utility_error_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 2. Governed conservative-decision cases
# ------------------------------------------------------------

conservative_path = (
    OUTPUT_DIR
    / "governed_conservative_failure_cases.csv"
)

paired_conservative_df.to_csv(
    conservative_path,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 3. Cases showing actual governance benefit
# ------------------------------------------------------------

benefit_path = (
    OUTPUT_DIR
    / "governance_benefit_cases.csv"
)

governance_benefit_df.to_csv(
    benefit_path,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 4. All detailed Utility outcomes
# ------------------------------------------------------------

all_utility_details = pd.concat(
    detailed_utility_rows,
    ignore_index=True
)

utility_details_path = (
    OUTPUT_DIR
    / "all_utility_case_outcomes.csv"
)

all_utility_details.to_csv(
    utility_details_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Final report
# ============================================================

print("\n")
print("=" * 90)
print("FILES GENERATED")
print("=" * 90)

print(
    f"1. {summary_path}"
)

print(
    f"2. {conservative_path}"
)

print(
    f"3. {benefit_path}"
)

print(
    f"4. {utility_details_path}"
)

print("\nAnalysis complete.")