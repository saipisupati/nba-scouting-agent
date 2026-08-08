"""
Script-style test for report.py, following the same convention as
test_router.py: run real known players through the real assembly functions
and check the output against expectations grounded in the README's own
documented findings, rather than mocking anything.

This exists because report.py's section-assembly logic (generate_scouting_report_data,
compare_players_data) had no test coverage at all before this — only
query_router.py's routing was exercised by test_router.py. Per README Principle 5
("AI-assisted development requires the same verification discipline as any other
output"), a function that runs without error and looks plausible is not the same
as a function that's actually correct — this checks known ground truth, not just
"did it crash."
"""

import sys

from report import generate_scouting_report_data, compare_players_data

EXPECTED_SECTION_TITLES = [
    "Hustle / Activity Profile",
    "Shot Suppression",
    "Defensive Play-Type Profile",
    "Offensive Play-Type Profile",
    "Drive Efficiency",
    "Signature Play Type",
    "Hustle-vs-Suppression Gap",
    "Year-Over-Year Trend",
]

failures = []


def check(condition: bool, message: str):
    if not condition:
        failures.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok: {message}")


# ── Test 1: report structure for a known, well-qualified player ──────────────
print("=" * 70)
print("Test 1: generate_scouting_report_data('Alex Caruso') — structure")
print("=" * 70)

data = generate_scouting_report_data("Alex Caruso", "2025-26")
check(data["player_name"] == "Alex Caruso", "player_name echoed correctly")
check(data["season"] == "2025-26", "season echoed correctly")
titles = [s["title"] for s in data["sections"]]
check(titles == EXPECTED_SECTION_TITLES, f"section titles/order match expected: {titles}")

hustle_section = data["sections"][0]
check(
    any(row["qualified"] for row in hustle_section["rows"]),
    "Caruso qualifies for at least one hustle row (he's a real, high-minutes NBA player)",
)

# README Principle 2: Caruso leads deflections but has unremarkable shot suppression —
# both sections should be present with actual numbers for him, not "insufficient sample".
shot_suppression_section = data["sections"][1]
check(
    any(row["qualified"] for row in shot_suppression_section["rows"]),
    "Caruso qualifies for shot suppression (README discusses his actual suppression numbers)",
)


# ── Test 2: unknown player returns unqualified rows, not a crash ─────────────
print("\n" + "=" * 70)
print("Test 2: generate_scouting_report_data() — nonexistent player name")
print("=" * 70)

data_unknown = generate_scouting_report_data("Not A Real Player Zzyzx", "2025-26")
all_unqualified = all(
    not row["qualified"] for section in data_unknown["sections"] for row in section["rows"]
)
check(all_unqualified, "unknown player yields all-unqualified rows across every section, no exception")


# ── Test 3: compare_players_data structure + winner logic ────────────────────
print("\n" + "=" * 70)
print("Test 3: compare_players_data('Alex Caruso', 'Donovan Clingan')")
print("=" * 70)

cmp = compare_players_data("Alex Caruso", "Donovan Clingan", "2025-26")
check(cmp["player_a"] == "Alex Caruso" and cmp["player_b"] == "Donovan Clingan", "player labels correct")
cmp_titles = [s["title"] for s in cmp["sections"]]
check(cmp_titles == EXPECTED_SECTION_TITLES, "compare sections match single-report sections (same order)")

for section in cmp["sections"]:
    for row in section["rows"]:
        check(
            row["winner"] in ("a", "b", None),
            f"[{section['title']}/{row['label']}] winner is a valid value: {row['winner']!r}",
        )
        if row["winner"] is not None:
            check(
                row["a"]["qualified"] and row["b"]["qualified"],
                f"[{section['title']}/{row['label']}] winner only set when both players qualify",
            )


# ── Test 4: Drive Efficiency section present with a real numeric value ───────
print("\n" + "=" * 70)
print("Test 4: Drive Efficiency section has a numeric PTS_PER_DRIVE value")
print("=" * 70)

drive_section = data["sections"][EXPECTED_SECTION_TITLES.index("Drive Efficiency")]
drive_row = drive_section["rows"][0]
check(drive_row["qualified"] is True, "Caruso qualifies for drive efficiency (real rotation player)")
check(
    isinstance(drive_row["value"], float) and drive_row["value"] > 0,
    f"drive efficiency value is a positive float: {drive_row['value']!r}",
)
check(drive_row["better"] == "higher", "drive efficiency uses 'higher is better' convention")


# ── Test 5: Signature Play Type section — Curry's real, verified ground truth ─
# 99.6% Handoff percentile is the same figure independently confirmed via
# compute_offense.signature_play_type() directly earlier in this session's
# testing (see AUDIT_AND_SIGNATURE_DUMP.txt) — reused here as ground truth
# rather than re-guessed.
print("\n" + "=" * 70)
print("Test 5: Signature Play Type section — Stephen Curry ground truth")
print("=" * 70)

curry_data = generate_scouting_report_data("Stephen Curry", "2025-26")
curry_titles = [s["title"] for s in curry_data["sections"]]
check(curry_titles == EXPECTED_SECTION_TITLES, f"Curry's section titles/order match expected: {curry_titles}")

sig_section = curry_data["sections"][EXPECTED_SECTION_TITLES.index("Signature Play Type")]
sig_row = sig_section["rows"][0]
check(sig_section["title"] == "Signature Play Type", "section title is exactly 'Signature Play Type'")
check(sig_row["qualified"] is True, "Curry qualifies for a signature play type")
check(
    "Handoff" in sig_row["text"] or "handoff" in sig_row["text"].lower(),
    f"Curry's signature is Handoff, per text: {sig_row['text']!r}",
)
check(
    sig_row["value"] is not None and sig_row["value"] > 0.99,
    f"Curry's signature percentile is >99% as expected: {sig_row['value']!r}",
)


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
