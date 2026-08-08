import pandas as pd
from query_router import route

df = pd.read_csv("hustle_stats_2025_26.csv")
prior_df = pd.read_csv("hustle_stats_2024_25.csv")

questions = [
    "Who leads the league in deflections per 36 minutes?",
    "Which players contest the most shots at the rim?",
    "Who are the best perimeter closeout defenders?",
    "Which player boxes out most effectively?",
    "Who has the highest defensive IQ based on hustle stats?",
    "Which guards draw the most charges?",
    "Who reads the game well defensively?",
    "Who is the best scorer in the league?",
    "What's the trade value of Donovan Clingan?",
    "Who should my team draft this year?",
    "How much is Victor Wembanyama worth in a trade?",
    "Which player has the best defensive instincts based on the numbers?",
    "What's Clingan's points per game?",
    "Which players suppress shooting the most at the rim?",
    "Who's an underrated defender that doesn't show up in the box score?",
    "Which player has good hustle numbers that don't translate to results?",
    "How does he defend the pick and roll ball handler?",
    "Who's the best roll man defender in the league?",
    "How good is he at defending cuts to the basket?",
    "Who navigates screens best on defense?",
    "Which player has improved the most at drawing charges this year?",
    "Who has declined the most in deflections?",
    "Is this player trending up or down defensively?",
    # ── Q24-Q29: playtype_offense + disambiguation ────────────────────────────
    "Who's the best scorer in isolation?",                          # Q24 → playtype_offense, Isolation
    "How does he score coming off screens?",                        # Q25 → playtype_offense, OffScreen
    "Who cuts the most?",                                           # Q26 → playtype_offense, Cut (volume)
    "Who's the best off-screen scorer?",                            # Q27 → playtype_offense, OffScreen
    "How does he defend the pick and roll?",                        # Q28 → playtype_defense, PRBallHandler (DISAMBIGUATION)
    "How does he score in the pick and roll as the ball handler?",  # Q29 → playtype_offense, PRBallHandler (DISAMBIGUATION)
    # ── Q30-Q33: additional PRBallHandler/PRRollman disambiguation ────────────
    "Is he a good pick and roll defender?",                         # Q30 → playtype_defense, PRBallHandler (DISAMBIGUATION)
    "Rate his offensive efficiency as a pick and roll ball handler", # Q31 → playtype_offense, PRBallHandler (DISAMBIGUATION)
    "Can he guard ball handlers in the pick and roll?",             # Q32 → playtype_defense, PRBallHandler (DISAMBIGUATION)
    "How efficient is he finishing as the roll man?",               # Q33 → playtype_offense, PRRollman (DISAMBIGUATION)
    # ── Q34-Q35: drive_efficiency ─────────────────────────────────────────────
    "Who's the most efficient driver to the basket?",               # Q34 → drive_efficiency
    "Which player scores the most points per drive?",               # Q35 → drive_efficiency
    # ── Q36-Q41: college draft-class layer ────────────────────────────────────
    "What were AJ Dybantsa's college stats?",                       # Q36 → college_player_lookup
    "What was Mikel Brown Jr.'s college season like?",              # Q37 → college_player_lookup (name suffix)
    "What were Karim Lopez's college stats?",                       # Q38 → college_player_lookup (international, no data)
    "Who had the highest scoring average in the 2026 draft class?", # Q39 → college_leaderboard, PTS
    "Who had the highest usage rate in this draft class?",          # Q40 → college_leaderboard, USG%
    "Who's a high-usage prospect that's also efficient in this draft class?",  # Q41 → college_efficiency_volume
    # ── Q42-Q43: signature_play_type (LLM-fallback only — no deterministic
    # rule, since extracting an arbitrary NBA player's name from free text
    # isn't something the regex router can do; requires GROQ_API_KEY to
    # route successfully, same as Q8-Q11/Q13/Q23 above) ──────────────────────
    "What's Stephen Curry's signature play type?",                  # Q42 → signature_play_type
    "What does Nikola Jokic do best offensively?",                  # Q43 → signature_play_type
]

for i, q in enumerate(questions, 1):
    result = route(q, df, prior_df=prior_df)
    print(f"\n{'='*70}")
    print(f"Q{i}: {result['question']}")
    print(f"  Routing method : {result['method']}")
    print(f"  Function matched: {result['function_matched']}")
    print(f"  Answer: {result['answer']}")
