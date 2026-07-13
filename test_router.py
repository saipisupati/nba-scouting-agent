import pandas as pd
from query_router import route

df = pd.read_csv("hustle_stats_2025_26.csv")

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
]

for i, q in enumerate(questions, 1):
    result = route(q, df)
    print(f"\n{'='*70}")
    print(f"Q{i}: {result['question']}")
    print(f"  Routing method : {result['method']}")
    print(f"  Function matched: {result['function_matched']}")
    print(f"  Answer: {result['answer']}")
