import random
from models import FootballModel
from collections import defaultdict

def simulate_match(model: FootballModel, team_a: int, team_b: int) -> int:
    """Returns 1 if team_a wins, 0 if draw, -1 if team_b wins"""
    p_win, p_draw, p_lose = model.predict_poisson(team_a, team_b, is_neutral=True)
    r = random.random()
    if r < p_win:
        return 1
    if r < p_win + p_draw:
        return 0
    return -1

def simulate_knockout_match(model: FootballModel, team_a: int, team_b: int) -> int:
    """Returns the winner team_id. No draws allowed in knockouts."""
    p_win, p_draw, p_lose = model.predict_poisson(team_a, team_b, is_neutral=True)
    
    # Re-normalize ignoring draw
    total_decisive = p_win + p_lose
    if total_decisive == 0:
        return random.choice([team_a, team_b])
        
    p_win_decisive = p_win / total_decisive
    if random.random() < p_win_decisive:
        return team_a
    return team_b

def simulate_group_stage(model: FootballModel, groups: list) -> list:
    """Simulates 8 groups and returns 16 advancing teams"""
    advancing_teams = []
    
    for group_idx, group in enumerate(groups):
        points = {t: 0 for t in group}
        
        # Round robin
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                t1, t2 = group[i], group[j]
                res = simulate_match(model, t1, t2)
                if res == 1:
                    points[t1] += 3
                elif res == 0:
                    points[t1] += 1
                    points[t2] += 1
                else:
                    points[t2] += 3
                    
        # Sort by points, break ties randomly
        sorted_group = sorted(group, key=lambda x: (points[x], random.random()), reverse=True)
        advancing_teams.append((sorted_group[0], sorted_group[1])) # (Winner, Runner-up)
        
    return advancing_teams

def simulate_knockouts(model: FootballModel, group_results: list) -> int:
    """Takes 8 group results (A-H) and plays through to the final winner."""
    # Round of 16 Bracket
    # Left side
    r16 = [
        (group_results[0][0], group_results[1][1]), # A1 vs B2
        (group_results[2][0], group_results[3][1]), # C1 vs D2
        (group_results[4][0], group_results[5][1]), # E1 vs F2
        (group_results[6][0], group_results[7][1]), # G1 vs H2
        # Right side
        (group_results[1][0], group_results[0][1]), # B1 vs A2
        (group_results[3][0], group_results[2][1]), # D1 vs C2
        (group_results[5][0], group_results[4][1]), # F1 vs E2
        (group_results[7][0], group_results[6][1])  # H1 vs G2
    ]
    
    # Simulate to Final
    current_round = r16
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            match1 = current_round[i]
            match2 = current_round[i+1]
            
            winner1 = simulate_knockout_match(model, match1[0], match1[1])
            winner2 = simulate_knockout_match(model, match2[0], match2[1])
            
            next_round.append((winner1, winner2))
        current_round = next_round
        
    # Final Match
    final = current_round[0]
    champion = simulate_knockout_match(model, final[0], final[1])
    return champion

def monte_carlo_world_cup(model: FootballModel, n_sims: int = 1000) -> list:
    valid_teams = [k for k, v in model.team_stats.items() if v["matches"] >= 5]
    sorted_teams = sorted(valid_teams, key=lambda x: model.elo_ratings[x], reverse=True)
    
    if len(sorted_teams) < 32:
        print("Not enough teams with sufficient matches to run 32-team simulation.")
        return []
        
    top_32 = sorted_teams[:32]
    
    # 模拟抽签 (按实力分 4 档)
    pots = [top_32[i:i+8] for i in range(0, 32, 8)]
    
    champion_counts = defaultdict(int)
    
    for _ in range(n_sims):
        # 每次模拟重新抽签，保证随机性
        groups = [[] for _ in range(8)]
        for pot in pots:
            shuffled_pot = list(pot)
            random.shuffle(shuffled_pot)
            for i in range(8):
                groups[i].append(shuffled_pot[i])
                
        # 1. 小组赛
        group_results = simulate_group_stage(model, groups)
        
        # 2. 淘汰赛
        champion = simulate_knockouts(model, group_results)
        champion_counts[champion] += 1
        
    results = [(model.team_names[k], v / n_sims, v) for k, v in champion_counts.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    print("Loading model, calculating ELO, and fitting Dixon-Coles MLE parameters...")
    model = FootballModel()
    model.fit("raw_fixtures.json")
    
    n_sims = 1000
    print(f"\nRunning {n_sims} Monte Carlo simulations of a full World Cup (Group Stage + Knockout)...")
    results = monte_carlo_world_cup(model, n_sims)
    
    print("\n🏆 --- 2026 World Cup Predicted Winner Probabilities (V3 Topology) --- 🏆")
    for i, (name, prob, wins) in enumerate(results[:15], 1):
        print(f"{i:2d}. {name:20s} | Win Probability: {prob*100:>5.1f}% | Simulated Wins: {wins}")
