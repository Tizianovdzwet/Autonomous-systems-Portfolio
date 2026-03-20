def hard_total_policy(observation):
    player_score, dealer_card, _ = observation
    
    if player_score >= 17:
        return 0 # Stand
    if 13 <= player_score <= 16 and dealer_card <= 6:
        return 0 # Stand
    return 1 # Hit