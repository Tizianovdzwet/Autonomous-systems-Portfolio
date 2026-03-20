def soft_total_policy(observation):
    player_score, dealer_card, has_ace = observation
    
    if not has_ace:
        return 1 if player_score < 17 else 0 # Fallback

    # BJA Soft regels
    if player_score >= 19:
        return 0 # Stand
    if player_score == 18 and dealer_card <= 8:
        return 0 # Stand
    return 1 # Hit