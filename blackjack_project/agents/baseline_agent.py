def baseline_policy(observation):
    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, _, _ = observation
    
    # Altijd hitten onder de 17
    return 1 if player_score < 17 else 0