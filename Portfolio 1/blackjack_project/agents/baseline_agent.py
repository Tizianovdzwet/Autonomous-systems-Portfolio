def baseline_policy(observation):
    """
    Baseline Agent.
    Deze agent hit altijd onder 17, ongeacht wat er in het spel gebeurt.
    """

    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, _, _ = observation
    
    # Altijd hitten onder de 17
    return 1 if player_score < 17 else 0