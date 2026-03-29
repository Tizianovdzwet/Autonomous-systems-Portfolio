def dealer_copy_policy(observation):
    """
    Bonus: Dealer Mimic.
    Deze agent volgt exact wat de blackjack dealer doet, waar het tot 17 altijd hit. Deze agent doet exact hetzelfde, ongeacht zijn eigen kaarten of de dealer-upcard.
    """
    
    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, _, _ = observation
    
    if player_score < 17:
        return 1
    return 0