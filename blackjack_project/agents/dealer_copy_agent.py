def dealer_copy_policy(observation):
    player_score, _, _ = observation
    
    # De dealer moet in Blackjack (ALE) altijd hitten tot minimaal 17.
    # Deze agent doet exact hetzelfde, ongeacht zijn eigen kaarten of de dealer-upcard.
    if player_score < 17:
        return 1  # Hit
    return 0      # Stand