def hard_total_policy(observation):
    """
    Expert agent 1: Hard Totals.
    Deze agent focust op de totale resultaten van zichzelf en de dealer.
    Als zijn score boven de 17 is, zal hij altijd passen.
    Als de score van de agent tussen de 12 en 17 is, zal het doorgaan, mits de kaart van de dealer 6 of minder is.
    """
    
    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, dealer_card, _ = observation
    
    if player_score >= 17:
        return 0
    if 12 < player_score < 17 and dealer_card <= 6:
        return 0 
    return 1