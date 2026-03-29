def soft_total_policy(observation):
    """
    Expert 2: Soft Totals
    Deze agent kijkt of er een aas is, als deze er niet is, EN de score is onder de 17, zal het altijd hitten.
    Verder als er wel een aas is, zal de speler boven de 18 nooit hitten, daaronder wordt er gekeken naar de dealer:
        - Als de dealer een een 8 of lager heeft, zal de speler niet hitten met een 18
        - Als de dealer wel hoger dan een 9 heeft, zal de speler hitten om te proberen de dealer voor te zijn.
    """

    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, dealer_card, has_ace = observation
    
    if not has_ace:
        return 1 if player_score < 17 else 0 # Fallback

    # BJA Soft regels
    if player_score >= 19:
        return 0 # Stand
    if player_score == 18 and dealer_card <= 8:
        return 0 # Stand
    return 1 # Hit