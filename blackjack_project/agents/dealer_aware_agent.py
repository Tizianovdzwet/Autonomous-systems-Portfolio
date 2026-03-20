def dealer_aware_policy(observation):
    """
    Expert Agent 3: Dealer-Aware Strategy.
    Deze agent past zijn agressiviteit aan op de 'upcard' van de dealer.
    Focus: Risico nemen als de dealer een sterke kaart heeft (7-A), 
    en veilig spelen als de dealer een zwakke kaart heeft (2-6).
    """
    player_score, dealer_card, has_ace = observation

    # 1. Als de dealer een zwakke kaart heeft (2 t/m 6), verwachten we dat hij 'bust'.
    # We stoppen dan al vroeg (vanaf 13 punten) om zelf niet kapot te gaan.
    if dealer_card <= 6:
        if player_score >= 13:
            return 0  # Stand
        return 1      # Hit

    # 2. Als de dealer een sterke kaart heeft (7 t/m 11), moet de speler 
    # agressiever zijn om een hogere score te halen.
    else:
        if player_score >= 17:
            return 0  # Stand
        return 1      # Hit (We hitten door tot we minimaal 17 hebben)