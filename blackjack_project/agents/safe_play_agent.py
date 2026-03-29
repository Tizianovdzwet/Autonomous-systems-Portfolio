def safe_play_policy(observation):
    """
    Bonus: Ultra-Conservative
    Zodra er een statistische kans is om boven de 21 te komen (vanaf 12 punten),
    stopt deze agent direct. Hij gokt erop dat de dealer kapot gaat.
    """

    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, _, _ = observation
    
    if player_score >= 12:
        return 0
    return 1