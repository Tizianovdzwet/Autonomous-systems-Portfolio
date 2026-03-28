def safe_play_policy(observation):
    player_score, _, _ = observation
    
    # Zodra er een statistische kans is om boven de 21 te komen (vanaf 12 punten),
    # stopt deze agent direct. Hij gokt erop dat de dealer kapot gaat.
    if player_score >= 12:
        return 0  # Stand
    return 1      # Hit