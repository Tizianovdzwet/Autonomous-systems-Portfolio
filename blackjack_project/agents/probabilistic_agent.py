import random

def probabilistic_policy(observation):
    """
    Bonus: Stochastic Rule.
    Deze agent heeft de harde regels als volgt:
        - Als de score 11 of minder is, hit dan altijd.
        - Als de score 17 of hoger is, stand dan altijd.
    Daarnaast, als de score van de speler tussen de 11 of 17 is, wordt er gekeken naar de kaart van de dealer, specifiek of het boven de 7 of hoger is:
        - bij hoger, hit 80% van de tijd
        - bij lager, hit 20% van de tijd.
    """
    
    # observation = (speler_punten, dealer_kaart, heeft_aas)
    player_score, dealer_card, _ = observation
    
    # Harde regels
    if player_score <= 11: return 1
    if player_score >= 17: return 0
    
    # De "Grijze Zone" (12-16): 
    if dealer_card >= 7:
        return 1 if random.random() < 0.8 else 0
    else:
        return 1 if random.random() < 0.2 else 0