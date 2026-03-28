import random

def probabilistic_policy(observation):
    player_score, dealer_card, _ = observation
    
    # Harde regels
    if player_score <= 11: return 1 # Altijd hit
    if player_score >= 17: return 0 # Altijd stand
    
    # De "Grijze Zone" (12-16): 
    # Als de dealer een sterke kaart heeft (>=7), hit 80% van de tijd.
    # Als de dealer een zwakke kaart heeft, hit slechts 20% van de tijd.
    if dealer_card >= 7:
        return 1 if random.random() < 0.8 else 0
    else:
        return 1 if random.random() < 0.2 else 0