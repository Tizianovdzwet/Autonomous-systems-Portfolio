from agents.soft_total_agent import soft_total_policy
from agents.dealer_aware_agent import dealer_aware_policy

def bja_policy(observation):
    player_score, dealer_card, has_ace = observation
    
    # Prioriteit 1: Gebruik Soft-logica als we een Aas hebben
    if has_ace:
        return soft_total_policy(observation)
    
    # Prioriteit 2: Gebruik de Dealer-Aware strategie voor harde totalen
    return dealer_aware_policy(observation)