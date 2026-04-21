from agents.soft_total_agent import soft_total_policy
from agents.dealer_aware_agent import dealer_aware_policy

def bja_policy(observation):
    """
    Expert 4: Pro BJA (Combined)
    Deze agent kijkt eerst of de speler eerst een aas heeft, als dat niet zo is, gebruikt het de
    dealer aware policy (Agent Expert 3)
    """

    # observation = (speler_punten, dealer_kaart, heeft_aas)
    _, _, has_ace = observation
    
    if has_ace:
        return soft_total_policy(observation)
    
    return dealer_aware_policy(observation)