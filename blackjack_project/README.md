# Autonomous Systems: Blackjack Rule-Based AI
Dit project bevat een rule-based AI-systeem dat verschillende strategieën voor het spel Blackjack simuleert en valideert binnen een Gymnasium-omgeving. Het doel is om aan te tonen dat een expert-systeem (gebaseerd op de Blackjack Apprenticeship methode) beter presteert dan een willekeurige of simpele baseline.

# 1. Root Directory (Hoofdmap)
De bovenste laag van je project bevat de bestanden die nodig zijn om de simulatie te starten en de afhankelijkheden te beheren.

main.py: Het centrale script dat de Gymnasium-omgeving aanroeert, de 5 agents aanstuurt, 10.000 rondes per agent simuleert en de eindtabel en grafiek genereert.

requirements.txt: Een tekstbestand met de benodigde Python-bibliotheken (gymnasium, matplotlib, pandas, numpy) voor eenvoudige installatie.

# 2. De environment/ Module
Deze map beheert de interactie met de simulatie-omgeving.

__init__.py: Markeert deze map als een Python-pakket (leeg bestand).

blackjack_env.py: Bevat de wrapper-klasse voor de officiële Gymnasium Blackjack-v1 omgeving. Hierin worden de reset() en step() functies beheerd.

# 3. De agents/ Module (Rule-Based Logica)
Deze map bevat het "brein" van je systeem, opgedeeld in 5 specifieke bestanden om modulariteit te waarborgen.

__init__.py: Markeert deze map als een Python-pakket (leeg bestand).

baseline_agent.py: Bevat de Baseline (Expert 0). Een simpele strategie die altijd stopt op 17 en hitten tot 16, ongeacht de dealer.

hard_total_agent.py: Bevat Expert 1. Gebruikt basisregels voor harde totalen (handen zonder bruikbare Aas) en houdt rekening met de dealer-kaart bij lage scores.

soft_total_agent.py: Bevat Expert 2. Specifieke logica voor "Soft" handen (met een Aas), waarbij agressiever wordt gespeeld omdat de speler niet direct "bust" kan gaan.

dealer_aware_agent.py: Bevat Expert 3. Een strategie die de eigen acties volledig aanpast aan de zwakte (2-6) of sterkte (7-A) van de dealer.

pro_bja_agent.py: Bevat Expert 4 (Master Agent). Dit is het eindproduct dat de logica van de andere experts combineert in een hiërarchische beslisboom (eerst Soft-check, dan Hard/Dealer-check).

# 4. Documentatie
README.md: Het bestand waarin je uitlegt hoe de code werkt, hoe je het installeert en wat de resultaten van de simulatie betekenen voor je onderzoek.