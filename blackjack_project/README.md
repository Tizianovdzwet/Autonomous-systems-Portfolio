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

__init__.py: Markeert de directory als een Python-pakket, waardoor de verschillende agents eenvoudig geïmporteerd kunnen worden in de hoofdsimulatie.

baseline_agent.py (Expert 0): Dient als het wetenschappelijke nulpunt. De agent hanteert een statische regel (stoppen bij 17), ongeacht de context van de dealer of de samenstelling van de eigen hand.

hard_total_agent.py (Expert 1): Implementeert specifieke heuristieken voor 'harde totalen' (handen zonder bruikbare Aas). Deze agent begint rekening te houden met de dealer-kaart bij kritieke lage scores.

soft_total_agent.py (Expert 2): Bevat logica voor 'zachte handen' waarbij een Aas aanwezig is. De agent speelt hierbij agressiever, aangezien de flexibele waarde van de Aas een vangnet biedt tegen direct verlies (bust).

dealer_aware_agent.py (Expert 3): Een context-bewuste strategie die zijn agressiviteit aanpast aan de waargenomen sterkte van de dealer (bijvoorbeeld behoudend spelen als de dealer een zwakke kaart zoals een 6 toont).

pro_bja_agent.py (Master Agent): Dit is de culminatie van de voorgaande logica. Het fungeert als een hiërarchische beslisboom die eerst controleert op 'soft totals' en vervolgens de 'hard/dealer' logica toepast voor een optimaal resultaat.

3.2 Experimentele Validatie-Agents
Om de effectiviteit van de expert-logica te toetsen, zijn twee extra gedragsmodellen toegevoegd:

safe_play_agent.py: Hanteert een extreem risicomijdend beleid door nooit te hitten bij een score van 12 of hoger. Deze agent dient als bewijsvoering dat het uitsluitend vermijden van fouten (busting) niet resulteert in een succesvolle strategie.

dealer_copy_agent.py: Kopieert exact het geprogrammeerde gedrag van het casino. Dit valideert of de regels die gunstig zijn voor 'het huis' ook optimaal zijn voor een individuele speler met een beperkt budget.

# 4. Documentatie
README.md: Het bestand waarin je uitlegt hoe de code werkt, hoe je het installeert en wat de resultaten van de simulatie betekenen voor je onderzoek.