import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importeer je agents
from agents.baseline_agent import baseline_policy
from agents.hard_total_agent import hard_total_policy
from agents.soft_total_agent import soft_total_policy
from agents.dealer_aware_agent import dealer_aware_policy
from agents.pro_bja_agent import bja_policy

def run_simulation(policy_func, name, num_episodes=10000):
    """Runt de simulatie voor een specifieke agent in de Gymnasium omgeving."""
    env = gym.make("Blackjack-v1")
    results = []
    cumulative_profit = 0
    history = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = policy_func(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        
        results.append(reward)
        cumulative_profit += reward
        history.append(cumulative_profit)

    env.close()
    
    win_rate = len([r for r in results if r > 0]) / num_episodes * 100
    return {
        "name": name,
        "history": history,
        "total_profit": cumulative_profit,
        "win_rate": f"{win_rate:.2f}%",
        "avg_reward": f"{cumulative_profit/num_episodes:.4f}"
    }

if __name__ == "__main__":
    # Configuratie
    ROUNDS = 10000
    RESULTS_DIR = "results"
    
    # Zorg dat de results map bestaat
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f" Map '{RESULTS_DIR}' aangemaakt.")

    # Lijst met alle agents voor de vergelijking
    agents_to_test = [
        (baseline_policy, "Baseline (Hit < 17)"),
        (hard_total_policy, "Expert 1: Hard Totals"),
        (soft_total_policy, "Expert 2: Soft Totals"),
        (dealer_aware_policy, "Expert 3: Dealer Aware"),
        (bja_policy, "Expert 4: Pro BJA (Combined)") # Let op: check of de importnaam klopt met je bestand
    ]

    all_stats = []
    plt.figure(figsize=(12, 7))

    print(f" Simulatie gestart: {ROUNDS} rondes per agent...\n")

    for policy, name in agents_to_test:
        print(f"Systeem test: {name}...")
        res = run_simulation(policy, name, ROUNDS)
        
        all_stats.append({
            "Agent": res["name"],
            "Total Profit": res["total_profit"],
            "Win Rate": res["win_rate"],
            "Expected Value": res["avg_reward"]
        })
        
        plt.plot(res["history"], label=res["name"])

    # 1. Tabel genereren en opslaan
    df = pd.DataFrame(all_stats)
    print("\n--- PERFORMANCE RESULTATEN ---")
    print(df.to_string(index=False))
    
    csv_path = os.path.join(RESULTS_DIR, "performance_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Tabel opgeslagen in: {csv_path}")

    # 2. Grafiek opmaken en opslaan
    plt.title(f"Cumulatief Rendement: Baseline vs Expert Agents ({ROUNDS} ronden)")
    plt.xlabel("Aantal Gespeelde Handen")
    plt.ylabel("Winst / Verlies (Units)")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(RESULTS_DIR, "blackjack_comparison_plot.png")
    plt.savefig(plot_path)
    print(f" Grafiek opgeslagen in: {plot_path}")

    # Toon de grafiek aan de gebruiker
    plt.show()