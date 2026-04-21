
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the Q-table

with open("./llm-guided-curriculum-rl-main/logs/logs-20260419-standard-8-deck-0.9-no-curriculum\models\standard_agent_tabular_1.pkl", "rb") as f:
    q_table  = pickle.load(f)
# Extract best actions for "clean" states (no usable ace, no special flags)
# State tuple: (player_sum, dealer_card, usable_ace, can_double, can_split, is_blackjack)
action_labels = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split", 4: "Surrender", 5: "???"}
colors =        {0: "red",   1: "green", 2: "orange", 3: "purple", 4: "gray",    5: "white"}

player_sums = range(4, 22)
dealer_cards = range(2, 12)

grid = np.full((len(player_sums), len(dealer_cards)), -1)

for i, ps in enumerate(player_sums):
    for j, dc in enumerate(dealer_cards):
        state = (ps, dc, False, False, False, False)
        # Find all actions for this state
        actions = {k[1]: v for k, v in q_table.items() if k[0] == state}
        if actions:
            best_action = max(actions, key=actions.get)
            grid[i, j] = best_action

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
color_grid = np.vectorize(lambda x: list(colors.values())[x] if x >= 0 else "white")

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        action = grid[i, j]
        color = colors.get(action, "white")
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        if action >= 0:
            ax.text(j + 0.5, i + 0.5, action_labels[action],
                    ha='center', va='center', fontsize=7, fontweight='bold')

ax.set_xlim(0, len(dealer_cards))
ax.set_ylim(0, len(player_sums))
ax.set_xticks(np.arange(len(dealer_cards)) + 0.5)
ax.set_xticklabels(list(dealer_cards))
ax.set_yticks(np.arange(len(player_sums)) + 0.5)
ax.set_yticklabels(list(player_sums))
ax.set_xlabel("Dealer Upcard")
ax.set_ylabel("Player Sum")
ax.set_title("Learned Policy (No Usable Ace, No Special Flags)")

legend = [mpatches.Patch(color=c, label=l) for l, c in zip(action_labels.values(), colors.values())]
ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig("policy_heatmap.png", dpi=150)
plt.show()