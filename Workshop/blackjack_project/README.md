# Blackjack RL - Geiser & Hasseler (Stanford)

Recreated from: "Beating Blackjack - A Reinforcement Learning Approach"

## Requirements
```
pip install numpy pandas
```

## Run Order

### Step 1: Generate training data
In `blackjack.py`, make sure `action_type = 'random_policy'` and run:
```
python blackjack.py
```
This generates `random_policy_runs_mapping_1.csv` and/or `random_policy_runs_mapping_2.csv`
(change `state_mapping` to 1 or 2 at the top of the file).

### Step 2: Train a policy
Run any of the three algorithms (set `state_mapping` at the top of each file):
```
python value_iteration.py
python sarsa.py
python q_learning.py
```
Each outputs a `.policy` file.

### Step 3: Evaluate a policy
In `blackjack.py`, set:
- `action_type = 'fixed_policy'`
- `fixed_policy_filepath` to your `.policy` file
- `num_games = 20000`

Then run:
```
python blackjack.py
```

## Notes
- State mapping 1: agent's hand only (21 states)
- State mapping 2: agent's hand + dealer's upcard (183 states) — better results
- State mapping 2 requires 100,000 training games for good coverage
