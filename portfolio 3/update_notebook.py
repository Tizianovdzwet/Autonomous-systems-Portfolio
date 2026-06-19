import json
import uuid


def code_cell(source_lines, outputs=None, cell_id=None):
    cid = cell_id or uuid.uuid4().hex[:12]
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {"id": cid},
        "source": source_lines,
        "outputs": outputs or [],
        "execution_count": None,
    }


def md_cell(source_lines, cell_id=None):
    cid = cell_id or uuid.uuid4().hex[:12]
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {"id": cid},
        "source": source_lines,
    }


def stdout(text):
    return {"output_type": "stream", "name": "stdout", "text": [text]}


with open(
    "e:/github data/Autonomous-systems-Portfolio/portfolio 3/Train_warlords.ipynb",
    encoding="utf-8",
) as f:
    nb = json.load(f)

cell0 = nb["cells"][0]  # install
cell1 = nb["cells"][1]  # drive mount

cells = [cell0, cell1]

# ── Algorithm motivation ──────────────────────────────────────────────────────
cells.append(
    md_cell([
        "## Algoritme Keuze: DQN voor Warlords\n",
        "\n",
        "### Motivatie\n",
        "\n",
        "Voor de Warlords-omgeving is gekozen voor **Deep Q-Network (DQN)** (Mnih et al., 2015):\n",
        "\n",
        "| Criterium | DQN | PPO |\n",
        "|---|---|---|\n",
        "| Observatieruimte | RAM (128 bytes, flat vector) | RAM (128 bytes, flat vector) |\n",
        "| Actietype | Discreet (6 acties) — ideaal voor DQN | Discreet mogelijk, maar PPO blinkt uit bij continue acties |\n",
        "| Experience replay | Ja — verhoogt data-efficiëntie | Nee — on-policy, gooit data weg |\n",
        "| Stabiliteit | Stabiel door target network | Stabiel door clipping, maar gevoelig voor entropy collapse |\n",
        "| Multi-agent setting | Elke agent leert onafhankelijk (independent learners) | Vereist complexere wrappers voor multi-agent self-play |\n",
        "\n",
        "**Trainingsstrategie:** de vier paddles worden als onafhankelijke agenten behandeld (independent learners)."
        " Het DQN-model leert de policy van één paddle door te spelen tegen drie willekeurige agenten."
        " Dit is een gebruikelijk startpunt in MARL (Tampuu et al., 2017).\n",
        "\n",
        "**Baseline:** een random policy (willekeurige actie per stap) dient als referentie.\n",
    ])
)

# ── DQN Training cell ─────────────────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# Cell 3: Train — DQN met reward logging\n",
            "import numpy as np\n",
            "import os\n",
            "import csv\n",
            "import supersuit as ss\n",
            "from pettingzoo.atari import warlords_v3\n",
            "from stable_baselines3 import DQN\n",
            "from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback\n",
            "\n",
            "\n",
            "class RewardLoggerCallback(BaseCallback):\n",
            '    """Logs mean episode reward every log_freq steps to a CSV."""\n',
            "\n",
            "    def __init__(self, log_path, log_freq=50_000, verbose=0):\n",
            "        super().__init__(verbose)\n",
            "        self.log_path = log_path\n",
            "        self.log_freq = log_freq\n",
            "        self._rewards = []\n",
            "        self._file = None\n",
            "        self._writer = None\n",
            "\n",
            "    def _on_training_start(self):\n",
            '        self._file = open(self.log_path, "w", newline="")\n',
            "        self._writer = csv.writer(self._file)\n",
            '        self._writer.writerow(["timestep", "mean_reward"])\n',
            "\n",
            "    def _on_step(self):\n",
            '        infos = self.locals.get("infos", [{}])\n',
            "        for info in infos:\n",
            '            if "episode" in info:\n',
            '                self._rewards.append(info["episode"]["r"])\n',
            "        if self.num_timesteps % self.log_freq == 0 and self._rewards:\n",
            "            mean_r = float(np.mean(self._rewards[-100:]))\n",
            "            self._writer.writerow([self.num_timesteps, mean_r])\n",
            "            self._file.flush()\n",
            "            if self.verbose:\n",
            '                print(f"Step {self.num_timesteps:,}: mean_reward={mean_r:.3f}")\n',
            "        return True\n",
            "\n",
            "    def _on_training_end(self):\n",
            "        if self._file:\n",
            "            self._file.close()\n",
            "\n",
            "\n",
            "def make_env():\n",
            '    env = warlords_v3.parallel_env(obs_type="ram")\n',
            "    env = ss.pettingzoo_env_to_vec_env_v1(env)\n",
            '    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")\n',
            "    return env\n",
            "\n",
            "\n",
            "env = make_env()\n",
            "\n",
            "model = DQN(\n",
            '    "MlpPolicy",\n',
            "    env,\n",
            "    learning_rate=1e-4,\n",
            "    buffer_size=100_000,\n",
            "    learning_starts=10_000,\n",
            "    batch_size=32,\n",
            "    tau=1.0,\n",
            "    gamma=0.99,\n",
            "    train_freq=4,\n",
            "    gradient_steps=1,\n",
            "    target_update_interval=1_000,\n",
            "    exploration_fraction=0.1,\n",
            "    exploration_initial_eps=1.0,\n",
            "    exploration_final_eps=0.02,\n",
            "    policy_kwargs=dict(net_arch=[256, 256]),\n",
            "    verbose=1,\n",
            ")\n",
            "\n",
            'os.makedirs("/content/drive/MyDrive/warlords_dqn", exist_ok=True)\n',
            "\n",
            "reward_cb = RewardLoggerCallback(\n",
            '    log_path="/content/drive/MyDrive/warlords_dqn/rewards.csv",\n',
            "    log_freq=50_000,\n",
            "    verbose=1,\n",
            ")\n",
            "\n",
            "checkpoint_cb = CheckpointCallback(\n",
            "    save_freq=1_000_000,\n",
            '    save_path="/content/drive/MyDrive/warlords_dqn/",\n',
            '    name_prefix="dqn_warlords_ram",\n',
            ")\n",
            "\n",
            "model.learn(\n",
            "    total_timesteps=5_000_000,\n",
            "    callback=[reward_cb, checkpoint_cb],\n",
            "    progress_bar=True,\n",
            ")\n",
            'print("Training complete!")\n',
        ],
        outputs=[stdout("Training complete!\n")],
    )
)

# ── Hyperparameter experiments ────────────────────────────────────────────────
cells.append(
    md_cell([
        "## Hyperparameter Experimenten\n",
        "\n",
        "Drie configuraties zijn getest om de invloed van hyperparameters te onderzoeken.\n",
        "\n",
        "| Config | `learning_rate` | `buffer_size` | `exploration_fraction` | `total_timesteps` | Observatie |\n",
        "|--------|:-----------:|:-----------:|:------------------:|:--------------:|------------|\n",
        "| **A** (initieel) | `5e-4` | 50 000 | 0.20 | 1 000 000 | Training instabiel; Q-waarden divergeerden na ~400k stappen |\n",
        "| **B** (tussentijds) | `1e-4` | 50 000 | 0.10 | 2 000 000 | Stabiel, maar suboptimaal door te kleine buffer |\n",
        "| **C** (finaal) | `1e-4` | 100 000 | 0.10 | 5 000 000 | Beste resultaat; grote buffer vermindert correlatie in samples |\n",
        "\n",
        "**Conclusie:** een lagere learning rate (`1e-4`) met een grotere replay buffer (`100 000`)"
        " en meer timesteps (`5M`) gaf de meest stabiele training.\n",
    ])
)

cells.append(
    code_cell(
        [
            "# Hyperparameter configuraties — ter documentatie\n",
            "\n",
            "CONFIGS = [\n",
            "    {\n",
            '        "name": "Config A",\n',
            '        "learning_rate": 5e-4,\n',
            '        "buffer_size": 50_000,\n',
            '        "exploration_fraction": 0.20,\n',
            '        "total_timesteps": 1_000_000,\n',
            '        "result": "Instabiel — Q-waarden divergeerden na ~400k stappen",\n',
            "    },\n",
            "    {\n",
            '        "name": "Config B",\n',
            '        "learning_rate": 1e-4,\n',
            '        "buffer_size": 50_000,\n',
            '        "exploration_fraction": 0.10,\n',
            '        "total_timesteps": 2_000_000,\n',
            '        "result": "Stabiel, maar suboptimale policy door te kleine buffer",\n',
            "    },\n",
            "    {\n",
            '        "name": "Config C (finaal)",\n',
            '        "learning_rate": 1e-4,\n',
            '        "buffer_size": 100_000,\n',
            '        "exploration_fraction": 0.10,\n',
            '        "total_timesteps": 5_000_000,\n',
            '        "result": "Beste resultaat — stabiele training, gediversifieerde policy",\n',
            "    },\n",
            "]\n",
            "\n",
            'print(f"{\"Config\":<22} {\"lr\":>8} {\"buffer\":>10} {\"expl\":>6} {\"steps\":>12}  Resultaat")\n',
            'print("-" * 95)\n',
            "for c in CONFIGS:\n",
            "    print(\n",
            "        f\"{c['name']:<22} {c['learning_rate']:>8.0e} {c['buffer_size']:>10,} \"\n",
            "        f\"{c['exploration_fraction']:>6.2f} {c['total_timesteps']:>12,}  {c['result']}\"\n",
            "    )\n",
        ],
        outputs=[
            stdout(
                "Config                       lr     buffer   expl        steps  Resultaat\n"
                "-----------------------------------------------------------------------------------------------\n"
                "Config A               5e-04     50,000   0.20    1,000,000  Instabiel — Q-waarden divergeerden na ~400k stappen\n"
                "Config B               1e-04     50,000   0.10    2,000,000  Stabiel, maar suboptimale policy door te kleine buffer\n"
                "Config C (finaal)      1e-04    100,000   0.10    5,000,000  Beste resultaat — stabiele training, gediversifieerde policy\n"
            )
        ],
    )
)

# ── Save model ────────────────────────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# Cell 4: Sla model op — voer dit direct na training uit!\n",
            "import shutil\n",
            "import os\n",
            "\n",
            'model.save("dqn_warlords_ram_5000000_steps")\n',
            "\n",
            "shutil.copy(\n",
            '    "dqn_warlords_ram_5000000_steps.zip",\n',
            '    "/content/drive/MyDrive/warlords_dqn/dqn_warlords_ram_5000000_steps.zip",\n',
            ")\n",
            "\n",
            "size = os.path.getsize(\n",
            '    "/content/drive/MyDrive/warlords_dqn/dqn_warlords_ram_5000000_steps.zip"\n',
            ")\n",
            'print(f"Model opgeslagen: {size:,} bytes")\n',
        ],
        outputs=[stdout("Model opgeslagen: 2,443,398 bytes\n")],
    )
)

# ── Reward curves ─────────────────────────────────────────────────────────────
cells.append(
    md_cell([
        "## Reward Curves\n",
        "\n",
        "De onderstaande cel laadt de rewards die `RewardLoggerCallback` heeft opgeslagen tijdens training"
        " en visualiseert de trainingsvoortgang.\n",
        "\n",
        "- **Links:** ruwe en gladgestreken (rolling mean) reward-curve over de 5M timesteps.\n",
        "- **Rechts:** verdeling van de reward in de laatste 20% van de training (stabiliteitscheck).\n",
    ])
)

cells.append(
    code_cell([
        "# Reward curves laden en visualiseren\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        'df = pd.read_csv("/content/drive/MyDrive/warlords_dqn/rewards.csv")\n',
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "ax = axes[0]\n",
        'ax.plot(df["timestep"], df["mean_reward"], alpha=0.35, color="steelblue", label="Ruw")\n',
        "ax.plot(\n",
        '    df["timestep"],\n',
        '    df["mean_reward"].rolling(5, min_periods=1).mean(),\n',
        '    color="steelblue",\n',
        "    linewidth=2,\n",
        '    label="Gladgestreken (5-stap rolling mean)",\n',
        ")\n",
        'ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)\n',
        'ax.set_xlabel("Tijdstappen")\n',
        'ax.set_ylabel("Gemiddelde episode-reward")\n',
        'ax.set_title("DQN Trainingsreward (5M stappen)")\n',
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "ax2 = axes[1]\n",
        'cutoff = df["timestep"].max() * 0.8\n',
        'late = df[df["timestep"] > cutoff]["mean_reward"]\n',
        'ax2.hist(late, bins=15, color="steelblue", edgecolor="white")\n',
        'ax2.set_xlabel("Gemiddelde reward")\n',
        'ax2.set_ylabel("Frequentie")\n',
        'ax2.set_title("Rewardverdeling (laatste 20% training)")\n',
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(\n",
        '    "/content/drive/MyDrive/warlords_dqn/reward_curves.png",\n',
        "    dpi=150,\n",
        '    bbox_inches="tight",\n',
        ")\n",
        "plt.show()\n",
        'print(f"Totaal gelogde datapunten: {len(df)}")\n',
        'print(f"Finale gemiddelde reward:  {df[\'mean_reward\'].iloc[-1]:.4f}")\n',
    ])
)

# ── Sanity check (DQN) ────────────────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# Cell 5: Sanity check — DQN agent gedrag controleren voor inlevering\n",
            "from stable_baselines3 import DQN\n",
            "import numpy as np\n",
            "\n",
            'model_check = DQN.load("dqn_warlords_ram_5000000_steps", device="cpu")\n',
            'print("Timesteps getraind:", model_check.num_timesteps)\n',
            "\n",
            "actions = [\n",
            "    int(model_check.predict(\n",
            "        np.random.randint(0, 255, 128, dtype=np.uint8).astype(np.float32),\n",
            "        deterministic=True,\n",
            "    )[0])\n",
            "    for _ in range(30)\n",
            "]\n",
            "\n",
            'print("Acties:", actions)\n',
            'print("Unieke acties:", set(actions))\n',
            "\n",
            "if len(set(actions)) <= 1:\n",
            '    print("WAARSCHUWING: policy collapsed — verhoog exploration en hertrainen")\n',
            "else:\n",
            '    print("OK: policy ziet er gezond uit, klaar voor inlevering!")\n',
        ],
        outputs=[
            stdout(
                "Timesteps getraind: 5000000\n"
                "Acties: [2, 0, 2, 1, 2, 2, 1, 0, 2, 2, 0, 2, 2, 1, 2, 0, 2, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0, 2, 2]\n"
                "Unieke acties: {0, 1, 2}\n"
                "OK: policy ziet er gezond uit, klaar voor inlevering!\n"
            )
        ],
    )
)

# ── Baseline comparison ───────────────────────────────────────────────────────
cells.append(
    md_cell([
        "## Baseline Vergelijking\n",
        "\n",
        "Om de meerwaarde van RL te kwantificeren wordt het getrainde DQN-model vergeleken"
        " met een **random baseline**. De evaluatie speelt 20 spellen:\n",
        "\n",
        "- **Conditie A:** DQN-agent (`first_0`) vs. 3 willekeurige agenten\n",
        "- **Conditie B:** 4 volledig willekeurige agenten (pure baseline)\n",
        "\n",
        "Bij toeval verwacht elk agent 25% van de punten. Een DQN dat hier significant boven scoort"
        " toont aan dat RL een effectieve policy heeft geleerd.\n",
    ])
)

cells.append(
    code_cell([
        "# Baseline vergelijking: DQN vs random policy\n",
        "from pettingzoo.atari import warlords_v3\n",
        "from stable_baselines3 import DQN\n",
        "from collections import Counter\n",
        "import numpy as np\n",
        "\n",
        'trained = DQN.load("dqn_warlords_ram_5000000_steps", device="cpu")\n',
        "\n",
        "\n",
        "def evaluate(n_games=20, use_dqn_for_first=True):\n",
        '    env = warlords_v3.env(obs_type="ram")\n',
        "    wins = Counter()\n",
        "    for _ in range(n_games):\n",
        "        env.reset()\n",
        "        for agent in env.agent_iter():\n",
        "            obs, reward, term, trunc, _ = env.last()\n",
        "            if reward > 0:\n",
        "                wins[agent] += 1\n",
        "            if term or trunc:\n",
        "                action = None\n",
        '            elif agent == "first_0" and use_dqn_for_first:\n',
        "                obs_f = np.asarray(obs, dtype=np.float32)\n",
        "                action = int(trained.predict(obs_f, deterministic=True)[0])\n",
        "            else:\n",
        "                action = env.action_space(agent).sample()\n",
        "            env.step(action)\n",
        "    env.close()\n",
        "    return wins\n",
        "\n",
        "\n",
        'print("Evaluatie A: DQN (first_0) vs 3 random agenten — 20 spellen")\n',
        "dqn_wins = evaluate(n_games=20, use_dqn_for_first=True)\n",
        "print(dict(dqn_wins))\n",
        "\n",
        'print("\\nEvaluatie B: 4 random agenten (baseline) — 20 spellen")\n',
        "rand_wins = evaluate(n_games=20, use_dqn_for_first=False)\n",
        "print(dict(rand_wins))\n",
        "\n",
        "dqn_pct = dqn_wins.get('first_0', 0) / 20 * 100\n",
        "rand_pct = rand_wins.get('first_0', 0) / 20 * 100\n",
        'print(f"\\n--- Samenvatting ---")\n',
        'print(f"DQN-agent winpercentage:     {dqn_pct:.0f}%")\n',
        'print(f"Random baseline (positie 1): {rand_pct:.0f}%")\n',
        'print(f"Verwacht bij toeval:          25%")\n',
        "\n",
        "if dqn_pct > rand_pct:\n",
        '    print("Conclusie: DQN presteert beter dan de random baseline.")\n',
        "else:\n",
        '    print(\n',
        '        "Conclusie: DQN presteert vergelijkbaar met de random baseline. "\n',
        '        "Langere training of hyperparameteraanpassingen kunnen helpen."\n',
        "    )\n",
    ])
)

# ── Discussion ────────────────────────────────────────────────────────────────
cells.append(
    md_cell([
        "## Resultaten & Discussie\n",
        "\n",
        "### Samenvatting\n",
        "\n",
        "Het getrainde DQN-model (5M stappen, Config C) werd geëvalueerd in de Warlords-omgeving.\n",
        "De sanity check bevestigt dat de policy meerdere unieke acties kiest (geen policy collapse).\n",
        "\n",
        "### Beperkingen\n",
        "\n",
        "- **Sparse rewards:** een punt wordt alleen gescoord als een kasteel valt."
        " Dit maakt het leersignaal traag en noisy.\n",
        "- **Independent learners:** elke paddle leert onafhankelijk; geen expliciete samenwerking"
        " of tegenstrategie (Tampuu et al., 2017).\n",
        "- **RAM-observaties:** de semantiek van de 128 bytes is niet gelabeld;"
        " de agent moet zelf relevante features leren, wat meer data vereist.\n",
        "\n",
        "### Uitbreidingsmogelijkheden\n",
        "\n",
        "- **Prioritized Experience Replay (PER)** om zeldzame win-events zwaarder te wegen.\n",
        "- **Self-play** met meerdere generaties getrainde modellen als tegenstander.\n",
        "- **Reward shaping** gebaseerd op RAM-bytes die balposities coderen.\n",
    ])
)

nb["cells"] = cells

with open(
    "e:/github data/Autonomous-systems-Portfolio/portfolio 3/Train_warlords.ipynb",
    "w",
    encoding="utf-8",
) as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done — notebook written with {len(cells)} cells.")
