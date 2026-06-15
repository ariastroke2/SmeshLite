"""
PPOBot — Stable-Baselines3 PPO checkpoint trained against SmeshLiteEnv, wrapped as a
CharacterBrain for use in Match-based evaluation and round-robin notebooks (vs
RandomBot, ChaserBot, SmeshBot, QTableBot, the DeepQBot family).

Train via ppo_vs_all.ipynb (smeshlite.env.SmeshLiteEnv + agents.ppo_env's round-robin
opponent wrapper), which saves to agents/checkpoints/ppo_bot.zip by default. Then:

    match.set_player_brain(0, PPOBot())

Thin subclass of agents.sb3_template.SB3BrainTemplate with a PPO-specific name and
default checkpoint path — see that module for the underlying contract (Box(22) obs /
brain_context_to_obs(), MultiBinary(4) actions).
"""
from __future__ import annotations

from agents.sb3_template import SB3BrainTemplate


class PPOBot(SB3BrainTemplate):
    """PPO agent trained on the raw 22-dim observation (no feature engineering),
    round-robin vs RandomBot/ChaserBot/SmeshBot/frozen QTableBot."""

    BRAIN_NAME = "PPO Bot"

    def __init__(self, checkpoint_path: str = "agents/checkpoints/ppo_bot.zip", deterministic: bool = True) -> None:
        super().__init__(checkpoint_path=checkpoint_path, deterministic=deterministic)
