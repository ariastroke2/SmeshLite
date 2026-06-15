"""
RoundRobinOpponentEnv — wraps smeshlite.env.SmeshLiteEnv so player 1 cycles through
the same 4-opponent pool used by the DeepQBot family's round-robin notebooks
(RandomBot, ChaserBot, SmeshBot, frozen QTableBot), instead of SmeshLiteEnv's default
no-op opponent.

SmeshLiteEnv.reset() rewires a fresh ExternalBrain onto every character (including
player 1), and step() feeds player 1's ExternalBrain a no-op InputState — see that
module's docstring. This wrapper re-applies a real CharacterBrain to
match.characters[1] after every reset(), so player 1's brain.think() (called from
Match.tick() via Character.gather_input()) drives it instead.

Used by ppo_vs_all.ipynb to train a PPO agent (player 0, raw 22-dim observation) against
the same opponent pool the DeepQBot/DoubleDeepQBot notebooks train against.
"""
from __future__ import annotations

import gymnasium as gym

from agents.chaser_bot import ChaserBot
from agents.qtable_bot import QTableBot
from agents.random_bot import RandomBot
from agents.smesh_bot import SmeshBot
from smeshlite.env import SmeshLiteEnv

OPPONENT_NAMES = ["RandomBot", "ChaserBot", "SmeshBot", "QTableBot"]

# Frozen Q-table baseline (epsilon=alpha=0 -> no further learning during PPO's
# training), matching the DeepQBot notebooks' `qtable_opponent`. Re-instantiated each
# time the cycle reaches it (cheap: a ~500-state JSON table) rather than shared, so
# concurrent vectorized envs never mutate the same instance's `_pending` state.
OPPONENT_CYCLE = (
    RandomBot,
    ChaserBot,
    SmeshBot,
    lambda: QTableBot(q_table_path="agents/checkpoints/qtable_bot.json", epsilon=0.0, alpha=0.0),
)


class RoundRobinOpponentEnv(gym.Wrapper):
    """Cycles player 1 through OPPONENT_CYCLE each episode. `info["opponent"]` (the
    current opponent's name) is set on both reset() and every step(), so a callback
    can attribute each finished episode to an opponent without extra bookkeeping."""

    def __init__(self, env: gym.Env, opponent_cycle=OPPONENT_CYCLE, opponent_names=OPPONENT_NAMES):
        super().__init__(env)
        self._opponents = list(opponent_cycle)
        self._names = list(opponent_names)
        self._idx = 0
        self._current_name: str | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._idx = seed
        i = self._idx % len(self._opponents)
        self._idx += 1
        self.env.match.set_player_brain(1, self._opponents[i]())
        self._current_name = self._names[i]
        info["opponent"] = self._current_name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["opponent"] = self._current_name
        return obs, reward, terminated, truncated, info


def make_env(time_limit: int = 1800) -> RoundRobinOpponentEnv:
    """Factory for stable_baselines3.common.env_util.make_vec_env(make_env, ...)."""
    return RoundRobinOpponentEnv(SmeshLiteEnv(time_limit=time_limit))
