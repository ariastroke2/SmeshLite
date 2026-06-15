"""
DoubleDeepQBot — agents.deepq_bot_v2's dueling DQN + 47-feature encoding, plus a
Double DQN target computation.

Builds directly on agents/deepq_bot_v2.py (see that module's docstring for the full
state-encoding, dueling-head, and reward-shaping rationale — all unchanged here). The
one change is in `_learn()`'s TD target.

Double DQN target: standard DQN bootstraps off `r + gamma * max_a' Q_target(s', a')`.
Because `max` over several noisy estimates is itself a positively-biased estimator of
the true max — whichever action's Q happens to be overestimated this update gets
picked, and that overestimate then gets propagated as a target for other (s,a) pairs —
Q-values drift upward over training. The network becomes overconfident about how good
its best action actually is, independent of whether that action is in fact best.

Van Hasselt et al. (2015, "Deep Reinforcement Learning with Double Q-learning") decouple
action *selection* from action *evaluation*: pick `next_action = argmax_a' Q_policy(s',
a')` using the policy network (the one being trained this step), but evaluate its value
via `Q_target(s', next_action)` using the (lagged) target network. Because policy_net
and target_net rarely agree on which action is overestimated at the same time, an action
that policy_net over-values is unlikely to also be over-valued by target_net by the same
margin — so the systematic upward bias mostly cancels rather than compounding.

Everything else — dueling head, 47-dim feature encoding, action repeat, replay buffer,
target network sync, reward shaping — is identical to agents/deepq_bot_v2.py.

Requires the optional 'train' extra (`pip install -e ".[train]"`) for torch. Like
SB3BrainTemplate, the import is deferred so this module is importable (and discoverable
by the brain registry) even without torch installed — only constructing a
DoubleDeepQBot requires it.
"""
from __future__ import annotations

import math
import os
import random
from collections import deque

import numpy as np

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState
from smeshlite.core.character import Action, RESPAWN_INVINCIBILITY

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _QNet(nn.Module):
        """Dueling head: a shared trunk feeds separate V(s) and A(s,a) outputs,
        recombined as Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)) — see
        agents/deepq_bot_v2.py's module docstring."""

        def __init__(self, obs_dim: int, n_actions: int, hidden1: int = 32, hidden2: int = 8) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden1), nn.ReLU(),
                nn.Linear(hidden1, hidden2), nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden2, 1)
            self.advantage_head = nn.Linear(hidden2, n_actions)

        def forward(self, x):
            features = self.trunk(x)
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))

except ImportError:
    torch = None
    nn = None
    F = None
    _QNet = None


# ---------------------------------------------------------------------------
# State encoding — identical to agents/deepq_bot_v2.py
# ---------------------------------------------------------------------------

OBS_DIM = 47

_N_ACTION_TYPES = len(Action)

_THREAT_ACTIONS = (Action.ATTACK.value, Action.SMASH.value, Action.CHARGING.value)

_FEATURE_SCALE = np.array(
    # self (21): height, vx, vy, facing, damage_pct, stocks, one-hot action,
    # action_frame, in_air, charge_amount, invincibility, self_vulnerable
    [500., 20., 20., 1., 100., 3.]
    + [1.] * _N_ACTION_TYPES
    + [60., 1., 1., RESPAWN_INVINCIBILITY, 1.]
    # opponent (17): damage_pct, stocks, one-hot action, action_frame, in_air,
    # charge_amount, facing, opponent_threat
    + [100., 3.]
    + [1.] * _N_ACTION_TYPES
    + [60., 1., 1., 1., 1.]
    # relative (9): dir_x, dir_y, dist, dx_facing, opponent_facing_toward_me,
    # closing_speed, damage_diff, edge_left, edge_right
    + [1., 1., 1000., 1., 1., 20., 100., 500., 500.],
    dtype=np.float32,
)


def _one_hot(value: int, n: int) -> list[float]:
    vec = [0.0] * n
    if 0 <= value < n:
        vec[value] = 1.0
    return vec


def _encode_state(context: BrainContext) -> np.ndarray:
    target = context.opponents[0]

    dx = target.x - context.x
    dy = target.y - context.y
    # Guard against division by zero when the two characters perfectly overlap.
    dist = math.hypot(dx, dy)
    dist_safe = max(dist, 1e-6)
    dir_x = dx / dist_safe
    dir_y = dy / dist_safe

    dvx = target.vx - context.vx
    dvy = target.vy - context.vy

    self_block = [
        context.y - context.stage_y_floor,
        context.vx,
        context.vy,
        context.facing,
        context.damage_pct,
        context.stocks,
        *_one_hot(context.action, _N_ACTION_TYPES),
        context.action_frame,
        context.in_air,
        context.charge_amount,
        context.invincibility,
        context.action in _THREAT_ACTIONS,  # self_vulnerable
    ]

    opponent_block = [
        target.damage_pct,
        target.stocks,
        *_one_hot(target.action, _N_ACTION_TYPES),
        target.action_frame,
        target.in_air,
        target.charge_amount,
        target.facing,
        target.action in _THREAT_ACTIONS,  # opponent_threat
    ]

    relative_block = [
        dir_x,
        dir_y,
        dist,
        dir_x * context.facing,            # dx_facing
        -dir_x * target.facing,             # opponent_facing_toward_me
        -(dir_x * dvx + dir_y * dvy),        # closing_speed
        context.damage_pct - target.damage_pct,  # damage_diff
        context.x - context.stage_platform_x1,   # edge_left
        context.stage_platform_x2 - context.x,   # edge_right
    ]

    vec = np.array(self_block + opponent_block + relative_block, dtype=np.float32)
    return vec / _FEATURE_SCALE


# ---------------------------------------------------------------------------
# Action space — same mapping as agents.qtable_bot / agents.deepq_bot_v2
# ---------------------------------------------------------------------------

# (left, right, up, attack)
_ACTIONS: tuple[tuple[bool, bool, bool, bool], ...] = (
    (False, False, False, False),  # 0 NONE
    (True,  False, False, False),  # 1 LEFT
    (False, True,  False, False),  # 2 RIGHT
    (False, False, True,  False),  # 3 UP
    (False, False, False, True),   # 4 ATTACK
    (True,  False, False, True),   # 5 LEFT+ATTACK
    (False, True,  False, True),   # 6 RIGHT+ATTACK
    (False, False, True,  True),   # 7 UP+ATTACK
)


# ---------------------------------------------------------------------------
# DoubleDeepQBot
# ---------------------------------------------------------------------------

class DoubleDeepQBot(CharacterBrain):
    """Double DQN brain (PyTorch MLP, dueling head). Reuse one instance across
    matches to keep learning."""

    BRAIN_NAME = "Double Deep Q Bot"

    N_ACTIONS = len(_ACTIONS)
    OBS_DIM = OBS_DIM

    def __init__(
        self,
        alpha: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        dmg_scale: float = 0.01,
        ko_reward: float = 1.0,
        death_penalty: float = 1.0,
        time_penalty: float = 0.0,
        hidden1: int = 32,
        hidden2: int = 8,
        action_repeat: int = 6,
        train_every: int = 4,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_every: int = 500,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "DoubleDeepQBot requires torch. Install with: pip install -e \".[train]\""
            )

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.dmg_scale = dmg_scale
        self.ko_reward = ko_reward
        self.death_penalty = death_penalty
        self.time_penalty = time_penalty
        self.action_repeat = action_repeat
        self.train_every = train_every
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = _QNet(self.OBS_DIM, self.N_ACTIONS, hidden1, hidden2).to(self.device)
        self.target_net = _QNet(self.OBS_DIM, self.N_ACTIONS, hidden1, hidden2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)

        self.replay_buffer: deque = deque(maxlen=buffer_size)
        self.episode_reward = 0.0
        self.last_loss: float | None = None
        self.step_count = 0
        self._learn_steps = 0

        # (decision_state, action, accumulated_reward) for the action currently being
        # held, awaiting either action_repeat ticks of accumulated reward (pushed as a
        # normal transition) or new_episode() (pushed as a done=True transition).
        self._pending: tuple[np.ndarray, int, float] | None = None
        # (self_dmg, self_stocks, opp_dmg, opp_stocks) from the previous tick, used to
        # compute this tick's reward regardless of where we are in the repeat cycle.
        self._last_snapshot: tuple[float, int, float, int] | None = None
        # Ticks remaining before the current action is replaced with a fresh decision.
        self._repeat_counter = 0

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    # ------------------------------------------------------------------

    @property
    def buffer_size_used(self) -> int:
        return len(self.replay_buffer)

    def new_episode(self, final_snapshot: tuple[float, int, float, int] | None = None) -> float:
        """
        Call between matches. If a transition is still pending and `final_snapshot`
        (the just-finished match's final (self_dmg, self_stocks, opp_dmg, opp_stocks))
        is given, finalize it as a `done=True` transition first (crediting this final
        tick's reward on top of whatever was already accumulated for the held action).

        Returns the episode's total reward (including the finalized transition, if
        any), then resets `episode_reward` to 0 and clears the pending transition.
        """
        if self._pending is not None and final_snapshot is not None:
            decision_state, action, accumulated_reward = self._pending
            reward = self._reward(self._last_snapshot, final_snapshot)
            self.episode_reward += reward
            accumulated_reward += reward
            self.replay_buffer.append((decision_state, action, accumulated_reward, decision_state, True))
            self.step_count += 1
            if self.step_count % self.train_every == 0:
                self._learn()

        total = self.episode_reward
        self._pending = None
        self._last_snapshot = None
        self._repeat_counter = 0
        self.episode_reward = 0.0
        return total

    # ------------------------------------------------------------------

    def think(self, context: BrainContext, out: InputState) -> None:
        out.clear()
        if not context.opponents:
            return
        target = context.opponents[0]

        state = _encode_state(context)
        snapshot = (context.damage_pct, context.stocks, target.damage_pct, target.stocks)

        if self._pending is not None:
            decision_state, action, accumulated_reward = self._pending
            reward = self._reward(self._last_snapshot, snapshot)
            self.episode_reward += reward
            accumulated_reward += reward
            self._repeat_counter -= 1

            if self._repeat_counter <= 0:
                self.replay_buffer.append((decision_state, action, accumulated_reward, state, False))
                self.step_count += 1
                if self.step_count % self.train_every == 0:
                    self._learn()

                action = self._select_action(state)
                self._pending = (state, action, 0.0)
                self._repeat_counter = self.action_repeat
            else:
                self._pending = (decision_state, action, accumulated_reward)
        else:
            action = self._select_action(state)
            self._pending = (state, action, 0.0)
            self._repeat_counter = self.action_repeat

        self._last_snapshot = snapshot
        self._apply_action(action, out)

    # ------------------------------------------------------------------

    def _reward(
        self,
        prev_snapshot: tuple[float, int, float, int],
        snapshot: tuple[float, int, float, int],
    ) -> float:
        _prev_self_dmg, prev_self_stocks, prev_opp_dmg, prev_opp_stocks = prev_snapshot
        _self_dmg, self_stocks, opp_dmg, opp_stocks = snapshot

        reward = -self.time_penalty
        dmg_dealt = max(0.0, opp_dmg - prev_opp_dmg)
        reward += dmg_dealt * self.dmg_scale
        if opp_stocks < prev_opp_stocks:
            reward += self.ko_reward
        if self_stocks < prev_self_stocks:
            reward -= self.death_penalty
        return reward

    def _select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        with torch.no_grad():
            q = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))
        return int(torch.argmax(q, dim=1).item())

    def _learn(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.stack(states)).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN: select the next action with policy_net, evaluate it with
            # target_net — see module docstring.
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.item())

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    @staticmethod
    def _apply_action(action: int, out: InputState) -> None:
        out.left, out.right, out.up, out.attack = _ACTIONS[action]

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state_dict"])
        self.target_net.load_state_dict(data["policy_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.step_count = data.get("step_count", 0)
