"""
DeepQBot — a from-scratch DQN brain (PyTorch MLP over continuous features).

Follow-up to agents.qtable_bot.QTableBot: where QTableBot discretizes the world into
504 bins and learns a lookup table, DeepQBot keeps the same action space and reward
shaping but replaces the table with a small neural network over continuous features,
trained with standard DQN (replay buffer + target network).

State (47 continuous floats, normalized): built directly from named `context` /
`target` (`context.opponents[0]`) attributes rather than `brain_context_to_obs()`'s
flat vector — that function exists to give the SB3/gym environment index-parity
with its `Box(22,)` observation space, a constraint DeepQBot doesn't share. Three
concatenated blocks:

- Self (21): height above the stage floor (`y - stage_y_floor`, replacing raw `y`
  — like raw `x`, a raw `y` coordinate is meaningless without knowing where the
  floor is); raw `vx`/`vy` (kept un-decomposed — a character's own velocity is
  directly "am I in danger" information, e.g. flying off-stage during hitstun,
  independent of where the opponent is); `facing`; `damage_pct`; `stocks`; a
  10-dim one-hot of `action` (NONE/RUN/JUMP/AERIAL/ATTACK/SMASH/CHARGING/HITSTUN/
  RESPAWN/DEAD — see "Categorical actions" below); `action_frame`; `in_air`;
  `charge_amount`; `invincibility` (frames remaining — previously read by no
  brain); and `self_vulnerable` (1 if `action` is ATTACK/SMASH/CHARGING — mirrors
  `opponent_threat` below, but for self: "am I currently committed to something I
  can't cancel out of").

- Opponent (17): the self block minus position/velocity — `damage_pct`, `stocks`,
  a 10-dim one-hot of `action`, `action_frame`, `in_air`, `charge_amount`,
  `facing`, and `opponent_threat` (1 if the opponent is mid ATTACK/SMASH/CHARGING
  — "is something dangerous about to happen").

- Relative (9): `dir_x`, `dir_y` (unit vector from self to opponent — direction,
  independent of distance); `dist` (the magnitude that direction omits);
  `dx_facing` (`dir_x * self.facing`, in [-1, 1] — is the opponent in front of or
  behind me, now direction-normalized rather than scaled by raw distance);
  `opponent_facing_toward_me` (`-dir_x * opponent.facing`, in [-1, 1] — is the
  opponent's front toward me, i.e. an exploitable approach, or their back to me);
  `closing_speed` (`-(dir_x * dvx + dir_y * dvy)`, the rate the gap between us is
  shrinking (positive) or growing (negative) — relative velocity projected onto
  the direction between us, rather than raw `dvx`/`dvy` components); `damage_diff`
  (`self.damage_pct - opponent.damage_pct` — who's relatively closer to a KO); and
  the 2 edge-distance features (`x - stage_platform_x1`, `stage_platform_x2 - x`,
  the geometry exposed for the SmeshBot border fix).

Categorical actions: `action` is a 10-value enum (NONE..DEAD), not a continuous
quantity — encoding it as `action / 9` (the v1 scheme) imposes a false ordinal
structure (e.g. "DEAD(9) is ~2x ATTACK(4)") the network has no reason to expect and
has to actively unlearn. One-hot removes that structure at the cost of 9 extra dims
per character (18 total). The old `self_hitstun` flag — a 1:1 duplicate of one of
these 10 dims — and the old raw-magnitude `dx`/`dy`/`dvx`/`dvy` features are dropped
now that their information is represented more directly above.

Each feature is divided by a fixed per-feature scale (`_FEATURE_SCALE`) before being
fed to the network. With `OBS_DIM` growing from 31 to 47, `hidden1` is widened from
16 to 32 to give the first layer enough capacity for the larger input.

Action space (8): identical mapping to QTableBot's — NONE, LEFT, RIGHT, UP, ATTACK,
LEFT+ATTACK, RIGHT+ATTACK, UP+ATTACK — defined independently here to keep the two
agent modules decoupled.

Reward: same shaping as QTableBot (dmg_scale * damage dealt, +ko_reward, -death_penalty,
-time_penalty per tick), computed from a (self_dmg, self_stocks, opp_dmg, opp_stocks)
snapshot diff each tick.

Terminal-reward fix: `think()` is never called again after the KO-causing tick (Match
stops ticking once `done` is set), so the final ±1.0 KO/death reward would normally
never be observed. `new_episode(final_snapshot=...)` lets the training loop hand in the
just-finished match's final (damage, stocks) snapshot so that reward gets credited and
pushed to the replay buffer as a `done=True` transition before the pending state is
cleared. It returns the episode's total reward (including this final one).

Action repeat: at 30 ticks/sec with mostly-zero per-tick rewards, a single tick's action
has almost no visible effect on the outcome a TD update can bootstrap from (gamma=0.95
gives an effective horizon of ~20 ticks, i.e. ~0.67s). `action_repeat` (default 6) holds
each chosen action for that many ticks, accumulating the per-tick rewards into a single
transition `(decision_state, action, summed_reward, next_decision_state, done)`. This
both reduces the number of "decisions" the network has to assign credit across and
widens the effective real-time horizon (gamma's ~20-decision lookback now spans ~120
ticks / 4s instead of ~0.67s).

Dueling head: a first training run of this exact network (frozen as `agents/deepq_bot.py`,
the "v1" failure) had its Q-values collapse to a near-constant value across essentially
every state/action — the network learned V(s) (a state's baseline value, dominated by
the average reward rate under a sparse reward signal) but never differentiated A(s,a)
(how much better one action is than another in that state). Because V(s) explains almost
all of the variance in the TD target while A(s,a) is a small residual driven by rare
reward events, a small network with a single shared head finds "predict V(s) for
everything" both trivial to represent and close to optimal-on-average — which crowds out
the much smaller A(s,a) signal. `_QNet` now splits its final layer into a `value_head`
(1 output, V(s)) and an `advantage_head` (N_ACTIONS outputs, A(s,a)), recombined as
`Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))`. This gives V(s) its own dedicated output, so
fitting it well doesn't require "spending" the same shared trunk capacity that A(s,a)
needs to represent per-action differences (Wang et al., 2016, "Dueling Network
Architectures for Deep Reinforcement Learning"). Nothing else about the training loop
changes — same replay buffer, target network, reward, and feature encoding as v1.

Requires the optional 'train' extra (`pip install -e ".[train]"`) for torch. Like
SB3BrainTemplate, the import is deferred so this module is importable (and discoverable
by the brain registry) even without torch installed — only constructing a DeepQBot
requires it.
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
        recombined as Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)) — see module docstring."""

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
            # Subtract the mean (not max) advantage: this is what makes V(s) and A(s,a)
            # identifiable — without it, an arbitrary constant could shift freely between
            # the two heads and V(s) would no longer mean "the state's baseline value".
            return value + (advantage - advantage.mean(dim=1, keepdim=True))

except ImportError:
    torch = None
    nn = None
    F = None
    _QNet = None


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

OBS_DIM = 47

# Action is a 10-value enum (NONE, RUN, JUMP, AERIAL, ATTACK, SMASH, CHARGING,
# HITSTUN, RESPAWN, DEAD) — one-hot'd per character, see module docstring.
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
# Action space — same mapping as agents.qtable_bot
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
# DeepQBot
# ---------------------------------------------------------------------------

class DeepQBot(CharacterBrain):
    """DQN brain (PyTorch MLP). Reuse one instance across matches to keep learning."""

    BRAIN_NAME = "Deep Q Bot"

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
                "DeepQBot requires torch. Install with: pip install -e \".[train]\""
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
            next_q = self.target_net(next_states_t).max(dim=1).values
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
