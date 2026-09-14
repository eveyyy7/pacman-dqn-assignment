# Training a DQN Agent to Play Ms. Pac-Man

Assignment submission — a Deep Q-Network trained on `ALE/MsPacman-v5` using the
classroom notebook, run end-to-end on a Google Colab T4 GPU.

**Headline result: 150 episodes of DQN training produced no measurable improvement
over the untrained baseline (492.0 → 500.0 mean score, a +8 point change that is
well inside the noise of a 5-game evaluation).** Two additional runs with different
hyperparameters did not improve on the baseline either. This README reports what
actually happened, including the runs that did not work.

---

## 1. My three choices

| Setting | Value | Why |
|---|---|---|
| **Exploration** (ε, constant after warm-up) | **0.20** | The notebook default. 20% random / 80% greedy keeps enough novelty flowing into a replay buffer that only holds 5,000 transitions, while still letting the learned policy drive most decisions. Keeping the default also makes this run directly comparable to classmates' runs. |
| **Episodes** | **150** | 50% more games than the default 100, chosen to give the network a meaningfully larger number of gradient updates (21,906 vs. 15,417) while keeping a single Colab session under ~6 minutes of training. |
| **Learning rate** | **0.0001** | The notebook default. With Adam + Huber loss this is the stable choice; a larger step size on a 5,000-transition replay buffer risks overfitting the most recent experience and diverging Q-values. |

**Prediction before running:** I expected a small positive change — maybe 10–20% — on
the grounds that 150 games is more than the default. I expected the training-score
curve to drift upward and the evaluation mean to follow it.

**What actually happened:** the training curve did drift upward, and the evaluation
mean did *not* follow. See §4.

---

## 2. State, actions, and reward

**State.** The agent never sees RAM or object positions. Each raw 210×160 RGB Atari
frame is converted to greyscale and resized to **84×84**, and the **four most recent
frames are stacked** into an `(4, 84, 84)` `uint8` tensor. Four frames rather than one
is what makes the state approximately Markov: a single image shows where Pac-Man and
the ghosts *are*, but not which way they are *moving*. Pixel values are scaled to
[0, 1] inside the network's forward pass. On reset the stack is filled with four
copies of the opening frame.

One agent **decision** covers **4 emulator frames** (frame skip), so a 3,000-decision
cap is roughly 200 seconds of game time.

**Actions.** The discrete Ms. Pac-Man action set, 9 joystick actions:
`NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT`.
The network's output layer emits one Q-value per action — an estimate of discounted
future return, *not* a probability. Action selection is ε-greedy: with probability ε
a uniformly random action, otherwise `argmax_a Q(s, a)`.

**Reward.** The game's own score increments: 10 per pellet, 50 per power pellet,
200/400/800/1600 for successive ghosts eaten while powered, plus fruit bonuses.
Two things are worth separating:

* **For learning**, rewards are **clipped to [−1, +1]**. This keeps the gradient scale
  uniform across Atari games, but it also means the agent cannot tell a 10-point
  pellet from a 200-point ghost — every positive event looks identical. That is a
  deliberate simplification of the classroom notebook, and it matters for §6.
* **For reporting**, every score in this README is the **raw, unclipped game score**.

The learning target is `r + γ · max_a' Q_target(s', a')` with **γ = 0.99**, computed
from a slowly-updated target network (synced every 1,000 decisions). The future-value
term is zeroed on true game over, but *kept* when an episode ends by hitting the
3,000-decision time limit — the game could have continued, so bootstrapping is still
correct there.

**Other fixed settings** (identical across all runs): replay capacity 5,000, batch
size 32, 1,000 random warm-up decisions before any weight update, one update every 4
decisions, sticky-action probability 0.25, up to 30 no-ops on reset, losing a life
does *not* end the episode, seed 42.

---

## 3. Evaluation protocol

Before/after use **exactly the same settings**: the same five seeds
`[101, 202, 303, 404, 505]`, ε = 0.05, the same 3,000-decision cap, a separate
environment, and no weight updates or replay writes. The "before" baseline is the
**untrained network** (randomly initialised weights at ε = 0.05), not a uniformly
random agent — which is why the baseline already scores ~492 rather than ~200.

Because the untrained network is initialised from seed 42 in every run, all three
runs below share an identical baseline of **492.0**. That is convenient: the three
"after" numbers are directly comparable to each other.

---

## 4. Main run — what happened

`results/main_run_exp0.20_ep150_lr0.0001/`

Exploration 0.20 · 150 episodes · learning rate 0.0001 · seed 42 · CUDA (Colab T4)
150/150 episodes completed · 88,622 decisions · 21,906 learning updates · 331 s

### Five evaluation games, identical settings

| Game (seed) | Before (untrained) | After (150 episodes) |
|---|---:|---:|
| 1 (101) | 350 | 340 |
| 2 (202) | 500 | 340 |
| 3 (303) | 320 | 290 |
| 4 (404) | 800 | **1230** |
| 5 (505) | 490 | 300 |
| **Mean** | **492.0** | **500.0** |
| Std. dev. | 170.1 | 365.6 |
| Mean episode length (decisions) | 589.0 | 477.8 |
| Time-limited games | 0 / 5 | 0 / 5 |

**Change in mean score: +8.0.** With five games per condition the standard error of
that difference is roughly ±180 points, so +8 is about **0.04 standard errors** —
indistinguishable from zero. The trained agent won exactly one of the five matched
games (game 4, 800 → 1230); that single game is the entire mean difference, and it is
also the reason the "after" standard deviation more than doubled.

**One good-looking GIF is not evidence.** `gifs/final_best.gif` is the best of these
five games by full-game score, which is precisely the game that got lucky.

### Training curve

![Training dashboard](results/main_run_exp0.20_ep150_lr0.0001/training_dashboard.png)

| | First 25 episodes | Last 25 episodes | All 150 |
|---|---:|---:|---:|
| Mean training score | 579.2 | 822.8 | 701.7 (sd 472.0, max 3130) |

Training score rose by about 42% from the first 25 games to the last 25. That looks
encouraging until you note the per-episode standard deviation of 472: with 25 games
per bucket, the difference of +244 is only ~1.8 standard errors, and it did **not**
reproduce in the 300-episode run (§5). Training episodes also draw a fresh seed each
game (`SEED + episode`), while evaluation uses five fixed seeds — so the two numbers
are measuring slightly different things.

### Periodic checkpoints (single evaluation game, seed 101)

| After episode | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Score | 350 | 320 | 220 | 410 | 290 | 460 | 340 |

No trend — it wanders between 220 and 460 the whole way. GIFs for each of these
checkpoints are in `gifs/`.

### Loss went *up*, not down

Mean Huber loss per episode climbed steadily from **0.026** to **0.099** over the run.
This is normal and expected in early DQN training — as the target network propagates
reward information backwards, the magnitude of the Q-values themselves grows, so the
absolute prediction error grows with them. It is a useful illustration of the point
the assignment makes: **loss and play quality are measuring different things.** Here
loss moved a lot and the score did not move at all.

---

## 5. Comparison across three runs

All three runs share the identical untrained baseline of **492.0** (seed 42), so the
"after" column is the comparison.

| Run | ε | Episodes | LR | Updates | After mean | Δ vs. baseline | Δ / SE | Folder |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Default-length | 0.20 | 100 | 1e-4 | 15,417 | 414.0 | **−78.0** | −0.71 | `results/extra_run_exp0.20_ep100_lr0.0001/` |
| **Main** | 0.20 | **150** | 1e-4 | 21,906 | **500.0** | **+8.0** | +0.04 | `results/main_run_exp0.20_ep150_lr0.0001/` |
| Long / low-LR | 0.10 | 300 | 5e-5 | 45,551 | 478.0 | −14.0 | −0.15 | `results/extra_run_exp0.10_ep300_lr0.00005/` |

**Reporting the failures honestly:** the 100-episode run came out 78 points *worse*
than doing no training at all. The 300-episode run — twice the training of the main
run, lower exploration, lower learning rate — landed 14 points below baseline. None of
the three differences reaches even one standard error. Doubling the number of episodes
did not help; more episodes do not guarantee better play.

The most informative comparison is the 300-episode run, because it is the one with
enough updates to expect *something*. Its training curve also flattened (first-25 mean
578.4 → last-25 mean 634.0, a much weaker drift than the main run's), while its loss
climbed the highest of the three (0.025 → 0.109, peaking at 0.162). That combination —
most training, most loss growth, no score gain — is the clearest single piece of
evidence that these runs are all far short of the scale where Atari DQN starts to work.

---

## 6. One observed limitation

**The agent got faster at collecting pellets but did not learn to avoid ghosts, so
its episodes ended sooner and its total score stayed flat.**

The evidence is in the evaluation episode lengths, which are reported alongside the
scores:

| | Mean score | Mean decisions survived | Points per 100 decisions |
|---|---:|---:|---:|
| Untrained | 492.0 | 589.0 | 83.5 |
| After 150 episodes | 500.0 | 477.8 | **104.7** |

Scoring *rate* improved about 25%, survival time dropped about 19%, and the two
cancelled out. Watching `gifs/episode_0000.gif` next to `gifs/episode_0150.gif`, the
untrained agent dithers in place and often gets cornered; the trained agent commits to
a direction and clears a corridor of pellets, then walks into a ghost.

There is a clean mechanical reason to expect exactly this, and it is the reward
clipping described in §2. Clipping every reward to +1 tells the agent that a pellet
and a ghost-eat are worth the same, and — more importantly — with γ = 0.99 and no
life-loss termination, the cost of dying is diffuse and delayed, while the reward for
the next pellet is immediate and certain. The 5,000-transition replay buffer holds only
about six games' worth of experience, so rare, expensive events (dying in a specific
ghost configuration) are forgotten long before they can be learned from. The agent
optimises the part of the problem it can actually see.

**Caveat, stated plainly:** this is five evaluation games. The same rate-vs-survival
pattern does *not* appear in the other two runs (100 episodes: 65.8 points per 100
decisions, worse on both axes; 300 episodes: 84.7, unchanged). So this is an
observation about the main run that suggests a hypothesis, not a demonstrated effect.

---

## 7. What I would change next

Change **one** variable and hold the other two fixed:

1. **Replay capacity, 5,000 → 100,000.** Not one of the three student-facing knobs, but
   it is the binding constraint. The original DQN paper uses 1,000,000 transitions; at
   5,000 the buffer is overwritten roughly every six games, so the network is trained
   almost entirely on a sliding window of very recent, highly correlated experience.
   No setting of ε, episode count, or learning rate can compensate for that.
2. **Among the three choices I can actually set: episodes, 150 → 2,000+**, keeping
   ε = 0.20 and LR = 0.0001. On a T4 this run took 331 seconds; 2,000 episodes is
   roughly 75 minutes, which is well within a Colab session. Published DQN results on
   Ms. Pac-Man use tens of millions of frames — 88,622 decisions (~354,000 frames) is
   about 0.4% of a standard 50M-frame Atari run. The honest reading of this experiment
   is not "DQN does not work on Ms. Pac-Man"; it is "150 episodes is nowhere near
   enough to tell."
3. **Decaying ε** (1.0 → 0.05 over training) instead of a constant 0.20. A fixed 20%
   random action rate permanently caps performance even after the policy is good,
   because one move in five is thrown away.

---

## 8. Repository contents

```
pacman_dqn.ipynb      the notebook, with my three choices set in section 1
requirements.txt      package pins
pacman_player.py      optional local floating-GIF player (unused on Colab)
results/
  main_run_exp0.20_ep150_lr0.0001/      <- the submitted experiment
    config.json               settings, hardware, exact package versions
    baseline.json             untrained scores on the 5 evaluation seeds
    comparison.json           before/after, both conditions, all 5 games
    demo_scores.json          the 25-episode checkpoint evaluations
    training.csv              one row per episode: score, steps, loss, elapsed
    training_summary.json     episodes, decisions, updates, wall-clock
    training_dashboard.png    score / loss / exploration panels
    gifs/
      episode_0000.gif        untrained baseline gameplay
      episode_0025.gif .. episode_0150.gif   checkpoints every 25 episodes
      final_best.gif          best of the five final evaluation games
  extra_run_exp0.20_ep100_lr0.0001/      same structure, 100 episodes
  extra_run_exp0.10_ep300_lr0.00005/     same structure, 300 episodes
```

GIFs play at 4× speed and show the first 20 seconds of game time; the reported scores
cover the entire game.

Model checkpoints (`*.pt`, ~6.8 MB each) are **not** committed — they are kept locally,
as the notebook recommends, to keep this repository small. `config.json` plus the seed
is enough to reproduce a run.

## 9. Reproducing

Open `pacman_dqn.ipynb` in Google Colab (Runtime → Change runtime type → **T4 GPU**),
confirm the three values in section 1, and choose Runtime → Run all. The setup cell
installs everything. The main run took **331 seconds** of training plus evaluation
time on a T4.

Exact environment used: Python 3.13.15, torch 2.11.0+cu128, gymnasium 1.3.0,
ale-py 0.11.2, opencv-python-headless 4.14.0.94, numpy 2.1.3, matplotlib 3.10.0,
Pillow 11.3.0 (recorded in each run's `config.json`).

## 10. Credits

Notebook and DQN implementation: [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).
Method: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
Environment: [Arcade Learning Environment](https://ale.farama.org/) via
[Gymnasium](https://gymnasium.farama.org/).
