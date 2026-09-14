# Training a DQN Agent to Play Ms. Pac-Man

Class 3 assignment — a Deep Q-Network trained on `ALE/MsPacman-v5` with the ready-made
classroom notebook, run end to end on a Google Colab T4 GPU.

**My three hyperparameters: exploration `0.20`, episodes `150`, learning rate `0.0001`.**

**Result: mean score over the five evaluation games rose from 492.0 (untrained) to
754.0 (after 150 episodes), a change of +262.0.** That is about 1.4 standard errors on
a five-game test — suggestive, not conclusive. And the honest complication, reported in
full in §6: I ran the *identical* configuration a second time and it scored **500.0**.
Two runs with the same three hyperparameters and the same seed differ by 254 points —
the same order of magnitude as every difference I measured *between* hyperparameter
settings. At this training scale, run-to-run variance is not a small correction on top
of the hyperparameter effect; it is the same size as the effect.

### Evidence for the submitted run — [`results/main_run_exp0.20_ep150_lr0.0001/`](results/main_run_exp0.20_ep150_lr0.0001/)

| | |
|---|---|
| **Executed notebook**, saved after the final run with all outputs | [`pacman_dqn.ipynb`](pacman_dqn.ipynb) |
| Settings, hardware, exact package versions | [`config.json`](results/main_run_exp0.20_ep150_lr0.0001/config.json) |
| All five before/after evaluation scores | [`comparison.json`](results/main_run_exp0.20_ep150_lr0.0001/comparison.json) |
| Untrained baseline scores | [`baseline.json`](results/main_run_exp0.20_ep150_lr0.0001/baseline.json) |
| Per-episode training log, 150 rows | [`training.csv`](results/main_run_exp0.20_ep150_lr0.0001/training.csv) |
| Episodes, decisions, learning updates, wall clock | [`training_summary.json`](results/main_run_exp0.20_ep150_lr0.0001/training_summary.json) |
| Checkpoint evaluations every 25 episodes | [`demo_scores.json`](results/main_run_exp0.20_ep150_lr0.0001/demo_scores.json) |
| Training plot | [`training_dashboard.png`](results/main_run_exp0.20_ep150_lr0.0001/training_dashboard.png) |
| Gameplay GIFs | [`gifs/`](results/main_run_exp0.20_ep150_lr0.0001/gifs/) |

---

## 1. Open and run the notebook

Open [`pacman_dqn.ipynb`](pacman_dqn.ipynb) in Google Colab
([the original starter notebook](https://colab.research.google.com/github/pepealonso95/pacman-dqn/blob/main/pacman_dqn.ipynb)),
select **Runtime → Change runtime type → T4 GPU**, confirm the three values in section 1,
and choose **Runtime → Run all**. The setup cell installs every package automatically.
To run it locally instead, open the notebook with a Python 3.11–3.13 kernel after
`pip install -r requirements.txt`.

The submitted run took **328.9 seconds** of training plus evaluation time on a T4.
Exact environment, recorded in `config.json`: Python 3.13.15, torch 2.11.0+cu128,
gymnasium 1.3.0, ale-py 0.11.2, opencv-python-headless 4.14.0.94, numpy 2.1.3,
matplotlib 3.10.0, Pillow 11.3.0.

---

## 2. My three hyperparameters, and why

| Setting | Value | Why I chose it |
|---|---|---|
| **Exploration** (ε, held constant after the 1,000-decision warm-up) | **0.20** | The notebook default. 20% random / 80% greedy keeps fresh experience flowing into a replay buffer that only holds 5,000 transitions, while still letting the learned policy drive most decisions. Keeping the default also makes my run directly comparable to classmates' runs on the leaderboard. |
| **Episodes** | **150** | 50% more games than the default 100, chosen to buy a meaningfully larger number of gradient updates (21,370 vs. ~15,000) while keeping training inside a single short Colab session. |
| **Learning rate** | **0.0001** | The notebook default. With Adam and a Huber loss this is the stable choice; a larger step on a 5,000-transition buffer risks overfitting the most recent experience and letting the Q-values diverge. |

I edited **only** these three values in section 1. Every other setting is the notebook's
default, unmodified: replay capacity 5,000, batch size 32, 1,000 warm-up decisions, one
update every 4 decisions, target-network sync every 1,000 decisions, γ = 0.99, frame
skip 4, sticky-action probability 0.25, up to 30 no-ops on reset, no termination on life
loss, a 3,000-decision cap per game, seed 42.

**What I expected before training:** a small positive change, maybe 10–20%, on the
reasoning that 150 games is more than the default 100. I expected the training-score
curve to rise and the evaluation mean to follow it.

**What I observed:** the evaluation mean rose much more than I expected (+53%), while the
training curve barely moved (+13% from the first 25 games to the last 25) — the opposite
relationship to the one I had assumed. And when I repeated the identical configuration,
the improvement largely disappeared. Details in §5 and §6.

---

## 3. What the agent observes, does, and is rewarded for

**Observations — four game screens.** The agent never sees RAM or object coordinates.
Each raw 210×160 RGB Atari frame is converted to greyscale and resized to **84×84**, and
the **four most recent frames are stacked** into a `(4, 84, 84)` array. Four frames
rather than one is what makes the state usable: a single image shows where Pac-Man and
the ghosts *are*, but not which way they are *moving*. Pixels are rescaled to [0, 1]
inside the network. On reset the stack is filled with four copies of the opening frame.

One agent **decision** covers **4 emulator frames**, so the 3,000-decision cap is about
200 seconds of game time.

**Actions — joystick moves.** The nine discrete Ms. Pac-Man actions:
`NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT`. The network's final
layer emits one Q-value per action — an estimate of discounted future return, not a
probability. Selection is ε-greedy: with probability ε a uniformly random move, otherwise
`argmax_a Q(s, a)`.

**Reward — game points.** The game's own score increments: 10 per pellet, 50 per power
pellet, 200/400/800/1600 for successive ghosts eaten while powered, plus fruit bonuses.
Two things are worth keeping separate:

* **For learning**, rewards are **clipped to [−1, +1]**. This keeps gradients on a
  uniform scale across Atari games, but it also means the agent cannot tell a 10-point
  pellet from a 200-point ghost — every positive event looks identical to it.
* **For reporting**, every score in this README is the **raw, unclipped game score**.

The learning target is `r + γ · max_a′ Q_target(s′, a′)` with **γ = 0.99**, taken from a
slowly-updated target network synced every 1,000 decisions. The future-value term is
zeroed at true game over but **kept** when an episode ends by hitting the decision cap —
the game could have continued there, so bootstrapping is still correct.

---

## 4. How the before/after comparison was made

Before and after use **exactly the same evaluation settings**, unchanged from the
notebook: the same five seeds `[101, 202, 303, 404, 505]`, **ε = 0.05**, the same
3,000-decision cap, a separate environment, and no weight updates or replay writes.

The "before" baseline is the **untrained network** — randomly initialised weights played
at ε = 0.05 — **not** a random-action agent. That is why the baseline already scores ~492
rather than ~200.

Because the untrained network is initialised from seed 42 in every run, all five runs in
§6 share an identical baseline of **492.0**, which makes their "after" numbers directly
comparable to one another.

---

## 5. The submitted run

Exploration 0.20 · 150 episodes · learning rate 0.0001 · seed 42

| Actual training budget | |
|---|---|
| Episodes completed | **150 / 150** — ran to completion, not interrupted |
| Agent decisions | **86,478** |
| Learning updates | **21,370** — non-zero; the warm-up was passed during episode 2 |
| Elapsed time, including periodic demos | **328.9 s** |
| Hardware | **CUDA — NVIDIA T4 GPU on Google Colab** |

### All five evaluation scores, both conditions

Raw data: [`comparison.json`](results/main_run_exp0.20_ep150_lr0.0001/comparison.json)

| Game (seed) | Before — untrained | After — 150 episodes | Change |
|---|---:|---:|---:|
| 1 (101) | 350 | 330 | −20 |
| 2 (202) | 500 | 770 | +270 |
| 3 (303) | 320 | 1000 | +680 |
| 4 (404) | 800 | 360 | **−440** |
| 5 (505) | 490 | 1310 | +820 |
| **Mean** | **492.0** | **754.0** | **+262.0** |
| Standard deviation | 170.1 | 375.5 | |
| Mean decisions survived | 589.0 | 617.0 | +28 |
| Games stopped by the time limit | 0 / 5 | 0 / 5 | |

The trained agent beat the untrained network on **three of the five matched games**, and
the mean rose 53%. With five games per condition the standard error of that difference is
about ±184 points, so +262 is roughly **1.4 standard errors** — enough to be interesting,
not enough to claim the effect is real. §6 explains why I am not claiming it.

### Gameplay

GIFs play at 4× speed and show at most the first 20 seconds of game time; the scores
above cover each full game.

**Before training — untrained network** (evaluation seed 101)

![Untrained gameplay](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0000.gif)

**Best trained game** — highest-scoring of the five final evaluation games (game 5, seed
505, 1310 points)

![Best trained gameplay](results/main_run_exp0.20_ep150_lr0.0001/gifs/final_best.gif)

This is the best of five, not the typical one. The five-score table above is the evidence;
the GIF is the illustration.

**Intermediate checkpoints** — one evaluation game on seed 101 every 25 episodes
([`demo_scores.json`](results/main_run_exp0.20_ep150_lr0.0001/demo_scores.json)):

| After 25 ep — 490 | After 50 ep — 350 | After 75 ep — 540 |
|---|---|---|
| ![25](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0025.gif) | ![50](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0050.gif) | ![75](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0075.gif) |

| After 100 ep — 1260 | After 125 ep — 460 | After 150 ep — 330 |
|---|---|---|
| ![100](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0100.gif) | ![125](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0125.gif) | ![150](results/main_run_exp0.20_ep150_lr0.0001/gifs/episode_0150.gif) |

Note that this single-seed checkpoint score does **not** climb: 490 → 350 → 540 → 1260 →
460 → 330. The one spike at episode 100 is a single game, exactly the kind of result the
five-seed evaluation exists to discount.

### Training plot

![Training dashboard — score, loss, exploration](results/main_run_exp0.20_ep150_lr0.0001/training_dashboard.png)

Per-episode data: [`training.csv`](results/main_run_exp0.20_ep150_lr0.0001/training.csv)

| | First 25 episodes | Last 25 episodes | All 150 |
|---|---:|---:|---:|
| Mean training score | 573.2 | 646.4 | 631.8 (sd 348.1, max 1970) |

The training score drifted up only about 13%, against a per-episode standard deviation of
348 — essentially flat. Yet the five-seed evaluation went up 53%. The two numbers are not
measuring the same thing: training games are played at ε = 0.20 with a fresh seed each
episode, while evaluation plays five fixed seeds at ε = 0.05, so a policy can look
mediocre in training and better under evaluation. It is a good reminder not to read the
training curve as the result.

**Loss went up, not down.** Mean Huber loss per episode climbed steadily from **0.026** to
**0.093**. This is normal in early DQN training — as the target network propagates reward
information backwards, the magnitude of the Q-values grows and the absolute prediction
error grows with them. It is a clean illustration of the point the assignment makes:
*lower training loss does not guarantee better play.* Here loss rose by a factor of 3.6
while play, by the evaluation measure, improved.

---

## 6. Five runs, and why I am not claiming a result

A second training run was optional; I ended up with five, including two pairs that repeat
the *same* configuration. All share the identical untrained baseline of **492.0**
(seed 42), so the "after" column is the whole comparison.

| Run | ε | Episodes | LR | Updates | After mean | Δ vs. baseline | Δ / SE | Folder |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Submitted** | **0.20** | **150** | **1e-4** | **21,370** | **754.0** | **+262.0** | **+1.42** | [`main_run_…ep150…`](results/main_run_exp0.20_ep150_lr0.0001/) |
| Same settings, repeat | 0.20 | 150 | 1e-4 | 21,906 | 500.0 | +8.0 | +0.04 | [`…ep150…_repeat`](results/additional_runs/exp0.20_ep150_lr0.0001_repeat/) |
| Default length, run 1 | 0.20 | 100 | 1e-4 | 15,417 | 414.0 | −78.0 | −0.71 | [`…ep100…_run1`](results/additional_runs/exp0.20_ep100_lr0.0001_run1/) |
| Default length, run 2 | 0.20 | 100 | 1e-4 | 14,740 | 406.0 | −86.0 | −0.72 | [`…ep100…_run2`](results/additional_runs/exp0.20_ep100_lr0.0001_run2/) |
| Long, low LR | 0.10 | 300 | 5e-5 | 45,551 | 478.0 | −14.0 | −0.15 | [`…ep300…`](results/additional_runs/exp0.10_ep300_lr0.00005/) |

**All five runs ran to completion. None was interrupted, and none finished with zero
learning updates** — every `training_summary.json` records `"status": "completed"` with a
non-zero `learning_updates` count.

**The finding I take most seriously is the replication failure.** Rows 1 and 2 use the
same three hyperparameters, the same seed 42, the same evaluation seeds, and the same
hardware. They differ only through non-determinism in GPU floating-point kernels, which
is enough to send the two runs down different trajectories. They ended **254 points
apart**.

That number is the problem, because the differences I am trying to measure are the same
size. The gaps between *different* settings in the table above range from 22 to 348
points, so a 254-point spread *within* one setting swallows most of them. The sharpest
way to see it: the apparent effect of "150 episodes instead of 100" depends entirely on
which of my two 150-episode runs I quote. Against the 100-episode pair it is either
**+86 to +94** (if I quote the 500.0 run) or **+340 to +348** (if I quote the 754.0 run) —
a fourfold difference produced by nothing but GPU floating-point non-determinism.

So the +262 in §5 cannot be attributed to the choice of 150 episodes. It is one draw from
a distribution I have sampled exactly twice, and the two draws are far enough apart that
I cannot say where its centre is.

The 100-episode pair replicated much more tightly (414 and 406, both ~80 points **below**
baseline), which suggests that at 100 episodes the agent reliably has not learned
anything useful, while at 150 it is on the edge — sometimes finding something, sometimes
not.

The 300-episode run is the strongest counter-evidence to a simple "more training is
better" story. It had **more than twice** the updates of the submitted run, plus lower
exploration and a lower learning rate, and finished 14 points below baseline. Its loss
also climbed the highest of the five (0.025 → 0.109, peaking at 0.162). More training,
more loss growth, no score gain.

---

## 7. One observed limitation

**The agent has not learned a policy that generalises across starting conditions — it
improved enormously on some evaluation seeds and got sharply worse on another, and the
same configuration trained twice produced two different agents.**

Within the submitted run, the per-game changes are extreme and not in the same direction:

| Game (seed) | Change | Decisions survived |
|---|---:|---|
| 3 (303) | **+680** | 614 → 760 |
| 5 (505) | **+820** | 534 → 588 |
| 4 (404) | **−440** | 640 → **360** (died 44% sooner) |

On seed 404 the trained agent scored less than half the untrained network's 800 and died
after 360 decisions instead of 640. It is not that the agent became uniformly better and
one game was unlucky — it acquired behaviour that pays off in some maze and ghost
configurations and actively costs it in others. Across the five games it does collect
points faster overall (83.5 → 122.2 points per 100 decisions) while surviving about as
long (589 → 617 decisions), so the behaviour it found is not simply reckless — but it is
not robust either.

There are clean mechanical reasons to expect exactly this at this scale. The
**5,000-transition replay buffer** holds only about eight games of experience, so the
network is trained almost entirely on a sliding window of recent, highly correlated
frames, and it forgets rare, expensive situations long before it can generalise from
them. **Reward clipping** (§3) tells it a pellet and a ghost-eat are worth the same, and
with γ = 0.99 and no termination on life loss, the cost of dying is diffuse and delayed
while the next pellet is immediate and certain. The agent optimises the part of the
problem it can actually see, which varies with where it happens to start.

**Stated plainly:** this is five evaluation games from one run, cross-checked against a
second run of the same configuration that did not reproduce it. It is an observation that
suggests a hypothesis, not a demonstrated effect.

---

## 8. My next experiment

**The single setting I would change: episodes, 150 → 1,500.** Exploration stays at 0.20
and the learning rate stays at 0.0001, so the training budget is the only thing that
moves and any change is attributable to it.

Why that one, and not ε or the learning rate: this run used 86,478 decisions, roughly
346,000 emulator frames — about **0.7%** of a standard 50M-frame Atari DQN run. §6 shows
that at this budget the run-to-run spread (254 points) is as large as the gaps between
hyperparameter settings (22–348 points), so comparing ε or learning-rate values now would
mostly be comparing noise. The only way to make any comparison meaningful is to train long enough
for a real effect to exceed that variance. The cost is manageable: 150 episodes took 329
seconds on a T4, so 1,500 episodes is roughly 55 minutes, comfortably inside one Colab
session. I would also run it **three times** and report the spread, not a single number —
this experiment's clearest lesson is that one run is not a measurement.

Two further changes I would make after that. Both sit outside the three student-facing
settings, and both are, I think, the real ceiling on this setup:

* **Replay capacity 5,000 → 100,000.** At 5,000 the buffer is overwritten about every
  eight games. The original DQN paper uses 1,000,000. No setting of ε, episode count, or
  learning rate can compensate for training on that narrow a window.
* **Decaying ε (1.0 → 0.05) instead of a constant 0.20.** A permanent 20% random-action
  rate caps performance even after the policy becomes good, because one move in five is
  thrown away.

---

## 9. Repository contents

```
pacman_dqn.ipynb      the executed notebook — outputs intact, three choices in section 1
requirements.txt      package pins
pacman_player.py      optional local floating-GIF player (unused on Colab)
results/
  main_run_exp0.20_ep150_lr0.0001/     <- THE SUBMITTED EXPERIMENT
    config.json               settings, hardware, exact package versions
    baseline.json             untrained scores on the five evaluation seeds
    comparison.json           before/after, both conditions, all five games
    demo_scores.json          the every-25-episode checkpoint evaluations
    training.csv              one row per episode: score, steps, loss, elapsed
    training_summary.json     episodes, decisions, learning updates, wall clock
    training_dashboard.png    score / loss / exploration panels
    gifs/
      episode_0000.gif                      untrained baseline gameplay
      episode_0025.gif … episode_0150.gif   checkpoint gameplay every 25 episodes
      final_best.gif                        best of the five final evaluation games
  additional_runs/
    exp0.20_ep150_lr0.0001_repeat/   same settings as the submitted run
    exp0.20_ep100_lr0.0001_run1/     default episode count
    exp0.20_ep100_lr0.0001_run2/     repeat of run1
    exp0.10_ep300_lr0.00005/         longer, lower exploration, lower learning rate
```

Every folder has the same structure, so the five runs can be compared file by file.

**Where the model checkpoints are.** Each run also writes `untrained.pt`, a `*.pt`
checkpoint every 25 episodes, and `trained.pt` — about 6.8 MB each. Following the
notebook's guidance these are **not committed**, to keep this repository small. They stay
on my machine inside the original Colab result ZIPs, kept in `~/Downloads/` —
`20260914_074241_547030.zip` is the submitted run, and the four additional runs are
`20260914_064802_384913.zip`, `20260914_002831_708610.zip`, `20260914_073401_325662.zip`
and `20260914_054748_975263.zip`. Everything needed to reproduce a run — every setting
plus seed 42 — is in each `config.json`.

---

## 10. Credits

Notebook and DQN implementation: [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).
Method: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
Environment: [Arcade Learning Environment](https://ale.farama.org/) via
[Gymnasium](https://gymnasium.farama.org/).
