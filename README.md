# RL Lab — Learn to Play Tic-Tac-Toe
> **Reinforcement Learning Playground & Live Algorithm Visualizer**  
> Tagline: *Learn. Train. Play.*  
> Built for Undergraduate Computer Science & Design Engineering Machine Learning Demonstrations.

---

## 1. Project Overview

**RL Lab** is an educational, interactive web application designed to demonstrate how a Reinforcement Learning agent improves its Tic-Tac-Toe gameplay through **genuine tabular Q-Learning**.

Unlike games with hardcoded heuristic rules or Minimax trees disguised as artificial intelligence, RL Lab implements a **true model-free Temporal Difference learning agent**. The agent starts completely inexperienced with an empty Q-table (selecting random legal moves), and learns to block losses, control territory, and win matches solely from experience and numerical rewards.

### Core Educational Objective
Demonstrate the complete learning journey in three distinct visual stages:
- **01 — Before Training:** Establish an honest baseline by playing against an inexperienced agent with an empty Q-table.
- **02 — Train the Agent:** Execute real Q-Learning in non-blocking batches with configurable hyperparameters and live progress metrics.
- **03 — After Training:** Challenge the trained policy ($\epsilon = 0$), inspect learned state-action Q-values in "How the Agent Decides", and verify empirical improvement using standardized before-and-after benchmarks.

---

## 2. Reinforcement Learning Foundations & Algorithm Design

### A. Agent and Environment
- **Agent:** The learning decision-maker that observes the board, estimates expected future returns $Q(s, a)$, and chooses actions.
- **Environment:** The 3&times;3 Tic-Tac-Toe board and the opponent. The opponent's counter-move is modeled as the environment's transition dynamics $P(s' \mid s, a)$.

### B. State Representation ($S$)
States are canonically encoded from the perspective of the acting agent as a 9-character string:
- `'1'`: Acting agent's marks
- `'2'`: Opponent's marks
- `'0'`: Empty cells

This canonical representation guarantees **perspective invariance**: an opening move in the center has the identical canonical key (`"000010000"`) whether the agent is assigned Player X (moves first) or Player O (moves second).

### C. Legal Action Filtering ($A$)
Actions are grid indices $a \in \{0, 1, \dots, 8\}$. Prior to evaluating or choosing an action, occupied cells are strictly filtered out:
$$A_{\text{legal}}(s) = \{ a \in \{0, \dots, 8\} \mid \text{board}[a] = \text{' '} \}$$
The agent can never overwrite an occupied square or take an illegal move.

### D. Reward System ($R$)
Rewards are assigned consistently from the learning agent's perspective:
- **$+1.0$:** Agent victory (terminal state)
- **$-1.0$:** Opponent victory / Agent defeat (terminal state)
- **$0.0$:** Draw (terminal state)
- **$0.0$:** Non-terminal move (intermediate transition)

### E. The Q-Table
The Q-table is a lookup table mapping each canonical board configuration $s$ to an array of 9 numerical estimates $Q(s, a)$.
- A state-action pair that has never been reached during training is labeled **"Unvisited"** (initialized to 0.0).
- As experience accumulates, $Q(s, a)$ converges toward the expected discounted return from taking action $a$ in state $s$ and acting greedily thereafter.

### F. The Q-Learning Update Rule
Transitions update state-action values using the off-policy Bellman optimality equation:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \big[ r + \gamma \max_{a' \in A_{\text{legal}}(s')} Q(s', a') - Q(s, a) \big]$$

Where:
- $\alpha \in (0, 1]$ is the **learning rate** (step size).
- $\gamma \in [0, 1]$ is the **discount factor**.
- $r$ is the observed immediate reward.
- $\max_{a'} Q(s', a')$ is the maximum value over legal actions in the successor state $s'$.
- **Terminal States:** For terminal next states $s'$, future value is strictly **$0$** without bootstrapping.

#### Numerical Verification Benchmark
The update rule is tested against this exact lecture example:
- Current $Q(s, a) = 4.0$
- Learning rate $\alpha = 0.5$
- Reward $r = 2.0$
- Discount factor $\gamma = 0.9$
- Maximum successor value $\max_{a'} Q(s', a') = 6.0$

$$\begin{aligned}
\text{TD Target} &= 2.0 + (0.9 \times 6.0) = 7.40 \\
\text{TD Error } \delta &= 7.40 - 4.0 = +3.40 \\
\text{Updated } Q(s, a) &= 4.0 + 0.5 \times (+3.40) = \mathbf{5.70}
\end{aligned}$$

The implementation produces **5.70** exactly.

### G. &epsilon;-Greedy Action Selection & Decay
To balance exploration and exploitation:
- With probability $\epsilon$, the agent chooses uniformly at random among all legal moves.
- With probability $1 - \epsilon$, the agent chooses the maximal action $a^* = \arg\max_{a \in A_{\text{legal}}} Q(s, a)$ with uniform random tie-breaking.
- **Key Nuance:** Entering the random exploration branch does *not* imply picking a non-greedy move! If there are $k$ legal moves and $m$ greedy moves, a random draw still has a $m/k$ chance of picking a greedy action:
  $$P(\text{Greedy Action}) = (1 - \epsilon) + \epsilon \cdot \frac{m}{k}$$
- **Decay Schedule:** During training, $\epsilon$ decays linearly from $1.0$ down to $0.05$ as episodes progress, transitioning smoothly from pure discovery to fine-tuning.

---

## 3. The Three-Stage User Journey

```
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│  01 — Before Training   │ -> │   02 — Train Agent      │ -> │   03 — After Training   │
│  Play untrained agent   │    │   Q-Learning in batches │    │   Play trained agent    │
│  100-game baseline eval │    │   Live learning curve   │    │   Side-by-side metrics  │
└─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
```

### Stage 01: Before Training
- **Playable Board:** User can play as Player X (first mover) or Player O (second mover) against an inexperienced agent with an empty Q-table.
- **Baseline Benchmark:** Runs an unbiased 100-game evaluation (50 as Player X, 50 as Player O) against a random baseline opponent. Records the pre-training Win, Draw, and Loss rates.

### Stage 02: Train the Agent
- **Hyperparameter Controls:** Configurable episodes (default 10,000; presets: 2k, 5k, 10k, 20k), learning rate $\alpha$ (0.10), discount factor $\gamma$ (0.95), exploration rate $\epsilon$ (1.00), decay toggle, opponent strategy (Random, Heuristic Defensive, Self-Play), and optional random seed for reproducibility.
- **Non-blocking Execution:** Chunked batches yielded via `setTimeout` prevent the browser from freezing. Can be paused or stopped safely at any time.
- **Live Observable Metrics:** Episodes completed, current $\epsilon$, unique state-action pairs learned, training W/D/L outcomes, and an empirical learning curve plotting periodic evaluation checkpoints.

### Stage 03: After Training
- **Playable Board:** Challenge the trained agent using pure exploitation ($\epsilon = 0.00$). The Q-table is preserved between games without silent retraining or minimax substitution.
- **How the Agent Decides (Explainability Drawer):** Displays the current canonical state, all candidate legal moves, actual learned Q-values (marked "Unvisited" if never seen), greedy badges, selection probabilities, and the latest TD update.
- **Empirical Comparison Panel:** Directly compares the 100-game Baseline Benchmark with the 100-game Post-Training Benchmark under identical unbiased conditions.
- **Policy Export / Import:** Export the learned Q-table to a `.json` file or import pre-trained student checkpoints.

---

## 4. Model-Based Dynamic Programming vs. Model-Free RL

The application includes an educational section clearly distinguishing Dynamic Programming from Q-Learning:

| Dimension | Model-Based Dynamic Programming (DP) | Model-Free Q-Learning (RL) |
| :--- | :--- | :--- |
| **Model Knowledge** | Requires complete prior knowledge of $P(s' \mid s, a)$ and $R(s, a)$. | Requires **zero prior model** of transitions or opponent strategies. |
| **Update Mechanism** | Computes expected value by sweeping across all states: $\sum_{s'} P(s' \mid s, a)[R + \gamma V(s')]$. | Uses single sampled experience tuples $(s, a, r, s')$ from gameplay. |
| **Execution** | Offline mathematical calculation over state space without taking steps. | Agent must interact with the environment, explore, and sample trajectories. |
| **Tic-Tac-Toe Suitability** | Requires modeling exact transition probabilities for every opponent move. | Naturally suited: learns robust strategies directly against diverse human or AI opponents. |

An interactive **3&times;3 GridWorld** with an explicit transition model and reward matrix illustrates Bellman Value Iteration updates alongside the Q-learning engine.

---

## 5. Automated Verification & 28-Point Test Suite

The project includes an automated test suite verifying **28 critical invariants**:

### Game Engine (Tests 1–8)
1. **Empty-Board Initialization:** Initializes to 9 empty cells.
2. **Legal-Action Generation:** Returns only unoccupied squares.
3. **Rejection of Occupied Cells:** Reject moves on occupied cells.
4. **All Eight Winning Lines:** Verifies 3 rows, 3 columns, and 2 diagonals.
5. **Draw Detection:** Full board with no winner detected as draw.
6. **Terminal-State Protection:** Terminal states halt subsequent moves.
7. **Full-Board Behavior:** Full board yields 0 legal actions.
8. **Turn Progression:** Alternating turns apply sequentially.

### Q-Learning Algorithm (Tests 9–18)
9. **Numerical Q-Update Formula:** $4 + 0.5 \times [2 + 0.9 \times 6 - 4] = 5.70$ exact match.
10. **Terminal State Target Zeroed:** Terminal transition targets reward without bootstrapping ($3.00$).
11. **Reward Signs Consistency:** Win is $+1.0$, Loss is $-1.0$, Draw is $0.0$.
12. **Next-State Handling:** Bootstraps from $\max Q(s', a')$ in successor state.
13. **Agent Legal-Action Filtering:** Agent never chooses occupied squares.
14. **$\epsilon = 0$ Greedy Exploitation:** 100% exploitation branch when $\epsilon = 0$.
15. **$\epsilon = 1$ Random Exploration:** 100% exploration branch when $\epsilon = 1$.
16. **Tied Q-Values Tie Breaking:** Maximal candidates resolved with random tie-breaking.
17. **Persistence Between Games:** Q-table preserved across subsequent games.
18. **Separation of Train & Eval:** Evaluation runs do not pollute training episode counts.

### Integration & Usability (Tests 19–28)
19. **Complete Game Finishes:** Simulated matches terminate within 9 moves.
20. **AI Move Always Legal:** Selected move is always an empty cell.
21. **Knowledge Growth:** Training populates actual state-action pairs.
22. **Training Chunking:** Batches divide into non-blocking responsive iterations.
23. **Evaluation Does Not Mutate Q-Values:** Evaluation leaves Q-table strictly unmodified.
24. **Board Reset Preserves Learning:** Game board resets do not clear Q-table.
25. **Resetting Policy Clears Learning:** Explicit reset clears Q-table to 0 entries.
26. **Parameter Modification:** Hyperparameters update dynamically.
27. **Canonical State Encoding:** Perspective invariance correctly maps marks for X and O.
28. **Empty Analytics State:** Initial state contains zero fabricated checkpoints.

### Running the Tests

#### In the Browser
Open `test-runner.html` or open `index.html` and click the **"🧪 Test Suite (28)"** tab.

#### From the Terminal
The test suite exercises the actual JavaScript codebase directly:
```bash
# Direct JavaScript execution (Node.js)
node run_node_tests.js

# Or via Python automation runner (locates Node.js and executes JS suite)
python run_tests.py
```
**Output:**
```text
=======================================================
RL Lab — JavaScript Test Suite (Node.js Engine)
Running on Node.js: v24.13.1
=======================================================

  [PASS] Test 1: Empty-Board Initialization: Board initializes to 9 empty cells
  [PASS] Test 2: Legal-Action Generation: Returns exactly empty cells [1, 3, 5, 6, 7]
  [PASS] Test 3: Rejection of Occupied Cell: Attempt to play on cell 0 returns null
  [PASS] Test 4: All Eight Winning Lines: Accurately detects all 3 rows, 3 columns, and 2 diagonals
  [PASS] Test 5: Draw Detection: Full non-winning board detected as terminal draw
  [PASS] Test 6: Terminal-State Flag: Terminal state flagged immediately to halt subsequent moves
  [PASS] Test 7: Full-Board Behavior: Full board returns 0 legal moves
  [PASS] Test 8: Turn Progression: Moves for alternating players apply sequentially
  [PASS] Test 9: Numerical Q-Update Formula: 4 + 0.5 * [2 + 0.9 * 6 - 4] equals exactly 5.7 (Actual: 5.7, Expected: 5.7)
  [PASS] Test 10: Terminal State Target Zeroed: Terminal transition zeroes future state value (yields 3.0) (Actual: 3, Expected: 3)
  [PASS] Test 11: Reward Signs Consistency: Win is +1.0, Loss is -1.0, Draw is 0.0
  [PASS] Test 12: Next-State Handling: Bootstraps from max Q(s', a') in successor state (Actual: 2.25, Expected: 2.25)
  [PASS] Test 13: Agent Legal-Action Filtering: Agent only selects from empty cells (7 or 8)
  [PASS] Test 14: ε = 0 Greedy Exploitation: 100% exploitation branch when ε = 0
  [PASS] Test 15: ε = 1 Random Exploration: 100% exploration branch when ε = 1
  [PASS] Test 16: Tied Q-Values Random Resolution: Ties between maximal actions break randomly
  [PASS] Test 17: Persistence Between Games: Q-table values persist between matches without silent deletion
  [PASS] Test 18: Separation of Train & Eval: Evaluation does not increment training episode count
  [PASS] Test 19: Complete Game Finishes: Game reaches terminal state within 9 moves
  [PASS] Test 20: AI Move Always Legal: Agent selections never choose occupied squares
  [PASS] Test 21.1: Initially Zero Visited States: New agent has 0 visited state-action pairs
  [PASS] Test 21.2: Knowledge Growth: Training updates actual visited states and Q-table entries
  [PASS] Test 22: Async Training Function Exists: Asynchronous chunked training interface exists
  [PASS] Test 23: Evaluation Does Not Mutate Q-Values: Evaluation policy execution leaves Q-values completely unchanged
  [PASS] Test 24: Board Reset Preserves Learning: Resetting game board does not wipe learned Q-table
  [PASS] Test 25: Resetting Policy Clears Learning: Explicit resetPolicy empties Q-table and zeroes episode count
  [PASS] Test 26: Parameter Modification: Hyperparameters update dynamically
  [PASS] Test 27: Canonical Perspective Invariance: Encodes perspective consistently for both X and O
  [PASS] Test 28: Empty Analytics State: Initial state contains 0 episodes and empty evaluation history
  [PASS] Test 29: Read-Only Safety: Inspecting states and greedy action selection never add phantom entries to Q-table
  [PASS] Test 30: Export/Import Fidelity & Validation: Policy JSON preserves exact visited action set (including Q=0) and safely rejects malformed payloads
  [PASS] Test 31: Training Lifecycle & Safe Reset: Active asynchronous training transitions through states and cancels cleanly on reset
  [PASS] Test 32: Balanced Opponent Curriculum: Balanced training strategy generates strictly valid actions from curriculum distribution
  [PASS] Test 33: Epsilon Decay Schedule Synchronization: Training engine snapshots current exploration slider setting as starting epsilon

=======================================================
ALL 34/34 JAVASCRIPT INVARIANTS PASSED (100% SUCCESS)!
=======================================================
```

---

## 6. How to Run Locally

### Direct Browser Opening (No Server Needed)
Double-click `index.html` or open in any web browser:
```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

### Local Static Server (Optional)
```bash
python -m http.server 8000
```
Navigate to: `http://localhost:8000`

---

## 7. Deploying to Vercel

1. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: complete 3-stage RL Lab educational workflow"
   ```
2. **Push to GitHub:**
   ```bash
   git push origin main
   ```
3. **Import to Vercel:**
   - Go to [Vercel](https://vercel.com) &rarr; **Add New...** &rarr; **Project**.
   - Select your repository (`rl-lab`).
   - Framework Preset: `Other` (Static Site).
   - Build Command: leave blank.
   - Output Directory: `./`.
   - Click **Deploy**.
4. The deployment will be live at `https://<your-project>.vercel.app`.

---

## 8. Limitations of Tabular Q-Learning in Tic-Tac-Toe

1. **Dependence on Opponent Exploration:**
   When trained strictly against a *Random Opponent*, the agent learns to exploit random mistakes and achieve &gt;90% win rate against random play. However, because a random opponent rarely plays optimal defensive moves, the agent may not encounter all defensive counter-lines unless trained against diverse opponents or self-play.
2. **Tabular State Space:**
   While Tic-Tac-Toe has a finite state space ($\approx 5,478$ valid canonical board states), tabular representations do not generalize knowledge across symmetrically similar positions without explicit rotation/reflection augmentation or function approximation.
3. **Finite Training Convergence:**
   A finite state space does not guarantee rapid convergence to an unbeatable policy under arbitrary hyperparameters. Selecting a very low learning rate ($\alpha$) or insufficient training episodes can result in suboptimal value estimates for rarely visited endgames.

---

## 9. License
MIT License. Created for university computer science and design engineering machine learning educational courses.