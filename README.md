# RL Lab — Learn to Play Tic-Tac-Toe
> **Interactive Reinforcement Learning Playground & Algorithm Visualizer**  
> Built for Undergraduate Machine Learning Presentations and University Demonstrations.

---

## 1. Project Overview

**RL Lab** is an educational web application designed to demonstrate true, tabular Reinforcement Learning (RL) operating in real-time inside the browser. Unlike conventional games that employ hard-coded heuristics or Minimax search disguised as AI, **RL Lab implements true model-free Tabular Q-Learning**. The agent starts with zero initial knowledge about Tic-Tac-Toe and discovers winning and defensive strategies purely through simulated experience and Temporal Difference updates.

### Educational Topics Covered
1. **Exploration versus Exploitation:** The fundamental trade-off between trying new actions to discover superior states and leveraging existing value estimates.
2. **&epsilon;-Greedy Action Selection:** Action branch partitioning and the critical mathematical nuance: *entering the random exploration branch does not guarantee taking a non-greedy action*.
3. **Dynamic Programming (DP) vs. Model-Free RL:** A clear theoretical and interactive distinction between model-based dynamic programming (known transition model $P(s'|s,a)$ and reward function $R$) and model-free Temporal Difference learning ($Q$-Learning).
4. **The Q-Learning Update Rule:** Step-by-step Temporal Difference calculation, TD Target, and TD Error.
5. **Reinforcement Learning for Turn-Based Games:** MDP formulation from the learning player's canonical perspective.

---

## 2. Key Enhancements & Presentation Features

1. **Light / Dark Theme Toggle:**
   - Full CSS design system with light and dark palettes.
   - Automatically detects OS preference (`prefers-color-scheme`) and persists user choices in `localStorage`.
   - The canvas learning curve and grid visualizers dynamically adapt their axes, grids, and glowing gradients to the active theme.
2. **Interactive Lecture Teaching Scenarios:**
   - Dropdown presets allowing presenters to immediately jump to key board configurations:
     - **Empty Board (Opening):** Inspect opening cell preferences.
     - **Center Control:** Demonstrates value of central territory.
     - **Immediate Win Threat:** Demonstrates offensive greedy exploitation ($Q(s,a) = +1.0$).
     - **Defensive Block:** Demonstrates defensive prioritization to prevent negative reward ($-1.0$).
     - **Corner Fork:** Shows spatial threat evaluation.
3. **Action Probability Visual Distribution:**
   - Q-value inspector displays animated mini-progress bars showing the effective probability distribution $P(a)$ for every legal move.
4. **Native Web Audio Synthesizer:**
   - Clean, subtle audio feedback for moves, wins, and draws using 100% native Web Audio API (zero audio file downloads, mute toggle included).
5. **Master Policy Instant Loader:**
   - One-click **"⭐ Load Master Policy"** button executes 3,500 training episodes against mixed opponents in ~100ms, allowing presenters to immediately showcase an expert agent without waiting.
6. **Policy Import & Export:**
   - Full JSON export and import for saving, sharing, and comparing Q-tables.

---

## 3. Mathematical Formulation & Algorithmic Design

### A. The Markov Decision Process (MDP) for Turn-Based 2-Player Games
In RL Lab, 2-player Tic-Tac-Toe is modeled as a single-agent MDP from the perspective of the learning agent:
- **State Space ($S$):** Canonical 9-cell board string where `'1'` represents the agent's marks, `'2'` represents the opponent's marks, and `'0'` represents empty cells. This canonical encoding guarantees policy invariance whether the agent is assigned `'X'` (first mover) or `'O'` (second mover).
- **Action Space ($A$):** Indices $\{0, 1, \dots, 8\}$. Legal action filtering ensures the agent never attempts to place a mark in an already occupied square:
  $$A_{\text{legal}}(s) = \{ a \in \{0, \dots, 8\} \mid \text{board}[a] = \text{' '} \}$$
- **Transition Dynamics ($P(s' \mid s, a)$):** When the agent takes legal action $a$ in state $s$, the environment responds by allowing the opponent to make a counter-move according to their chosen policy (Random baseline, Heuristic defensive/offensive, or Self-play). The resulting configuration is successor state $s'$.
- **Reward Function ($R$):**
  - **$+1.0$:** Agent victory (terminal transition)
  - **$-1.0$:** Opponent victory / Agent defeat (terminal transition)
  - **$0.0$:** Draw (terminal transition)
  - **$0.0$:** Ongoing match (intermediate non-terminal transition to next agent turn)

### B. Tabular Q-Learning Update Equation
The action-value function $Q(s, a)$ is updated using the Bellman Optimality Temporal Difference equation:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \big[ R + \gamma \max_{a' \in A_{\text{legal}}(s')} Q(s', a') - Q(s, a) \big]$$

Where:
- $\alpha \in (0, 1]$ is the **learning rate** (step size).
- $\gamma \in [0, 1]$ is the **discount factor**.
- For **terminal successor states** $s'$, the future discounted value $\gamma \max_{a'} Q(s', a')$ is strictly **$0$**.

#### Numerical Benchmark Verification
The application and test suite verify this formula against the following test case:
- $\text{Current } Q(s, a) = 4.0$
- $\alpha = 0.5$
- $R = 2.0$
- $\gamma = 0.9$
- $\max_{a'} Q(s', a') = 6.0$

$$\begin{aligned}
\text{TD Target} &= R + \gamma \max_{a'} Q(s', a') = 2.0 + 0.9 \times 6.0 = 7.40 \\
\text{TD Error } \delta &= \text{TD Target} - Q(s, a) = 7.40 - 4.0 = +3.40 \\
\text{New } Q(s, a) &= 4.0 + 0.5 \times (+3.40) = \mathbf{5.70}
\end{aligned}$$

The implementation produces **5.70** exactly.

### C. &epsilon;-Greedy Action Selection
Let legal actions be $A_{\text{legal}}$ with size $k = |A_{\text{legal}}|$, and let maximal Q-value actions be $A^* = \arg\max_{a \in A_{\text{legal}}} Q(s, a)$ with size $m = |A^*|$.

1. Sample uniform random variable $u \sim U(0, 1)$.
2. **Exploration Branch ($u < \epsilon$):** Pick uniformly at random from $A_{\text{legal}}$.
   - Each legal action has probability $\frac{\epsilon}{k}$ of being sampled in this branch.
3. **Exploitation / Greedy Branch ($u \ge \epsilon$):** Pick uniformly at random from $A^*$.
   - Each maximal greedy action has probability $\frac{1 - \epsilon}{m}$ of being selected.
4. **Effective Action Probability:**
   $$P(a) = (1 - \epsilon) \cdot \frac{\mathbb{I}(a \in A^*)}{m} + \frac{\epsilon}{k}$$

> **Key Lecture Insight:** Because the random exploration branch selects uniformly across *all* legal actions (including greedy ones), an exploration step can still take a greedy action. The total probability of taking a greedy action is $(1 - \epsilon) + \epsilon \cdot \frac{m}{k} > (1 - \epsilon)$.

---

## 4. Dynamic Programming vs. Model-Free RL

A dedicated educational module contrasts Dynamic Programming with Q-Learning:

| Dimension | Model-Based Dynamic Programming (DP) | Model-Free Q-Learning (RL) |
| :--- | :--- | :--- |
| **Model Requirements** | Requires exact mathematical model of transitions $P(s' \mid s, a)$ and reward function $R(s, a)$. | Requires **zero prior model** of transitions or opponent policies. |
| **Update Mechanism** | Computes expected value by sweeping across all possible transitions: $\sum_{s'} P(s' \mid s, a)[R + \gamma V(s')]$. | Uses sampled experience tuples $(s, a, r, s')$ from empirical gameplay. |
| **Execution** | Offline mathematical calculation over state space without taking physical actions. | Agent must interact with the environment, explore, and sample trajectories. |
| **Tic-Tac-Toe Relevance** | Would require modeling the exact probability distribution of every opponent's moves. | Naturally suited: learns robust strategies directly against arbitrary human or AI opponents. |

The application includes an interactive **3&times;3 GridWorld** with an explicit transition model and reward matrix to illustrate **Bellman Value Iteration** updates alongside tabular Q-learning.

---

## 5. Project Structure

```
RL Lab/
│
├── index.html           # Main semantic, accessible dashboard interface (6 tabs, theme toggle)
├── test-runner.html     # Dedicated in-browser test runner page
├── vercel.json          # Vercel deployment configuration (clean URLs & security headers)
├── run_tests.py         # Terminal test script for automated invariant verification
├── README.md            # Comprehensive documentation and lecture guide
│
├── css/
│   └── style.css        # Academic light & dark design system, glassmorphism, responsive
│
└── js/
    ├── game.js          # Tic-Tac-Toe environment logic, state hashing, win/draw detection
    ├── agent.js         # Tabular Q-Learning agent, ε-greedy logic, async training, JSON export/import
    ├── analytics.js     # Responsive high-DPI Canvas learning curve with theme awareness & gradients
    ├── dp-demo.js       # Dynamic Programming 3x3 GridWorld Value Iteration demo
    ├── tests.js         # Complete in-browser test suite covering all 12 core requirements
    └── app.js           # UI controller, audio synthesizer, teaching scenarios, theme management
```

---

## 6. Automated Verification & Test Suite

The test suite validates **12 comprehensive invariants**:
1. **Win Detection:** Evaluates rows, columns, and diagonal lines for both `'X'` and `'O'`.
2. **Draw Detection:** Flags full boards with no 3-in-a-row marks as terminal draws.
3. **Legal Action Filtering:** Ensures occupied cells are never selected or overwritten.
4. **Terminal State Handling:** Terminal states halt game continuation and assign terminal rewards.
5. **Exact Numerical Q-Update:** Computes $4 + 0.5 \times [2 + 0.9 \times 6 - 4] = 5.70$ and verifies terminal next-state zeroing.
6. **&epsilon; Boundary Conditions:** $\epsilon = 0.0$ selects greedy actions 100% of the time; $\epsilon = 1.0$ executes 100% exploration.
7. **Reward Perspective Consistency:** +1.0 for agent win, -1.0 for agent defeat, 0.0 for draw.
8. **Reset Behavior:** `resetPolicy()` empties the Q-table and resets episode counters.
9. **Statistics Consistency:** Verifies $\text{Wins} + \text{Draws} + \text{Losses} = \text{Total Games}$.
10. **UI Synchronization & State Encoding:** Verifies canonical state hash mapping (`'102000000'`).
11. **Responsive Coordinate Mapping:** Validates 1D index to $(row, col)$ grid mapping.
12. **Standalone Deployment Readiness:** Verifies zero external remote dependencies.

### Running the Tests

#### Option 1: In the Browser
- Open `test-runner.html` in any browser, or
- Open `index.html` and click the **"🧪 Test Suite"** tab to run all tests with 1 click.

#### Option 2: From the Terminal
Run the standalone Python validation script:
```bash
python run_tests.py
```
**Output:**
```
Running RL Lab Invariant Verification Suite...

  [PASS] Test 1.1: Row win for X: Detected X win on row 0
  [PASS] Test 1.2: Diagonal win for O: Detected O win on main diagonal
  [PASS] Test 2: Draw detection: Full non-winning board is draw
  [PASS] Test 3: Legal-action filtering: Occupied cells filtered out
  [PASS] Test 4: Terminal-state handling: Terminal state halts play
  [PASS] Test 5.1: Numerical Q update: 4 + 0.5 * [2 + 0.9*6 - 4] = 5.7 (Expected 5.7)
  [PASS] Test 5.2: Terminal next-state future value zeroed: Terminal next state yields 3.0 (Expected 3.0)
  [PASS] Test 6: Epsilon boundaries: Bounds 0.0 and 1.0 are well defined
  [PASS] Test 7: Reward perspective: Agent perspective rewards are consistent
  [PASS] Test 8: Reset behavior: Q-table clears properly on reset
  [PASS] Test 9: Statistics consistency: Sum 15+10+5 == 30
  [PASS] Test 10: Canonical perspective encoding: Canonical key is 102000000
  [PASS] Test 11: Action to Grid Coordinates: Coordinates correctly aligned
  [PASS] Test 12: Offline deployment readiness: 100% client side static web application ready for Vercel

Completed: 14/14 tests passed successfully!
```

---

## 7. How to Run Locally

### Direct Browser Opening (No Server Needed)
Double-click `index.html` or open it in Chrome, Firefox, Safari, or Edge:
```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

### Local Static Web Server (Optional)
```bash
python -m http.server 8000
```
Then navigate to: `http://localhost:8000`

---

## 8. Step-by-Step Vercel Deployment Instructions

1. **Commit code to Git:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete RL Lab with themes and audio"
   ```
2. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/<your-username>/rl-lab.git
   git branch -M main
   git push -u origin main
   ```
3. **Import to Vercel:**
   - Log into [Vercel](https://vercel.com) &rarr; **Add New...** &rarr; **Project**.
   - Select your GitHub repo (`rl-lab`).
   - Framework Preset: `Other`
   - Build Command: leave blank
   - Output Directory: `./`
   - Click **Deploy**.
4. The site is live at `https://<your-project>.vercel.app`!

---

## 9. Mobile & Accessibility Support
- **Touch-Friendly:** Cell touch targets are $\ge 84\text{px}$ on mobile with distinct tap animations and sound feedback.
- **Responsive Layout:** Grid dynamically shifts from 2-column desktop dashboards to single-column mobile views.
- **Keyboard Navigation:** Cells support Tab navigation and activation via `Enter` or `Space`.
- **Accessibility:** ARIA labels, semantic roles, high-contrast text ratios, and screen-reader status updates.

---

## 10. License
MIT License. Created for university lecture presentations, Machine Learning educational demonstrations, and reinforcement learning research workshops.
#   R L - L a b  
 