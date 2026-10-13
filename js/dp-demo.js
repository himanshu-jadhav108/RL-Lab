/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: dp-demo.js
 * Purpose: Model-Based Dynamic Programming educational demonstration.
 *          Explicitly defines a 3x3 GridWorld environment with known transition probabilities
 *          and reward matrices to illustrate Bellman Value Iteration, contrasting it clearly
 *          with Model-Free Q-Learning.
 */

class DynamicProgrammingDemo {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.gamma = 0.9;
    this.iteration = 0;
    this.isConverged = false;

    // 3x3 GridWorld definition
    // (0,0) = Start
    // (1,1) = Hazard (-10, terminal)
    // (2,2) = Goal (+10, terminal)
    // Normal step reward = -1
    this.rows = 3;
    this.cols = 3;

    // V(s) table
    this.V = Array(9).fill(0.0);

    // Actions: 0: Up, 1: Right, 2: Down, 3: Left
    this.actions = [
      { name: 'Up', dr: -1, dc: 0, arrow: '↑' },
      { name: 'Right', dr: 0, dc: 1, arrow: '→' },
      { name: 'Down', dr: 1, dc: 0, arrow: '↓' },
      { name: 'Left', dr: 0, dc: -1, arrow: '←' }
    ];

    this.init();
  }

  coordToIndex(r, c) {
    return r * 3 + c;
  }

  indexToCoord(idx) {
    return { r: Math.floor(idx / 3), c: idx % 3 };
  }

  isTerminal(idx) {
    return idx === 4 || idx === 8; // (1,1) Hazard or (2,2) Goal
  }

  /**
   * Deterministic transition function P(s' | s, a) with boundary clipping.
   * Small well-defined model with known transitions and rewards.
   */
  getTransition(s, a) {
    if (this.isTerminal(s)) {
      return { nextState: s, reward: 0, isTerminal: true };
    }

    const { r, c } = this.indexToCoord(s);
    const action = this.actions[a];

    let nextR = Math.max(0, Math.min(2, r + action.dr));
    let nextC = Math.max(0, Math.min(2, c + action.dc));
    const nextState = this.coordToIndex(nextR, nextC);

    let reward = -1.0; // standard step penalty
    if (nextState === 8) reward = 10.0;  // Goal
    if (nextState === 4) reward = -10.0; // Hazard

    return { nextState, reward, isTerminal: this.isTerminal(nextState) };
  }

  /**
   * Perform one sweep of Bellman Optimality Value Iteration:
   * V_{k+1}(s) = max_a [ R(s,a) + gamma * V_k(s') ]
   */
  stepValueIteration() {
    if (this.isConverged) return;

    const nextV = [...this.V];
    let maxDelta = 0;

    for (let s = 0; s < 9; s++) {
      if (this.isTerminal(s)) {
        nextV[s] = 0.0; // Terminal value convention
        continue;
      }

      let bestVal = -Infinity;
      for (let a = 0; a < 4; a++) {
        const { nextState, reward, isTerminal } = this.getTransition(s, a);
        const expectedVal = reward + (isTerminal ? 0 : this.gamma * this.V[nextState]);
        if (expectedVal > bestVal) {
          bestVal = expectedVal;
        }
      }

      maxDelta = Math.max(maxDelta, Math.abs(bestVal - this.V[s]));
      nextV[s] = Number(bestVal.toFixed(3));
    }

    this.V = nextV;
    this.iteration++;

    if (maxDelta < 1e-4) {
      this.isConverged = true;
    }

    this.render();
  }

  /**
   * Compute greedy policy from current values.
   */
  getGreedyPolicy(s) {
    if (this.isTerminal(s)) return '—';

    let bestVal = -Infinity;
    let bestArrow = '—';

    for (let a = 0; a < 4; a++) {
      const { nextState, reward, isTerminal } = this.getTransition(s, a);
      const expectedVal = reward + (isTerminal ? 0 : this.gamma * this.V[nextState]);
      if (expectedVal > bestVal) {
        bestVal = expectedVal;
        bestArrow = this.actions[a].arrow;
      }
    }

    return bestArrow;
  }

  reset() {
    this.V = Array(9).fill(0.0);
    this.iteration = 0;
    this.isConverged = false;
    this.render();
  }

  runUntilConvergence() {
    let safetyCounter = 0;
    while (!this.isConverged && safetyCounter < 100) {
      this.stepValueIteration();
      safetyCounter++;
    }
  }

  init() {
    this.render();
  }

  render() {
    if (!this.container) return;

    let gridHtml = '';
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = this.coordToIndex(r, c);
        const val = this.V[idx];
        const isGoal = idx === 8;
        const isHazard = idx === 4;
        const isStart = idx === 0;

        let badge = '';
        let cellClass = 'dp-cell';
        if (isGoal) {
          cellClass += ' dp-cell-goal';
          badge = '<span class="dp-tag goal-tag">Goal (+10)</span>';
        } else if (isHazard) {
          cellClass += ' dp-cell-hazard';
          badge = '<span class="dp-tag hazard-tag">Hazard (-10)</span>';
        } else if (isStart) {
          badge = '<span class="dp-tag start-tag">Start</span>';
        }

        const policyArrow = this.getGreedyPolicy(idx);

        gridHtml += `
          <div class="${cellClass}">
            <div class="dp-cell-header">
              <span class="dp-cell-coord">(${r},${c})</span>
              ${badge}
            </div>
            <div class="dp-cell-value">V = ${val.toFixed(2)}</div>
            <div class="dp-cell-policy">
              <span class="dp-policy-label">Policy</span>
              <span class="dp-policy-arrow">${policyArrow}</span>
            </div>
          </div>
        `;
      }
    }

    this.container.innerHTML = `
      <div class="dp-visualizer-wrapper">
        <div class="dp-header-bar">
          <div class="dp-header-info">
            <div class="dp-header-title">Known 3×3 GridWorld Model (γ = ${this.gamma})</div>
            <div class="dp-header-sub">
              Iteration: <strong class="font-mono">${this.iteration}</strong>
              ${this.isConverged ? '<span class="badge badge-success" style="margin-left: 6px;">✓ Converged</span>' : ''}
            </div>
          </div>
          <div class="dp-actions-row">
            <button class="btn btn-sm btn-outline" id="btn-dp-step">Step 1 Iteration</button>
            <button class="btn btn-sm btn-primary" id="btn-dp-converge">Run to Convergence</button>
            <button class="btn btn-sm btn-secondary" id="btn-dp-reset">Reset</button>
          </div>
        </div>
        <div class="dp-grid-board">
          ${gridHtml}
        </div>
      </div>
    `;

    // Attach listeners
    const stepBtn = document.getElementById('btn-dp-step');
    const convBtn = document.getElementById('btn-dp-converge');
    const resetBtn = document.getElementById('btn-dp-reset');

    if (stepBtn) stepBtn.onclick = () => this.stepValueIteration();
    if (convBtn) convBtn.onclick = () => this.runUntilConvergence();
    if (resetBtn) resetBtn.onclick = () => this.reset();
  }
}

// Export for module and window environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DynamicProgrammingDemo };
} else {
  window.DynamicProgrammingDemo = DynamicProgrammingDemo;
}
