/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: agent.js
 * Purpose: Tabular Q-Learning agent, epsilon-greedy action selection with optional decay,
 *          visited state-action tracking, non-blocking asynchronous batch training,
 *          and unbiased two-perspective evaluation.
 */

class QLearningAgent {
  constructor(options = {}) {
    this.game = options.game || new TicTacToeGame();
    this.alpha = options.alpha !== undefined ? options.alpha : 0.1;       // Learning rate (default 0.1)
    this.gamma = options.gamma !== undefined ? options.gamma : 0.95;     // Discount factor (default 0.95)
    this.epsilon = options.epsilon !== undefined ? options.epsilon : 1.0; // Initial exploration rate
    this.initialEpsilon = this.epsilon;
    this.minEpsilon = options.minEpsilon !== undefined ? options.minEpsilon : 0.05;
    this.decayEpsilon = options.decayEpsilon !== undefined ? options.decayEpsilon : true;
    this.defaultQ = options.defaultQ !== undefined ? options.defaultQ : 0.0;

    // Q-Table: stateKey -> Array of 9 numbers
    this.qTable = new Map();

    // Set of visited (stateKey + ":" + action) pairs
    this.visitedActions = new Set();

    // Seeded PRNG state (null for Math.random)
    this.prngSeed = null;

    // Training state lifecycle
    this.isTraining = false;
    this.shouldStopTraining = false;
    this.trainingState = 'idle'; // 'idle' | 'training' | 'stopping'
    this.trainingTimerId = null;
    this.totalEpisodesTrained = 0;

    // Cumulative training outcomes (during exploration)
    this.trainingStats = {
      wins: 0,
      draws: 0,
      losses: 0,
      episodes: 0
    };

    // Separate Evaluation Records
    this.baselineEval = null;      // Benchmark run on untrained agent
    this.postTrainingEval = null;  // Benchmark run on trained agent
    this.evaluationHistory = [];   // Learning curve checkpoints

    // Last decision metadata for explainability drawer
    this.lastDecision = null;
    this.lastUpdate = null;
  }

  /**
   * Simple Mulberry32 PRNG for reproducible training runs.
   */
  setSeed(seed) {
    if (seed === null || seed === undefined || seed === '') {
      this.prngSeed = null;
    } else {
      let s = typeof seed === 'string' ? this.hashString(seed) : Number(seed);
      this.prngSeed = s >>> 0;
    }
  }

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = Math.imul(31, hash) + str.charCodeAt(i) | 0;
    }
    return hash;
  }

  random() {
    if (this.prngSeed === null) {
      return Math.random();
    }
    let t = this.prngSeed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }

  /**
   * Reset learned Q-table, visited tracking, and training history safely.
   */
  resetPolicy() {
    this.stopTraining();
    this.isTraining = false;
    this.trainingState = 'idle';
    this.qTable.clear();
    this.visitedActions.clear();
    this.totalEpisodesTrained = 0;
    this.epsilon = this.initialEpsilon;
    this.trainingStats = { wins: 0, draws: 0, losses: 0, episodes: 0 };
    this.evaluationHistory = [];
    this.baselineEval = null;
    this.postTrainingEval = null;
    this.lastDecision = null;
    this.lastUpdate = null;
  }

  /**
   * Non-mutating read-only lookup of Q-values.
   * Returns existing entry or a zero array WITHOUT adding to qTable.
   * @param {string} stateKey
   * @returns {number[]} Array of 9 Q-values
   */
  peekQValues(stateKey) {
    if (this.qTable.has(stateKey)) {
      return this.qTable.get(stateKey);
    }
    return new Array(9).fill(this.defaultQ);
  }

  /**
   * Get Q-values array for canonical state. Initializes if unseen.
   * @param {string} stateKey 
   * @returns {number[]} Array of 9 Q-values
   */
  getQValues(stateKey) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, new Array(9).fill(this.defaultQ));
    }
    return this.qTable.get(stateKey);
  }

  /**
   * Check if a specific state-action pair has ever been visited and updated.
   */
  hasVisited(stateKey, action) {
    return this.visitedActions.has(`${stateKey}:${action}`);
  }

  /**
   * Get value and visited status for inspector without mutating Q-table.
   */
  getQValueInfo(stateKey, action) {
    const visited = this.hasVisited(stateKey, action);
    const qValues = this.peekQValues(stateKey);
    return {
      value: qValues[action],
      visited: visited
    };
  }

  /**
   * Number of unique state-action pairs learned.
   */
  getVisitedCount() {
    return this.visitedActions.size;
  }

  /**
   * Static/pure numerical Q-Learning update rule verification function.
   * Formula: Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
   * For terminal next states, max_a' Q(s',a') is strictly 0.
   */
  static computeQUpdate(currentQ, alpha, reward, gamma, maxNextQ, isTerminal = false) {
    const nextStateTarget = isTerminal ? 0 : (gamma * maxNextQ);
    const tdTarget = reward + nextStateTarget;
    const tdError = tdTarget - currentQ;
    return currentQ + alpha * tdError;
  }

  /**
   * Perform in-place Q-value update. Only current state is added to qTable.
   */
  updateQ(stateKey, action, reward, nextStateKey, nextLegalMoves, isTerminal) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, new Array(9).fill(this.defaultQ));
    }
    const qValues = this.qTable.get(stateKey);
    const currentQ = qValues[action];

    let maxNextQ = 0;
    if (!isTerminal && nextStateKey && nextLegalMoves && nextLegalMoves.length > 0) {
      const nextQValues = this.peekQValues(nextStateKey);
      maxNextQ = Math.max(...nextLegalMoves.map(a => nextQValues[a]));
    }

    const updatedQ = QLearningAgent.computeQUpdate(
      currentQ,
      this.alpha,
      reward,
      this.gamma,
      maxNextQ,
      isTerminal
    );

    qValues[action] = updatedQ;
    this.visitedActions.add(`${stateKey}:${action}`);

    const updateInfo = {
      stateKey,
      action,
      reward,
      previousQ: currentQ,
      updatedQ,
      tdTarget: reward + (isTerminal ? 0 : this.gamma * maxNextQ),
      tdError: (reward + (isTerminal ? 0 : this.gamma * maxNextQ)) - currentQ,
      isTerminal
    };
    this.lastUpdate = updateInfo;
    return updateInfo;
  }

  /**
   * Select action using epsilon-greedy policy with legal action filtering.
   * Explains exploration vs exploitation, handles ties, and tracks whether
   * an exploratory pick coincided with the greedy action.
   *
   * @param {Array} board - 9-element board
   * @param {string} player - Perspective player ('X' or 'O')
   * @param {number|null} forcedEpsilon - Optional override (e.g. 0 for evaluation)
   */
  selectAction(board, player = 'X', forcedEpsilon = null) {
    const legalMoves = this.game.getLegalMoves(board);
    if (legalMoves.length === 0) return null;

    const stateKey = this.game.getCanonicalState(board, player);
    const qValues = this.peekQValues(stateKey);
    const eps = forcedEpsilon !== null ? forcedEpsilon : this.epsilon;

    // Identify greedy actions among legal moves
    let maxQ = -Infinity;
    for (const a of legalMoves) {
      if (qValues[a] > maxQ) {
        maxQ = qValues[a];
      }
    }

    const greedyActions = legalMoves.filter(a => Math.abs(qValues[a] - maxQ) < 1e-6);

    const roll = this.random();
    const isExplorationBranch = roll < eps;

    let selectedAction;
    let branchType = '';

    if (isExplorationBranch) {
      const randIdx = Math.floor(this.random() * legalMoves.length);
      selectedAction = legalMoves[randIdx];
      branchType = 'exploration';
    } else {
      const randIdx = Math.floor(this.random() * greedyActions.length);
      selectedAction = greedyActions[randIdx];
      branchType = 'exploitation';
    }

    const isGreedyAction = greedyActions.includes(selectedAction);

    const decision = {
      action: selectedAction,
      branch: branchType,
      roll,
      epsilon: eps,
      isGreedy: isGreedyAction,
      maxQ,
      greedyActions,
      legalMoves,
      stateKey,
      qValues: [...qValues],
      numLegal: legalMoves.length,
      probExploration: eps,
      probExploitation: 1 - eps,
      effectiveActionProb: (
        (1 - eps) * (isGreedyAction ? (1 / greedyActions.length) : 0) +
        (eps * (1 / legalMoves.length))
      )
    };

    this.lastDecision = decision;
    return decision;
  }

  /**
   * Opponent policy during training & evaluation.
   */
  selectOpponentAction(board, opponentPlayer, policy = 'random') {
    const legalMoves = this.game.getLegalMoves(board);
    if (legalMoves.length === 0) return null;

    if (policy === 'random') {
      const idx = Math.floor(this.random() * legalMoves.length);
      return legalMoves[idx];
    }

    if (policy === 'balanced') {
      // Documented curriculum mixture: 60% random exploratory moves for state-space coverage,
      // 40% tactical heuristic moves for immediate threat defense and winning execution.
      if (this.random() < 0.60) {
        return this.selectOpponentAction(board, opponentPlayer, 'random');
      } else {
        return this.selectOpponentAction(board, opponentPlayer, 'heuristic');
      }
    }

    if (policy === 'heuristic') {
      const agentPlayer = opponentPlayer === 'X' ? 'O' : 'X';

      // 1. Immediate win
      for (const a of legalMoves) {
        const testBoard = this.game.makeMove(a, opponentPlayer, board);
        if (testBoard && this.game.checkWinner(testBoard).winner === opponentPlayer) {
          return a;
        }
      }

      // 2. Immediate block
      for (const a of legalMoves) {
        const testBoard = this.game.makeMove(a, agentPlayer, board);
        if (testBoard && this.game.checkWinner(testBoard).winner === agentPlayer) {
          return a;
        }
      }

      // 3. Take center if open
      if (legalMoves.includes(4)) return 4;

      // 4. Otherwise random legal move
      const idx = Math.floor(this.random() * legalMoves.length);
      return legalMoves[idx];
    }

    if (policy === 'self-play') {
      const oppDecision = this.selectAction(board, opponentPlayer, 0.05);
      return oppDecision ? oppDecision.action : legalMoves[0];
    }

    return legalMoves[0];
  }

  /**
   * Play and learn from 1 training episode.
   */
  trainEpisode(agentPlayer = 'X', opponentPolicy = 'random') {
    let board = this.game.reset();
    const opponentPlayer = agentPlayer === 'X' ? 'O' : 'X';
    let turns = 0;

    if (agentPlayer === 'O') {
      const oppAction = this.selectOpponentAction(board, opponentPlayer, opponentPolicy);
      if (oppAction !== null) {
        board = this.game.makeMove(oppAction, opponentPlayer, board);
        turns++;
      }
    }

    while (true) {
      const currentCanonical = this.game.getCanonicalState(board, agentPlayer);
      const legalMoves = this.game.getLegalMoves(board);
      if (legalMoves.length === 0) break;

      const decision = this.selectAction(board, agentPlayer);
      const agentAction = decision.action;

      const boardAfterAgent = this.game.makeMove(agentAction, agentPlayer, board);
      turns++;
      const agentOutcome = this.game.checkWinner(boardAfterAgent);

      if (agentOutcome.isTerminal) {
        if (agentOutcome.winner === agentPlayer) {
          this.updateQ(currentCanonical, agentAction, 1.0, null, [], true);
          this.trainingStats.wins++;
          this.totalEpisodesTrained++;
          return { winner: agentPlayer, turns };
        } else {
          this.updateQ(currentCanonical, agentAction, 0.0, null, [], true);
          this.trainingStats.draws++;
          this.totalEpisodesTrained++;
          return { winner: 'draw', turns };
        }
      }

      const oppAction = this.selectOpponentAction(boardAfterAgent, opponentPlayer, opponentPolicy);
      const boardAfterOpponent = this.game.makeMove(oppAction, opponentPlayer, boardAfterAgent);
      turns++;
      const oppOutcome = this.game.checkWinner(boardAfterOpponent);

      if (oppOutcome.isTerminal) {
        if (oppOutcome.winner === opponentPlayer) {
          this.updateQ(currentCanonical, agentAction, -1.0, null, [], true);
          this.trainingStats.losses++;
          this.totalEpisodesTrained++;
          return { winner: opponentPlayer, turns };
        } else {
          this.updateQ(currentCanonical, agentAction, 0.0, null, [], true);
          this.trainingStats.draws++;
          this.totalEpisodesTrained++;
          return { winner: 'draw', turns };
        }
      }

      const nextCanonical = this.game.getCanonicalState(boardAfterOpponent, agentPlayer);
      const nextLegal = this.game.getLegalMoves(boardAfterOpponent);
      this.updateQ(currentCanonical, agentAction, 0.0, nextCanonical, nextLegal, false);

      board = boardAfterOpponent;
    }

    this.totalEpisodesTrained++;
    return { winner: 'draw', turns };
  }

  /**
   * Run unbiased evaluation (strictly greedy: epsilon = 0).
   * Evaluates exactly half the games playing as X (moves 1st)
   * and half as O (moves 2nd) to eliminate first-player bias!
   */
  evaluateUnbiased(numGames = 100, opponentPolicy = 'random') {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let winsAsX = 0;
    let winsAsO = 0;

    const gamesPerRole = Math.floor(numGames / 2);
    const totalActualGames = gamesPerRole * 2;

    for (let roleIndex = 0; roleIndex < 2; roleIndex++) {
      const agentRole = roleIndex === 0 ? 'X' : 'O';
      const opponentRole = agentRole === 'X' ? 'O' : 'X';

      for (let g = 0; g < gamesPerRole; g++) {
        let board = this.game.reset();

        // If agent is O, opponent moves first
        if (agentRole === 'O') {
          const oppAction = this.selectOpponentAction(board, opponentRole, opponentPolicy);
          if (oppAction !== null) {
            board = this.game.makeMove(oppAction, opponentRole, board);
          }
        }

        while (true) {
          // Agent moves with epsilon = 0.0 (pure exploitation)
          const agentDecision = this.selectAction(board, agentRole, 0.0);
          if (!agentDecision) break;

          board = this.game.makeMove(agentDecision.action, agentRole, board);
          let outcome = this.game.checkWinner(board);
          if (outcome.isTerminal) {
            if (outcome.winner === agentRole) {
              wins++;
              if (agentRole === 'X') winsAsX++;
              else winsAsO++;
            } else {
              draws++;
            }
            break;
          }

          // Opponent moves
          const oppAction = this.selectOpponentAction(board, opponentRole, opponentPolicy);
          board = this.game.makeMove(oppAction, opponentRole, board);
          outcome = this.game.checkWinner(board);
          if (outcome.isTerminal) {
            if (outcome.winner === opponentRole) {
              losses++;
            } else {
              draws++;
            }
            break;
          }
        }
      }
    }

    const winRate = Number(((wins / totalActualGames) * 100).toFixed(1));
    const drawRate = Number(((draws / totalActualGames) * 100).toFixed(1));
    const lossRate = Number(((losses / totalActualGames) * 100).toFixed(1));

    return {
      games: totalActualGames,
      wins,
      draws,
      losses,
      winsAsX,
      winsAsO,
      winRate,
      drawRate,
      lossRate,
      opponentPolicy
    };
  }

  /**
   * Backwards compatible evaluation wrapper.
   */
  evaluate(numGames = 100, agentPlayer = 'X', opponentPolicy = 'random') {
    return this.evaluateUnbiased(numGames, opponentPolicy);
  }

  /**
   * Asynchronous batch training with responsive UI yields and epsilon decay.
   */
  async trainAsync(totalEpisodes, opponentPolicy = 'random', agentPlayer = 'X', onProgress, onComplete) {
    if (this.isTraining) return;

    this.isTraining = true;
    this.shouldStopTraining = false;
    this.trainingState = 'training';
    // Snapshot initial exploration rate at start of this training run
    this.initialEpsilon = this.epsilon;

    const chunkSize = Math.max(25, Math.min(250, Math.floor(totalEpisodes / 30)));
    let episodesCompleted = 0;

    // Snapshot baseline evaluation before first training episode if not already recorded
    if (!this.baselineEval) {
      this.baselineEval = this.evaluateUnbiased(100, opponentPolicy);
      this.evaluationHistory.push({
        episode: 0,
        ...this.baselineEval
      });
      if (onProgress) {
        onProgress({
          current: 0,
          total: totalEpisodes,
          progressPct: 0,
          currentEpsilon: this.epsilon,
          visitedStates: this.getVisitedCount(),
          evalResult: this.baselineEval
        });
      }
    }

    const runChunk = () => {
      this.trainingTimerId = null;
      if (this.shouldStopTraining || episodesCompleted >= totalEpisodes) {
        this.isTraining = false;
        this.trainingState = 'idle';
        // Final evaluation
        this.postTrainingEval = this.evaluateUnbiased(100, opponentPolicy);
        this.evaluationHistory.push({
          episode: this.totalEpisodesTrained,
          ...this.postTrainingEval
        });
        if (onComplete) {
          onComplete({
            finalEval: this.postTrainingEval,
            stoppedEarly: this.shouldStopTraining
          });
        }
        return;
      }

      const currentBatch = Math.min(chunkSize, totalEpisodes - episodesCompleted);
      for (let i = 0; i < currentBatch; i++) {
        // Epsilon decay update
        if (this.decayEpsilon) {
          const progressRatio = Math.min(1.0, (episodesCompleted + i) / totalEpisodes);
          this.epsilon = Math.max(
            this.minEpsilon,
            this.initialEpsilon - progressRatio * (this.initialEpsilon - this.minEpsilon)
          );
        }

        // Randomly alternate between training as X and as O for balanced learning
        const activeRole = (i % 2 === 0) ? 'X' : 'O';
        this.trainEpisode(activeRole, opponentPolicy);
      }
      episodesCompleted += currentBatch;

      // Periodic evaluation checkpoint
      let evalResult = null;
      if (episodesCompleted % Math.max(chunkSize * 2, 500) === 0 || episodesCompleted === totalEpisodes) {
        evalResult = this.evaluateUnbiased(50, opponentPolicy);
        this.evaluationHistory.push({
          episode: this.totalEpisodesTrained,
          ...evalResult
        });
      }

      if (onProgress) {
        onProgress({
          current: episodesCompleted,
          total: totalEpisodes,
          progressPct: Math.round((episodesCompleted / totalEpisodes) * 100),
          currentEpsilon: Number(this.epsilon.toFixed(3)),
          visitedStates: this.getVisitedCount(),
          evalResult
        });
      }

      this.trainingTimerId = setTimeout(runChunk, 4);
    };

    runChunk();
  }

  stopTraining() {
    if (this.isTraining) {
      this.shouldStopTraining = true;
      this.trainingState = 'stopping';
      if (this.trainingTimerId !== null) {
        clearTimeout(this.trainingTimerId);
        this.trainingTimerId = null;
      }
    }
  }

  /**
   * Fast load pre-trained policy for lecture demonstrations.
   * Trains 3,500 balanced episodes.
   */
  loadMasterPolicy(agentPlayer = 'X') {
    this.resetPolicy();
    this.decayEpsilon = true;
    this.epsilon = 0.5;

    for (let i = 0; i < 2000; i++) {
      const role = i % 2 === 0 ? 'X' : 'O';
      this.trainEpisode(role, 'random');
    }
    for (let i = 0; i < 1500; i++) {
      const role = i % 2 === 0 ? 'X' : 'O';
      this.trainEpisode(role, 'heuristic');
    }

    this.epsilon = this.minEpsilon;
    this.postTrainingEval = this.evaluateUnbiased(100, 'random');
    this.evaluationHistory.push({
      episode: this.totalEpisodesTrained,
      ...this.postTrainingEval
    });
    return this.postTrainingEval;
  }

  exportPolicyJson() {
    const tableObj = {};
    for (const [k, v] of this.qTable.entries()) {
      tableObj[k] = v;
    }
    return JSON.stringify({
      version: '2.1',
      totalEpisodesTrained: this.totalEpisodesTrained,
      visitedCount: this.visitedActions.size,
      visitedActions: Array.from(this.visitedActions),
      stats: this.trainingStats,
      baselineEval: this.baselineEval,
      postTrainingEval: this.postTrainingEval,
      evaluationHistory: this.evaluationHistory,
      qTable: tableObj
    }, null, 2);
  }

  importPolicyJson(jsonStr) {
    if (this.isTraining) {
      console.warn('Cannot import policy while training is active.');
      return false;
    }
    try {
      if (!jsonStr || typeof jsonStr !== 'string') return false;
      const data = JSON.parse(jsonStr);
      if (!data || typeof data !== 'object') return false;
      if (!data.qTable || typeof data.qTable !== 'object' || Array.isArray(data.qTable)) return false;

      // Validate entries
      for (const [k, v] of Object.entries(data.qTable)) {
        if (!Array.isArray(v) || v.length !== 9 || !v.every(n => typeof n === 'number' && !isNaN(n))) {
          return false;
        }
      }

      this.qTable.clear();
      this.visitedActions.clear();

      for (const [k, v] of Object.entries(data.qTable)) {
        this.qTable.set(k, [...v]);
      }

      if (Array.isArray(data.visitedActions)) {
        for (const va of data.visitedActions) {
          if (typeof va === 'string') {
            this.visitedActions.add(va);
          }
        }
      } else {
        // Fallback for older v2.0 exports
        for (const [k, v] of Object.entries(data.qTable)) {
          for (let a = 0; a < 9; a++) {
            if (v[a] !== 0) {
              this.visitedActions.add(`${k}:${a}`);
            }
          }
        }
      }

      this.totalEpisodesTrained = typeof data.totalEpisodesTrained === 'number' ? data.totalEpisodesTrained : 0;
      this.trainingStats = data.stats && typeof data.stats === 'object'
        ? { ...data.stats }
        : { wins: 0, draws: 0, losses: 0, episodes: 0 };
      this.baselineEval = data.baselineEval || null;
      this.postTrainingEval = data.postTrainingEval || null;
      this.evaluationHistory = Array.isArray(data.evaluationHistory) ? [...data.evaluationHistory] : [];
      return true;
    } catch (e) {
      console.error('Failed to import policy:', e);
      return false;
    }
  }
}

// Export for module and window environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { QLearningAgent };
} else {
  window.QLearningAgent = QLearningAgent;
}
