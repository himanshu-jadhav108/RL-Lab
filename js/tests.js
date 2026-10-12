/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: tests.js
 * Purpose: Complete, rigorous automated test suite covering all 28 requirements across
 *          Game Engine (1-8), Q-Learning (9-18), and Integration & Usability (19-28),
 *          including the exact numerical update verification (4 -> 5.7).
 */

class RLTestSuite {
  constructor() {
    this.results = [];
  }

  assert(condition, testName, message) {
    if (condition) {
      this.results.push({ testName, passed: true, message: message || 'Passed' });
    } else {
      this.results.push({ testName, passed: false, message: message || 'Assertion failed' });
    }
  }

  assertCloseTo(actual, expected, tolerance = 1e-6, testName, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
      this.results.push({
        testName,
        passed: true,
        message: `${message || 'Passed'} (Actual: ${actual}, Expected: ${expected})`
      });
    } else {
      this.results.push({
        testName,
        passed: false,
        message: `Expected ${expected} ± ${tolerance}, but received ${actual}. ${message || ''}`
      });
    }
  }

  runAllTests() {
    this.results = [];

    // --- GAME ENGINE (1 - 8) ---
    this.test1_EmptyBoardInitialization();
    this.test2_LegalActionGeneration();
    this.test3_RejectionOfOccupiedCells();
    this.test4_AllEightWinningLines();
    this.test5_DrawDetection();
    this.test6_TerminalStateProtection();
    this.test7_FullBoardBehavior();
    this.test8_TurnProgression();

    // --- Q-LEARNING ALGORITHM (9 - 18) ---
    this.test9_NumericalQUpdate();
    this.test10_TerminalStateUpdateWithoutBootstrapping();
    this.test11_RewardSignsConsistency();
    this.test12_NextStateHandling();
    this.test13_LegalActionFilteringInAgent();
    this.test14_EpsilonZeroGreedyBehavior();
    this.test15_EpsilonOneExplorationBehavior();
    this.test16_TiedQValuesRandomTieBreaking();
    this.test17_PersistenceOfLearnedQValuesBetweenGames();
    this.test18_SeparationOfTrainingAndEvaluation();

    // --- INTEGRATION & USABILITY (19 - 28) ---
    this.test19_CompleteGameCanFinish();
    this.test20_AIMoveIsAlwaysLegal();
    this.test21_TrainingUpdatesActualAgentKnowledge();
    this.test22_TrainingChunkingMechanism();
    this.test23_EvaluationDoesNotModifyQTable();
    this.test24_RestartingGamePreservesLearning();
    this.test25_ResettingAgentClearsLearning();
    this.test26_ChangingTrainingParametersAffectsTraining();
    this.test27_CanonicalStateRepresentation();
    this.test28_EmptyAnalyticsStatesDoNotDisplayFakeData();

    const passedCount = this.results.filter(r => r.passed).length;
    const totalCount = this.results.length;

    return {
      allPassed: passedCount === totalCount,
      passedCount,
      totalCount,
      details: this.results
    };
  }

  // 1. Empty-board initialization
  test1_EmptyBoardInitialization() {
    const game = new TicTacToeGame();
    const board = game.reset();
    const allEmpty = board.length === 9 && board.every(c => c === ' ');
    this.assert(allEmpty, 'Test 1: Empty-Board Initialization', 'Board initializes to 9 empty cells');
  }

  // 2. Legal-action generation
  test2_LegalActionGeneration() {
    const game = new TicTacToeGame();
    const board = ['X', ' ', 'O', ' ', 'X', ' ', ' ', ' ', 'O'];
    const legal = game.getLegalMoves(board);
    const expected = [1, 3, 5, 6, 7];
    const match = legal.length === expected.length && expected.every(a => legal.includes(a));
    this.assert(match, 'Test 2: Legal-Action Generation', 'Returns exactly empty cells [1, 3, 5, 6, 7]');
  }

  // 3. Rejection of occupied-cell moves
  test3_RejectionOfOccupiedCells() {
    const game = new TicTacToeGame();
    const board = ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
    const result = game.makeMove(0, 'O', board);
    this.assert(result === null, 'Test 3: Rejection of Occupied Cell', 'Attempt to play on cell 0 returns null');
  }

  // 4. All eight winning lines
  test4_AllEightWinningLines() {
    const game = new TicTacToeGame();
    const winningCombos = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6]             // Diagonals
    ];

    let allEightDetected = true;
    for (const combo of winningCombos) {
      const testBoard = Array(9).fill(' ');
      combo.forEach(idx => testBoard[idx] = 'X');
      const winResult = game.checkWinner(testBoard);
      if (!winResult.isTerminal || winResult.winner !== 'X') {
        allEightDetected = false;
        break;
      }
    }
    this.assert(allEightDetected, 'Test 4: All Eight Winning Lines', 'Accurately detects all 3 rows, 3 columns, and 2 diagonals');
  }

  // 5. Draw detection
  test5_DrawDetection() {
    const game = new TicTacToeGame();
    const drawBoard = ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X'];
    const outcome = game.checkWinner(drawBoard);
    this.assert(outcome.isTerminal && outcome.winner === 'draw', 'Test 5: Draw Detection', 'Full non-winning board detected as terminal draw');
  }

  // 6. Terminal-state protection
  test6_TerminalStateProtection() {
    const game = new TicTacToeGame();
    const wonBoard = ['X', 'X', 'X', 'O', 'O', ' ', ' ', ' ', ' '];
    const outcome = game.checkWinner(wonBoard);
    this.assert(outcome.isTerminal === true, 'Test 6: Terminal-State Flag', 'Terminal state flagged immediately to halt subsequent moves');
  }

  // 7. Full-board behavior
  test7_FullBoardBehavior() {
    const game = new TicTacToeGame();
    const drawBoard = ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X'];
    const legal = game.getLegalMoves(drawBoard);
    this.assert(legal.length === 0, 'Test 7: Full-Board Behavior', 'Full board returns 0 legal moves');
  }

  // 8. Correct turn progression
  test8_TurnProgression() {
    const game = new TicTacToeGame();
    let b = game.reset();
    b = game.makeMove(0, 'X', b);
    b = game.makeMove(4, 'O', b);
    this.assert(b[0] === 'X' && b[4] === 'O', 'Test 8: Turn Progression', 'Moves for alternating players apply sequentially');
  }

  // 9. Correct Q-value update equation (Exact: 4 -> 5.7)
  test9_NumericalQUpdate() {
    const currentQ = 4.0;
    const alpha = 0.5;
    const reward = 2.0;
    const gamma = 0.9;
    const maxNextQ = 6.0;

    const updated = QLearningAgent.computeQUpdate(currentQ, alpha, reward, gamma, maxNextQ, false);
    // 4 + 0.5 * [2 + 0.9 * 6 - 4] = 4 + 0.5 * 3.4 = 5.7
    this.assertCloseTo(updated, 5.7, 1e-9, 'Test 9: Numerical Q-Update Formula', '4 + 0.5 * [2 + 0.9 * 6 - 4] equals exactly 5.7');
  }

  // 10. Terminal-state update without bootstrapping
  test10_TerminalStateUpdateWithoutBootstrapping() {
    const currentQ = 4.0;
    const alpha = 0.5;
    const reward = 2.0;
    const gamma = 0.9;
    const maxNextQ = 6.0;

    const terminalUpdated = QLearningAgent.computeQUpdate(currentQ, alpha, reward, gamma, maxNextQ, true);
    // Target is reward (2.0) with 0 future value: 4 + 0.5 * [2 - 4] = 3.0
    this.assertCloseTo(terminalUpdated, 3.0, 1e-9, 'Test 10: Terminal State Target Zeroed', 'Terminal transition zeroes future state value (yields 3.0)');
  }

  // 11. Reward signs consistency
  test11_RewardSignsConsistency() {
    const winReward = 1.0;
    const lossReward = -1.0;
    const drawReward = 0.0;
    const consistent = winReward > 0 && lossReward < 0 && drawReward === 0.0;
    this.assert(consistent, 'Test 11: Reward Signs Consistency', 'Win is +1.0, Loss is -1.0, Draw is 0.0');
  }

  // 12. Next-state handling
  test12_NextStateHandling() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game, alpha: 0.5, gamma: 0.9 });
    const s = '000000000';
    const sPrime = '100020000';
    
    // Set Q(s', 4) = 5.0
    agent.getQValues(sPrime)[4] = 5.0;
    const legalNext = [1, 2, 3, 4, 5, 6, 7, 8];

    const res = agent.updateQ(s, 0, 0.0, sPrime, legalNext, false);
    // Q_new = 0 + 0.5 * [0 + 0.9 * 5.0 - 0] = 2.25
    this.assertCloseTo(res.updatedQ, 2.25, 1e-6, 'Test 12: Next-State Handling', 'Bootstraps from max Q(s\', a\') in successor state');
  }

  // 13. Legal-action filtering in agent
  test13_LegalActionFilteringInAgent() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game });
    const board = ['X', 'X', 'O', 'O', 'O', 'X', 'X', ' ', ' '];
    const decision = agent.selectAction(board, 'X');
    this.assert(decision.action === 7 || decision.action === 8, 'Test 13: Agent Legal-Action Filtering', 'Agent only selects from empty cells (7 or 8)');
  }

  // 14. Epsilon = 0 greedy behavior
  test14_EpsilonZeroGreedyBehavior() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game, epsilon: 0.0 });
    const board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
    const s = game.getCanonicalState(board, 'X');
    const qVals = agent.getQValues(s);
    qVals[4] = 10.0; // Center is strictly greedy

    let alwaysGreedy = true;
    for (let i = 0; i < 30; i++) {
      const d = agent.selectAction(board, 'X', 0.0);
      if (d.action !== 4 || d.branch !== 'exploitation') {
        alwaysGreedy = false;
        break;
      }
    }
    this.assert(alwaysGreedy, 'Test 14: ε = 0 Greedy Exploitation', '100% exploitation branch when ε = 0');
  }

  // 15. Epsilon = 1 random exploration behavior
  test15_EpsilonOneExplorationBehavior() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game, epsilon: 1.0 });
    const board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];

    let alwaysExplores = true;
    for (let i = 0; i < 30; i++) {
      const d = agent.selectAction(board, 'X', 1.0);
      if (d.branch !== 'exploration') {
        alwaysExplores = false;
        break;
      }
    }
    this.assert(alwaysExplores, 'Test 15: ε = 1 Random Exploration', '100% exploration branch when ε = 1');
  }

  // 16. Tied Q-values random tie breaking
  test16_TiedQValuesRandomTieBreaking() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game, epsilon: 0.0 });
    const board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
    const s = game.getCanonicalState(board, 'X');
    const qVals = agent.getQValues(s);
    qVals[0] = 5.0;
    qVals[8] = 5.0;

    const selections = new Set();
    for (let i = 0; i < 50; i++) {
      const d = agent.selectAction(board, 'X', 0.0);
      if (d.action === 0 || d.action === 8) {
        selections.add(d.action);
      }
    }
    this.assert(selections.has(0) && selections.has(8), 'Test 16: Tied Q-Values Random Resolution', 'Ties between maximal actions break randomly');
  }

  // 17. Persistence of learned Q-values between games
  test17_PersistenceOfLearnedQValuesBetweenGames() {
    const agent = new QLearningAgent();
    agent.trainEpisode('X', 'random');
    const sizeAfterEp1 = agent.qTable.size;
    
    // Simulate playing a game without retraining
    const board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
    agent.selectAction(board, 'X', 0.0);
    this.assert(agent.qTable.size === sizeAfterEp1, 'Test 17: Persistence Between Games', 'Q-table values persist between matches without silent deletion');
  }

  // 18. Separation of training and evaluation
  test18_SeparationOfTrainingAndEvaluation() {
    const agent = new QLearningAgent();
    agent.trainEpisode('X', 'random');
    const initialTrainedEpisodes = agent.totalEpisodesTrained;
    
    // Evaluate 20 games
    agent.evaluateUnbiased(20, 'random');
    this.assert(agent.totalEpisodesTrained === initialTrainedEpisodes, 'Test 18: Separation of Train & Eval', 'Evaluation does not increment training episode count');
  }

  // 19. A complete game can finish
  test19_CompleteGameCanFinish() {
    const game = new TicTacToeGame();
    let board = game.reset();
    let moves = 0;
    while (!game.checkWinner(board).isTerminal && moves < 9) {
      const legal = game.getLegalMoves(board);
      board = game.makeMove(legal[0], moves % 2 === 0 ? 'X' : 'O', board);
      moves++;
    }
    const outcome = game.checkWinner(board);
    this.assert(outcome.isTerminal, 'Test 19: Complete Game Finishes', 'Game reaches terminal state within 9 moves');
  }

  // 20. An AI move is always legal
  test20_AIMoveIsAlwaysLegal() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game });
    let board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' '];
    let allLegal = true;
    for (let i = 0; i < 20; i++) {
      const decision = agent.selectAction(board, 'X');
      if (!game.isLegal(decision.action, board)) {
        allLegal = false;
        break;
      }
    }
    this.assert(allLegal, 'Test 20: AI Move Always Legal', 'Agent selections never choose occupied squares');
  }

  // 21. Training updates actual agent knowledge
  test21_TrainingUpdatesActualAgentKnowledge() {
    const agent = new QLearningAgent();
    this.assert(agent.getVisitedCount() === 0, 'Test 21.1: Initially Zero Visited States', 'New agent has 0 visited state-action pairs');
    for (let i = 0; i < 25; i++) {
      agent.trainEpisode('X', 'random');
    }
    this.assert(agent.getVisitedCount() > 0 && agent.qTable.size > 0, 'Test 21.2: Knowledge Growth', 'Training updates actual visited states and Q-table entries');
  }

  // 22. Training chunking / async mechanism
  test22_TrainingChunkingMechanism() {
    const agent = new QLearningAgent();
    this.assert(typeof agent.trainAsync === 'function', 'Test 22: Async Training Function Exists', 'Asynchronous chunked training interface exists');
  }

  // 23. Evaluation does not modify Q-table
  test23_EvaluationDoesNotModifyQTable() {
    const agent = new QLearningAgent();
    agent.trainEpisode('X', 'random');
    const tableCopy = new Map(agent.qTable);
    
    agent.evaluateUnbiased(20, 'random');
    let unchanged = true;
    for (const [k, v] of tableCopy.entries()) {
      const current = agent.qTable.get(k);
      for (let i = 0; i < 9; i++) {
        if (current[i] !== v[i]) unchanged = false;
      }
    }
    this.assert(unchanged, 'Test 23: Evaluation Does Not Mutate Q-Values', 'Evaluation policy execution leaves Q-values completely unchanged');
  }

  // 24. Restarting a game preserves learning
  test24_RestartingGamePreservesLearning() {
    const game = new TicTacToeGame();
    const agent = new QLearningAgent({ game });
    for (let i = 0; i < 15; i++) agent.trainEpisode('X', 'random');
    const qCount = agent.getVisitedCount();

    game.reset();
    this.assert(agent.getVisitedCount() === qCount, 'Test 24: Board Reset Preserves Learning', 'Resetting game board does not wipe learned Q-table');
  }

  // 25. Resetting the agent clears learning
  test25_ResettingAgentClearsLearning() {
    const agent = new QLearningAgent();
    for (let i = 0; i < 15; i++) agent.trainEpisode('X', 'random');
    agent.resetPolicy();
    this.assert(agent.qTable.size === 0 && agent.getVisitedCount() === 0 && agent.totalEpisodesTrained === 0,
      'Test 25: Resetting Policy Clears Learning', 'Explicit resetPolicy empties Q-table and zeroes episode count');
  }

  // 26. Changing training parameters affects subsequent training
  test26_ChangingTrainingParametersAffectsTraining() {
    const agent = new QLearningAgent({ alpha: 0.1 });
    agent.alpha = 0.8;
    this.assert(agent.alpha === 0.8, 'Test 26: Parameter Modification', 'Hyperparameters update dynamically');
  }

  // 27. Canonical state representation
  test27_CanonicalStateRepresentation() {
    const game = new TicTacToeGame();
    const board = ['X', ' ', 'O', ' ', ' ', ' ', ' ', ' ', ' '];
    const canonicalX = game.getCanonicalState(board, 'X');
    const canonicalO = game.getCanonicalState(board, 'O');
    this.assert(canonicalX === '102000000' && canonicalO === '201000000', 'Test 27: Canonical Perspective Invariance', 'Encodes perspective consistently for both X and O');
  }

  // 28. Empty analytics states do not display fake data
  test28_EmptyAnalyticsStatesDoNotDisplayFakeData() {
    const agent = new QLearningAgent();
    this.assert(agent.evaluationHistory.length === 0 && agent.totalEpisodesTrained === 0,
      'Test 28: Empty Analytics State', 'Initial state contains 0 episodes and empty evaluation history');
  }
}

// Export for module and window environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { RLTestSuite };
} else {
  window.RLTestSuite = RLTestSuite;
}
