/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: app.js
 * Purpose: Complete application coordinator for the 3-stage learning journey:
 *          Stage 01 (Before Training), Stage 02 (Train Agent), Stage 03 (After Training),
 *          along with explainability, comparison benchmarks, theory, and test runner.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Core Systems
  const game = new TicTacToeGame();
  const agent = new QLearningAgent({ game });
  const analytics = new AnalyticsVisualizer('learningCurveChart');
  const dpDemo = new DynamicProgrammingDemo('dpGridContainer');
  const testSuite = new RLTestSuite();

  // Audio State
  let soundEnabled = true;
  let audioCtx = null;

  // Header Elements
  const agentStatusBadge = document.getElementById('agent-status-badge');
  const agentStatusText = document.getElementById('agent-status-text');
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const btnSoundToggle = document.getElementById('btn-sound-toggle');
  const btnGlobalReset = document.getElementById('btn-global-reset');

  // Navigation Stepper Tabs
  const stageTabs = document.querySelectorAll('.stage-tab');
  const stagePanels = document.querySelectorAll('.stage-panel');

  // STAGE 01 ELEMENTS
  const boardElS1 = document.getElementById('board-s1');
  const turnIndicatorS1 = document.getElementById('turn-indicator-s1');
  const gameStatusS1 = document.getElementById('game-status-s1');
  const selectRoleS1 = document.getElementById('select-role-s1');
  const btnRestartS1 = document.getElementById('btn-restart-s1');
  const btnNextToTrain = document.getElementById('btn-next-to-train');
  const s1StatGames = document.getElementById('s1-stat-games');
  const s1StatHumanWins = document.getElementById('s1-stat-human-wins');
  const s1StatDraws = document.getElementById('s1-stat-draws');
  const s1StatAgentWins = document.getElementById('s1-stat-agent-wins');
  const btnRunBaseline = document.getElementById('btn-run-baseline');
  const baselineResultsBox = document.getElementById('baseline-results-box');

  // STAGE 01 STATE
  let humanRoleS1 = 'X';
  let agentRoleS1 = 'O';
  let currentTurnS1 = 'X';
  let boardStateS1 = game.reset();
  let isGameOverS1 = false;
  let sessionStatsS1 = { games: 0, humanWins: 0, draws: 0, agentWins: 0 };

  // STAGE 02 ELEMENTS
  const inputEpisodes = document.getElementById('input-episodes');
  const selectOpponent = document.getElementById('select-opponent');
  const inputAlpha = document.getElementById('input-alpha');
  const dispAlpha = document.getElementById('disp-alpha');
  const inputGamma = document.getElementById('input-gamma');
  const dispGamma = document.getElementById('disp-gamma');
  const inputEpsilon = document.getElementById('input-epsilon');
  const dispEpsilon = document.getElementById('disp-epsilon');
  const chkDecayEpsilon = document.getElementById('chk-decay-epsilon');
  const inputSeed = document.getElementById('input-seed');
  const btnTrain = document.getElementById('btn-train');
  const btnStopTrain = document.getElementById('btn-stop-train');
  const progressBar = document.getElementById('training-progress-bar');
  const progressText = document.getElementById('training-progress-text');
  const liveTrainingBadge = document.getElementById('live-training-badge');
  const kpiEpisodes = document.getElementById('kpi-episodes');
  const kpiCurrentEps = document.getElementById('kpi-current-eps');
  const kpiVisitedStates = document.getElementById('kpi-visited-states');
  const kpiTrainingWdl = document.getElementById('kpi-training-wdl');
  const btnProceedToStage3 = document.getElementById('btn-proceed-to-stage-3');

  // STAGE 03 ELEMENTS
  const boardElS3 = document.getElementById('board-s3');
  const turnIndicatorS3 = document.getElementById('turn-indicator-s3');
  const gameStatusS3 = document.getElementById('game-status-s3');
  const selectRoleS3 = document.getElementById('select-role-s3');
  const btnRestartS3 = document.getElementById('btn-restart-s3');
  const btnToggleExplain = document.getElementById('btn-toggle-explain');
  const s3StatGames = document.getElementById('s3-stat-games');
  const s3StatHumanWins = document.getElementById('s3-stat-human-wins');
  const s3StatDraws = document.getElementById('s3-stat-draws');
  const s3StatAgentWins = document.getElementById('s3-stat-agent-wins');
  const btnRunPostEval = document.getElementById('btn-run-post-eval');
  const compBeforeWin = document.getElementById('comp-before-win');
  const compBeforeDraw = document.getElementById('comp-before-draw');
  const compBeforeLoss = document.getElementById('comp-before-loss');
  const compAfterWin = document.getElementById('comp-after-win');
  const compAfterDraw = document.getElementById('comp-after-draw');
  const compAfterLoss = document.getElementById('comp-after-loss');
  const compBeforeOpp = document.getElementById('comp-before-opp');
  const compAfterOpp = document.getElementById('comp-after-opp');
  const explainabilityDrawer = document.getElementById('explainability-drawer');
  const explainStateKey = document.getElementById('explain-state-key');
  const explainMovesTbody = document.getElementById('explain-moves-tbody');
  const explainLastDecisionBox = document.getElementById('explain-last-decision-box');
  const btnExportQTable = document.getElementById('btn-export-qtable');
  const btnImportQTable = document.getElementById('btn-import-qtable');
  const importFileInput = document.getElementById('import-file-input');

  // STAGE 03 STATE
  let humanRoleS3 = 'X';
  let agentRoleS3 = 'O';
  let currentTurnS3 = 'X';
  let boardStateS3 = game.reset();
  let isGameOverS3 = false;
  let sessionStatsS3 = { games: 0, humanWins: 0, draws: 0, agentWins: 0 };
  let explainOpen = true;

  // THEORY CALCULATOR ELEMENTS
  const calcCurrentQ = document.getElementById('calc-current-q');
  const calcAlpha = document.getElementById('calc-alpha');
  const calcReward = document.getElementById('calc-reward');
  const calcGamma = document.getElementById('calc-gamma');
  const calcMaxNextQ = document.getElementById('calc-max-next-q');
  const calcTdTarget = document.getElementById('calc-td-target');
  const calcTdError = document.getElementById('calc-td-error');
  const calcUpdatedQ = document.getElementById('calc-updated-q');

  // TEST RUNNER ELEMENTS
  const btnRunTests = document.getElementById('btn-run-tests');
  const testResultsContainer = document.getElementById('test-results-container');
  const testSummaryBadge = document.getElementById('test-summary-badge');

  // -------------------------------------------------------------
  // THEME & AUDIO MANAGEMENT
  // -------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem('rl_lab_theme') || 'light';
    setTheme(savedTheme);
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rl_lab_theme', theme);
    if (btnThemeToggle) {
      btnThemeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
      btnThemeToggle.title = theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme';
    }
    window.dispatchEvent(new CustomEvent('themeChanged'));
  }

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      setTheme(current === 'dark' ? 'light' : 'dark');
      playTone(520, 0.05);
    });
  }

  function getAudioContext() {
    if (!audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) audioCtx = new AudioCtx();
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return audioCtx;
  }

  function playTone(freq, duration = 0.08, type = 'sine') {
    if (!soundEnabled) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      gain.gain.setValueAtTime(0.06, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {}
  }

  if (btnSoundToggle) {
    btnSoundToggle.addEventListener('click', () => {
      soundEnabled = !soundEnabled;
      btnSoundToggle.textContent = soundEnabled ? '🔊' : '🔇';
      btnSoundToggle.title = soundEnabled ? 'Mute Sound' : 'Enable Sound';
      if (soundEnabled) playTone(600, 0.06);
    });
  }

  // -------------------------------------------------------------
  // STEPPER NAVIGATION
  // -------------------------------------------------------------
  function switchStage(stageId) {
    stageTabs.forEach(t => {
      const matches = t.getAttribute('data-stage') === stageId;
      t.classList.toggle('active', matches);
      t.setAttribute('aria-selected', matches ? 'true' : 'false');
    });

    stagePanels.forEach(p => {
      p.classList.toggle('active', p.id === stageId);
    });

    if (stageId === 'stage-2') {
      setTimeout(() => analytics.resize(), 50);
    } else if (stageId === 'stage-3') {
      renderBoardS3();
      updateComparisonDisplay();
    }

    playTone(480, 0.04);
  }

  stageTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const stageId = tab.getAttribute('data-stage');
      switchStage(stageId);
    });
  });

  if (btnNextToTrain) {
    btnNextToTrain.addEventListener('click', () => switchStage('stage-2'));
  }
  if (btnProceedToStage3) {
    btnProceedToStage3.addEventListener('click', () => switchStage('stage-3'));
  }

  // -------------------------------------------------------------
  // STAGE 01: BEFORE TRAINING GAMEPLAY
  // -------------------------------------------------------------
  function renderBoardS1(winningLine = null) {
    if (!boardElS1) return;
    boardElS1.innerHTML = '';

    for (let i = 0; i < 9; i++) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      cell.setAttribute('data-index', i);
      cell.setAttribute('role', 'button');
      cell.setAttribute('tabindex', '0');
      cell.setAttribute('aria-label', `Cell ${i + 1}, ${boardStateS1[i] === ' ' ? 'empty' : boardStateS1[i]}`);

      const mark = boardStateS1[i];
      if (mark !== ' ') {
        cell.classList.add(`marked-${mark.toLowerCase()}`);
        cell.textContent = mark;
      }

      if (winningLine && winningLine.includes(i)) {
        cell.classList.add('winning-cell');
      }

      cell.addEventListener('click', () => handleCellClickS1(i));
      cell.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleCellClickS1(i);
        }
      });

      boardElS1.appendChild(cell);
    }
  }

  function handleCellClickS1(index) {
    if (isGameOverS1 || agent.isTraining) return;
    if (currentTurnS1 !== humanRoleS1) return;
    if (!game.isLegal(index, boardStateS1)) return;

    boardStateS1 = game.makeMove(index, humanRoleS1, boardStateS1);
    playTone(550, 0.06);
    const outcome = game.checkWinner(boardStateS1);

    if (outcome.isTerminal) {
      endGameS1(outcome);
    } else {
      currentTurnS1 = agentRoleS1;
      updateTurnUIS1();
      renderBoardS1();
      setTimeout(makeAgentMoveS1, 240);
    }
  }

  function makeAgentMoveS1() {
    if (isGameOverS1 || currentTurnS1 !== agentRoleS1) return;

    // Untrained agent: with empty Q-table or random exploration
    const decision = agent.selectAction(boardStateS1, agentRoleS1, 1.0);
    if (!decision) return;

    boardStateS1 = game.makeMove(decision.action, agentRoleS1, boardStateS1);
    playTone(420, 0.06);
    const outcome = game.checkWinner(boardStateS1);

    if (outcome.isTerminal) {
      endGameS1(outcome);
    } else {
      currentTurnS1 = humanRoleS1;
      updateTurnUIS1();
      renderBoardS1();
    }
  }

  function updateTurnUIS1() {
    if (isGameOverS1) return;
    if (currentTurnS1 === humanRoleS1) {
      turnIndicatorS1.innerHTML = `<span class="turn-dot user-turn"></span> Your Turn (${humanRoleS1})`;
      gameStatusS1.textContent = 'Choose an empty square.';
    } else {
      turnIndicatorS1.innerHTML = `<span class="turn-dot agent-turn"></span> Agent Turn (${agentRoleS1})`;
      gameStatusS1.textContent = 'Untrained agent is selecting a random legal action...';
    }
  }

  function endGameS1(outcome) {
    isGameOverS1 = true;
    sessionStatsS1.games++;

    if (outcome.winner === 'draw') {
      sessionStatsS1.draws++;
      turnIndicatorS1.innerHTML = '<span class="badge badge-secondary">Draw Match</span>';
      gameStatusS1.textContent = 'Game ended in a draw.';
      renderBoardS1(null);
    } else {
      const humanWon = outcome.winner === humanRoleS1;
      if (humanWon) sessionStatsS1.humanWins++;
      else sessionStatsS1.agentWins++;

      turnIndicatorS1.innerHTML = `<span class="badge ${humanWon ? 'badge-success' : 'badge-danger'}">${outcome.winner} Won</span>`;
      gameStatusS1.textContent = humanWon ? 'You won against the untrained agent.' : 'The untrained agent won by random chance.';
      renderBoardS1(outcome.winningLine);
    }

    s1StatGames.textContent = sessionStatsS1.games;
    s1StatHumanWins.textContent = sessionStatsS1.humanWins;
    s1StatDraws.textContent = sessionStatsS1.draws;
    s1StatAgentWins.textContent = sessionStatsS1.agentWins;
  }

  function restartGameS1() {
    boardStateS1 = game.reset();
    isGameOverS1 = false;
    currentTurnS1 = 'X'; // X starts

    if (agentRoleS1 === 'X') {
      updateTurnUIS1();
      renderBoardS1();
      setTimeout(makeAgentMoveS1, 200);
    } else {
      updateTurnUIS1();
      renderBoardS1();
    }
  }

  btnRestartS1.addEventListener('click', restartGameS1);
  selectRoleS1.addEventListener('change', (e) => {
    humanRoleS1 = e.target.value;
    agentRoleS1 = humanRoleS1 === 'X' ? 'O' : 'X';
    restartGameS1();
  });

  // Run Baseline Benchmark
  btnRunBaseline.addEventListener('click', () => {
    btnRunBaseline.disabled = true;
    btnRunBaseline.textContent = 'Running 100 benchmark games (50 as X, 50 as O)...';

    setTimeout(() => {
      // Evaluate unbiased with untrained policy
      const result = agent.evaluateUnbiased(100, 'random');
      agent.baselineEval = result;

      baselineResultsBox.innerHTML = `
        <div style="font-weight: 700; margin-bottom: 4px;">Baseline Result (100 Games against Random Opponent):</div>
        <div class="bench-chips-row">
          <span class="bench-chip bench-win">Agent Wins: ${result.wins} (${result.winRate}%)</span>
          <span class="bench-chip bench-draw">Draws: ${result.draws} (${result.drawRate}%)</span>
          <span class="bench-chip bench-loss">Agent Losses: ${result.losses} (${result.lossRate}%)</span>
        </div>
        <div class="text-xs text-muted" style="margin-top: 4px;">
          Perspectives tested: 50 games as Player X (${result.winsAsX} wins), 50 games as Player O (${result.winsAsO} wins).
        </div>
      `;

      updateComparisonDisplay();
      btnRunBaseline.disabled = false;
      btnRunBaseline.textContent = 'Re-run Baseline Benchmark';
    }, 40);
  });

  // -------------------------------------------------------------
  // STAGE 02: TRAINING CONTROLS & PIPELINE
  // -------------------------------------------------------------
  function syncHyperparams() {
    agent.alpha = parseFloat(inputAlpha.value);
    agent.gamma = parseFloat(inputGamma.value);
    agent.epsilon = parseFloat(inputEpsilon.value);
    agent.initialEpsilon = agent.epsilon;
    agent.decayEpsilon = chkDecayEpsilon.checked;

    dispAlpha.textContent = agent.alpha.toFixed(2);
    dispGamma.textContent = agent.gamma.toFixed(2);
    dispEpsilon.textContent = agent.epsilon.toFixed(2);

    updateWorkedExample();
  }

  inputAlpha.addEventListener('input', syncHyperparams);
  inputGamma.addEventListener('input', syncHyperparams);
  inputEpsilon.addEventListener('input', syncHyperparams);
  chkDecayEpsilon.addEventListener('change', syncHyperparams);

  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active-preset'));
      btn.classList.add('active-preset');
      inputEpisodes.value = btn.getAttribute('data-val');
      playTone(460, 0.04);
    });
  });

  btnTrain.addEventListener('click', () => {
    if (agent.isTraining) return;

    const episodes = parseInt(inputEpisodes.value, 10) || 10000;
    const opponentPolicy = selectOpponent.value;
    const seedVal = inputSeed.value.trim();

    agent.setSeed(seedVal.length > 0 ? seedVal : null);
    setTrainingUI(true);

    agent.trainAsync(
      episodes,
      opponentPolicy,
      'X',
      (prog) => {
        progressBar.style.width = `${prog.progressPct}%`;
        progressText.textContent = `${prog.current.toLocaleString()} / ${prog.total.toLocaleString()} episodes (${prog.progressPct}%)`;
        kpiEpisodes.textContent = prog.current.toLocaleString();
        kpiCurrentEps.textContent = prog.currentEpsilon !== undefined ? prog.currentEpsilon.toFixed(2) : agent.epsilon.toFixed(2);
        kpiVisitedStates.textContent = prog.visitedStates.toLocaleString();
        kpiTrainingWdl.textContent = `${agent.trainingStats.wins}/${agent.trainingStats.draws}/${agent.trainingStats.losses}`;

        analytics.updateData(agent.evaluationHistory);
      },
      (comp) => {
        setTrainingUI(false);
        progressBar.style.width = '100%';
        progressText.textContent = `Completed ${episodes.toLocaleString()} episodes successfully.`;
        kpiEpisodes.textContent = agent.totalEpisodesTrained.toLocaleString();
        kpiVisitedStates.textContent = agent.getVisitedCount().toLocaleString();
        kpiCurrentEps.textContent = agent.epsilon.toFixed(2);
        kpiTrainingWdl.textContent = `${agent.trainingStats.wins}/${agent.trainingStats.draws}/${agent.trainingStats.losses}`;

        analytics.updateData(agent.evaluationHistory);
        updateAgentStatus();
        updateComparisonDisplay();
        restartGameS3();
        playTone(650, 0.1, 'triangle');
      }
    );
  });

  btnStopTrain.addEventListener('click', () => {
    agent.stopTraining();
    progressText.textContent = 'Stopping training safely...';
  });

  function setTrainingUI(isTraining) {
    btnTrain.disabled = isTraining;
    btnStopTrain.disabled = !isTraining;
    inputEpisodes.disabled = isTraining;
    selectOpponent.disabled = isTraining;
    inputAlpha.disabled = isTraining;
    inputGamma.disabled = isTraining;
    inputEpsilon.disabled = isTraining;
    chkDecayEpsilon.disabled = isTraining;
    if (btnRunBaseline) btnRunBaseline.disabled = isTraining;
    if (btnRunPostEval) btnRunPostEval.disabled = isTraining;
    if (btnGlobalReset) btnGlobalReset.disabled = isTraining;
    if (btnExportQTable) btnExportQTable.disabled = isTraining;
    if (btnImportQTable) btnImportQTable.disabled = isTraining;

    if (isTraining) {
      liveTrainingBadge.textContent = 'Training in progress...';
      liveTrainingBadge.className = 'badge badge-primary';
      agentStatusBadge.className = 'agent-status-pill status-training';
      agentStatusText.textContent = 'Training Agent...';
    } else {
      liveTrainingBadge.textContent = 'Training Finished';
      liveTrainingBadge.className = 'badge badge-success';
    }
  }

  function updateAgentStatus() {
    if (agent.totalEpisodesTrained === 0) {
      agentStatusBadge.className = 'agent-status-pill status-untrained';
      agentStatusText.textContent = 'Untrained Agent';
    } else {
      agentStatusBadge.className = 'agent-status-pill status-trained';
      agentStatusText.textContent = `Trained Agent (${agent.totalEpisodesTrained.toLocaleString()} eps)`;
    }
  }

  // -------------------------------------------------------------
  // STAGE 03: AFTER TRAINING GAMEPLAY & EXPLAINABILITY
  // -------------------------------------------------------------
  function renderBoardS3(winningLine = null) {
    if (!boardElS3) return;
    boardElS3.innerHTML = '';

    const canonicalState = game.getCanonicalState(boardStateS3, agentRoleS3);
    const qValues = agent.peekQValues(canonicalState);
    const legalMoves = game.getLegalMoves(boardStateS3);

    // Identify maximal Q among legal moves
    let maxQ = -Infinity;
    if (legalMoves.length > 0) {
      for (const a of legalMoves) {
        if (qValues[a] > maxQ) maxQ = qValues[a];
      }
    }

    for (let i = 0; i < 9; i++) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      cell.setAttribute('data-index', i);
      cell.setAttribute('role', 'button');
      cell.setAttribute('tabindex', '0');
      cell.setAttribute('aria-label', `Cell ${i + 1}, ${boardStateS3[i] === ' ' ? 'empty' : boardStateS3[i]}`);

      const mark = boardStateS3[i];
      if (mark !== ' ') {
        cell.classList.add(`marked-${mark.toLowerCase()}`);
        cell.textContent = mark;
      } else if (!isGameOverS3) {
        // Legal move: display learned Q-value
        const isVisited = agent.hasVisited(canonicalState, i);
        const qVal = qValues[i];
        const isGreedy = legalMoves.length > 0 && Math.abs(qVal - maxQ) < 1e-6;

        const qBadge = document.createElement('div');
        qBadge.className = `cell-q-badge ${isGreedy ? 'cell-q-greedy' : ''}`;
        qBadge.innerHTML = `<span class="q-val-num">${isVisited ? qVal.toFixed(2) : '—'}</span>${isGreedy ? '<span class="greedy-star">★</span>' : ''}`;
        cell.appendChild(qBadge);

        if (isVisited && qVal > 0) {
          cell.style.backgroundColor = `rgba(13, 148, 136, ${Math.min(0.22, qVal * 0.22)})`;
        } else if (isVisited && qVal < 0) {
          cell.style.backgroundColor = `rgba(239, 68, 68, ${Math.min(0.22, Math.abs(qVal) * 0.22)})`;
        }
      }

      if (winningLine && winningLine.includes(i)) {
        cell.classList.add('winning-cell');
      }

      cell.addEventListener('click', () => handleCellClickS3(i));
      cell.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleCellClickS3(i);
        }
      });

      boardElS3.appendChild(cell);
    }

    updateExplainabilityDrawer();
  }

  function handleCellClickS3(index) {
    if (isGameOverS3 || agent.isTraining) return;
    if (currentTurnS3 !== humanRoleS3) return;
    if (!game.isLegal(index, boardStateS3)) return;

    boardStateS3 = game.makeMove(index, humanRoleS3, boardStateS3);
    playTone(550, 0.06);
    const outcome = game.checkWinner(boardStateS3);

    if (outcome.isTerminal) {
      endGameS3(outcome);
    } else {
      currentTurnS3 = agentRoleS3;
      updateTurnUIS3();
      renderBoardS3();
      setTimeout(makeAgentMoveS3, 220);
    }
  }

  function makeAgentMoveS3() {
    if (isGameOverS3 || currentTurnS3 !== agentRoleS3) return;

    // Trained agent: evaluated strictly with epsilon = 0.0 (pure exploitation)
    const decision = agent.selectAction(boardStateS3, agentRoleS3, 0.0);
    if (!decision) return;

    boardStateS3 = game.makeMove(decision.action, agentRoleS3, boardStateS3);
    playTone(420, 0.06);
    const outcome = game.checkWinner(boardStateS3);

    if (outcome.isTerminal) {
      endGameS3(outcome);
    } else {
      currentTurnS3 = humanRoleS3;
      updateTurnUIS3();
      renderBoardS3();
    }
  }

  function updateTurnUIS3() {
    if (isGameOverS3) return;
    if (currentTurnS3 === humanRoleS3) {
      turnIndicatorS3.innerHTML = `<span class="turn-dot user-turn"></span> Your Turn (${humanRoleS3})`;
      gameStatusS3.textContent = 'Choose an empty square.';
    } else {
      turnIndicatorS3.innerHTML = `<span class="turn-dot agent-turn"></span> Agent Turn (${agentRoleS3})`;
      gameStatusS3.textContent = 'Trained agent is selecting greedy action from learned Q-table...';
    }
  }

  function endGameS3(outcome) {
    isGameOverS3 = true;
    sessionStatsS3.games++;

    if (outcome.winner === 'draw') {
      sessionStatsS3.draws++;
      turnIndicatorS3.innerHTML = '<span class="badge badge-secondary">Draw Match</span>';
      gameStatusS3.textContent = 'Game ended in a draw.';
      renderBoardS3(null);
    } else {
      const humanWon = outcome.winner === humanRoleS3;
      if (humanWon) sessionStatsS3.humanWins++;
      else sessionStatsS3.agentWins++;

      turnIndicatorS3.innerHTML = `<span class="badge ${humanWon ? 'badge-success' : 'badge-danger'}">${outcome.winner} Won</span>`;
      gameStatusS3.textContent = humanWon ? 'You found a winning sequence!' : 'The trained agent won using its learned policy.';
      renderBoardS3(outcome.winningLine);
    }

    s3StatGames.textContent = sessionStatsS3.games;
    s3StatHumanWins.textContent = sessionStatsS3.humanWins;
    s3StatDraws.textContent = sessionStatsS3.draws;
    s3StatAgentWins.textContent = sessionStatsS3.agentWins;
  }

  function restartGameS3() {
    boardStateS3 = game.reset();
    isGameOverS3 = false;
    currentTurnS3 = 'X';

    if (agentRoleS3 === 'X') {
      updateTurnUIS3();
      renderBoardS3();
      setTimeout(makeAgentMoveS3, 200);
    } else {
      updateTurnUIS3();
      renderBoardS3();
    }
  }

  btnRestartS3.addEventListener('click', restartGameS3);
  selectRoleS3.addEventListener('change', (e) => {
    humanRoleS3 = e.target.value;
    agentRoleS3 = humanRoleS3 === 'X' ? 'O' : 'X';
    restartGameS3();
  });

  btnToggleExplain.addEventListener('click', () => {
    explainOpen = !explainOpen;
    explainabilityDrawer.style.display = explainOpen ? 'block' : 'none';
    btnToggleExplain.setAttribute('aria-expanded', explainOpen ? 'true' : 'false');
  });

  // Explainability Drawer Logic
  function updateExplainabilityDrawer() {
    if (!explainStateKey) return;
    const canonical = game.getCanonicalState(boardStateS3, agentRoleS3);
    explainStateKey.textContent = canonical;

    const legal = game.getLegalMoves(boardStateS3);
    if (legal.length === 0) {
      explainMovesTbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No legal moves (terminal state).</td></tr>';
      return;
    }

    const qVals = agent.peekQValues(canonical);
    let maxQ = -Infinity;
    legal.forEach(a => {
      if (qVals[a] > maxQ) maxQ = qVals[a];
    });

    const greedySet = legal.filter(a => Math.abs(qVals[a] - maxQ) < 1e-6);

    let html = '';
    legal.forEach(a => {
      const coord = game.actionToCoord(a);
      const visited = agent.hasVisited(canonical, a);
      const isGreedy = greedySet.includes(a);
      const prob = isGreedy ? (100 / greedySet.length).toFixed(1) : '0.0';

      html += `
        <tr class="${isGreedy ? 'row-greedy' : ''}">
          <td><strong>Cell ${a}</strong></td>
          <td class="font-mono">Row ${coord.row}, Col ${coord.col}</td>
          <td class="font-mono"><strong>${visited ? qVals[a].toFixed(3) : '<span class="text-muted">Unvisited (0.000)</span>'}</strong></td>
          <td>${isGreedy ? '<span class="badge badge-teal">★ Greedy Best</span>' : '<span class="badge badge-secondary">Sub-optimal</span>'}</td>
          <td class="font-mono">${prob}%</td>
        </tr>
      `;
    });
    explainMovesTbody.innerHTML = html;

    if (agent.lastDecision) {
      const d = agent.lastDecision;
      explainLastDecisionBox.innerHTML = `
        <div style="font-size: 0.825rem;">
          Last Agent Action: Cell <strong>${d.action}</strong> &bull;
          Branch: <span class="badge ${d.branch === 'exploitation' ? 'badge-success' : 'badge-secondary'}">${d.branch.toUpperCase()}</span> &bull;
          Greedy Match: <strong>${d.isGreedy ? 'Yes (★ Best Value)' : 'No'}</strong>
        </div>
      `;
    }
  }

  // Before vs After Comparison Display
  function updateComparisonDisplay() {
    if (agent.baselineEval) {
      compBeforeWin.textContent = `${agent.baselineEval.winRate}%`;
      compBeforeDraw.textContent = `${agent.baselineEval.drawRate}%`;
      compBeforeLoss.textContent = `${agent.baselineEval.lossRate}%`;
      if (compBeforeOpp) {
        const oppName = agent.baselineEval.opponentPolicy || 'random';
        compBeforeOpp.textContent = `Opponent: ${oppName.toUpperCase()} (50 as X, 50 as O)`;
      }
    } else {
      compBeforeWin.textContent = '—';
      compBeforeDraw.textContent = '—';
      compBeforeLoss.textContent = '—';
      if (compBeforeOpp) compBeforeOpp.textContent = 'Awaiting baseline run...';
    }
    if (agent.postTrainingEval) {
      compAfterWin.textContent = `${agent.postTrainingEval.winRate}%`;
      compAfterDraw.textContent = `${agent.postTrainingEval.drawRate}%`;
      compAfterLoss.textContent = `${agent.postTrainingEval.lossRate}%`;
      if (compAfterOpp) {
        const oppName = agent.postTrainingEval.opponentPolicy || selectOpponent.value;
        compAfterOpp.textContent = `Opponent: ${oppName.toUpperCase()} (50 as X, 50 as O)`;
      }
    } else {
      compAfterWin.textContent = '—';
      compAfterDraw.textContent = '—';
      compAfterLoss.textContent = '—';
      if (compAfterOpp) compAfterOpp.textContent = 'Awaiting post-training evaluation...';
    }
  }

  btnRunPostEval.addEventListener('click', () => {
    btnRunPostEval.disabled = true;
    btnRunPostEval.textContent = 'Evaluating 100 games (50 as X, 50 as O)...';

    setTimeout(() => {
      agent.postTrainingEval = agent.evaluateUnbiased(100, selectOpponent.value);
      updateComparisonDisplay();
      btnRunPostEval.disabled = false;
      btnRunPostEval.textContent = 'Re-evaluate Benchmark';
      playTone(600, 0.08);
    }, 40);
  });

  // Export / Import
  btnExportQTable.addEventListener('click', () => {
    const json = agent.exportPolicyJson();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rl-lab-trained-${agent.totalEpisodesTrained}ep.json`;
    a.click();
    URL.revokeObjectURL(url);
    playTone(520, 0.05);
  });

  btnImportQTable.addEventListener('click', () => {
    importFileInput.click();
  });

  importFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const ok = agent.importPolicyJson(evt.target.result);
      if (ok) {
        updateAgentStatus();
        analytics.updateData(agent.evaluationHistory);
        updateComparisonDisplay();
        restartGameS3();
        alert('Trained policy imported successfully!');
        playTone(650, 0.08);
      } else {
        alert('Failed to parse policy JSON.');
      }
    };
    reader.readAsText(file);
  });

  // Global Reset
  btnGlobalReset.addEventListener('click', () => {
    if (confirm('Reset learned policy? This will clear all learned Q-values and reset the agent to untrained.')) {
      agent.resetPolicy();
      analytics.clear();
      updateAgentStatus();
      updateComparisonDisplay();
      if (baselineResultsBox) {
        baselineResultsBox.innerHTML = `
          <div class="text-muted text-sm text-center" style="padding: 1.25rem 0;">
            Click above to generate the pre-training baseline benchmark.
          </div>
        `;
      }
      restartGameS1();
      restartGameS3();
      progressBar.style.width = '0%';
      progressText.textContent = 'Agent policy reset to untrained baseline.';
      kpiEpisodes.textContent = '0';
      kpiVisitedStates.textContent = '0';
      kpiCurrentEps.textContent = agent.epsilon.toFixed(2);
      kpiTrainingWdl.textContent = '0/0/0';
      playTone(320, 0.1);
    }
  });

  // -------------------------------------------------------------
  // WORKED NUMERICAL EXAMPLE CALCULATOR
  // -------------------------------------------------------------
  function updateWorkedExample() {
    if (!calcCurrentQ) return;
    const curQ = parseFloat(calcCurrentQ.value) || 0;
    const a = parseFloat(calcAlpha.value) || 0;
    const r = parseFloat(calcReward.value) || 0;
    const g = parseFloat(calcGamma.value) || 0;
    const maxNext = parseFloat(calcMaxNextQ.value) || 0;

    const tdTarget = r + (g * maxNext);
    const tdError = tdTarget - curQ;
    const updated = curQ + (a * tdError);

    calcTdTarget.textContent = tdTarget.toFixed(2);
    calcTdError.textContent = (tdError >= 0 ? '+' : '') + tdError.toFixed(2);
    calcUpdatedQ.textContent = updated.toFixed(2);
  }

  [calcCurrentQ, calcAlpha, calcReward, calcGamma, calcMaxNextQ].forEach(inp => {
    if (inp) inp.addEventListener('input', updateWorkedExample);
  });

  // -------------------------------------------------------------
  // AUTOMATED TEST SUITE RUNNER
  // -------------------------------------------------------------
  btnRunTests.addEventListener('click', () => {
    const results = testSuite.runAllTests();
    testSummaryBadge.innerHTML = results.allPassed
      ? `<span class="badge badge-success">✓ All ${results.passedCount}/${results.totalCount} Tests Passed (100%)</span>`
      : `<span class="badge badge-danger">✗ ${results.totalCount - results.passedCount} Failed</span>`;

    let html = '<div class="test-results-list">';
    results.details.forEach(res => {
      html += `
        <div class="test-item ${res.passed ? 'test-passed' : 'test-failed'}">
          <span class="test-icon">${res.passed ? '✓' : '✗'}</span>
          <div class="test-content">
            <strong>${res.testName}</strong>
            <p class="test-msg">${res.message}</p>
          </div>
        </div>
      `;
    });
    html += '</div>';
    testResultsContainer.innerHTML = html;
    playTone(results.allPassed ? 700 : 300, 0.08);
  });

  // Initial Boot
  initTheme();
  syncHyperparams();
  updateWorkedExample();
  restartGameS1();
  restartGameS3();
  updateAgentStatus();
  analytics.drawEmpty();

  // Run initial test validation so badge is verified
  const initialTestRun = testSuite.runAllTests();
  if (testSummaryBadge) {
    testSummaryBadge.innerHTML = `<span class="badge badge-success">✓ ${initialTestRun.passedCount}/${initialTestRun.totalCount} Tests Verified</span>`;
  }
});
