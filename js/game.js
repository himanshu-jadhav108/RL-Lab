/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: game.js
 * Purpose: Tic-Tac-Toe environment logic, state encoding, win/draw detection, and legal action filtering.
 */

class TicTacToeGame {
  constructor() {
    this.board = Array(9).fill(' ');
    this.WINNING_COMBINATIONS = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6]             // Diagonals
    ];
  }

  /**
   * Reset the board to initial empty state.
   */
  reset() {
    this.board = Array(9).fill(' ');
    return [...this.board];
  }

  /**
   * Clone board state.
   */
  cloneBoard(board = this.board) {
    return [...board];
  }

  /**
   * Check if a move is legal on the current or given board.
   * @param {number} action - Cell index (0-8)
   * @param {Array} board - Optional board array
   */
  isLegal(action, board = this.board) {
    return action >= 0 && action < 9 && board[action] === ' ';
  }

  /**
   * Get all legal action indices (empty cells).
   * @param {Array} board - Optional board array
   * @returns {number[]} Array of legal cell indices
   */
  getLegalMoves(board = this.board) {
    const moves = [];
    for (let i = 0; i < 9; i++) {
      if (board[i] === ' ') {
        moves.push(i);
      }
    }
    return moves;
  }

  /**
   * Apply move for player on board.
   * @param {number} action - Cell index (0-8)
   * @param {string} player - 'X' or 'O'
   * @param {Array} board - Optional board array
   * @returns {Array|null} New board copy, or null if illegal
   */
  makeMove(action, player, board = this.board) {
    if (!this.isLegal(action, board)) {
      return null;
    }
    const nextBoard = [...board];
    nextBoard[action] = player;
    return nextBoard;
  }

  /**
   * Detect winner or draw.
   * @param {Array} board - Optional board array
   * @returns {{winner: string|null, isTerminal: boolean, winningLine: number[]|null}}
   */
  checkWinner(board = this.board) {
    for (const combo of this.WINNING_COMBINATIONS) {
      const [a, b, c] = combo;
      if (board[a] !== ' ' && board[a] === board[b] && board[b] === board[c]) {
        return {
          winner: board[a], // 'X' or 'O'
          isTerminal: true,
          winningLine: combo
        };
      }
    }

    const hasEmpty = board.some(cell => cell === ' ');
    if (!hasEmpty) {
      return {
        winner: 'draw',
        isTerminal: true,
        winningLine: null
      };
    }

    return {
      winner: null,
      isTerminal: false,
      winningLine: null
    };
  }

  /**
   * Returns canonical state key from the perspective of the acting player.
   * '1': Acting agent's marks
   * '2': Opponent's marks
   * '0': Empty cells
   * This ensures the state space is invariant to whether the agent plays X or O.
   *
   * @param {Array} board - 9-element array
   * @param {string} player - Perspective player ('X' or 'O')
   * @returns {string} 9-character string key, e.g. "100020000"
   */
  getCanonicalState(board = this.board, player = 'X') {
    const opp = player === 'X' ? 'O' : 'X';
    return board.map(cell => {
      if (cell === player) return '1';
      if (cell === opp) return '2';
      return '0';
    }).join('');
  }

  /**
   * Human-readable state representation for inspector display.
   * @param {Array} board 
   * @returns {string} e.g. "X | . | O / . | X | . / . | . | O"
   */
  formatBoard(board = this.board) {
    const b = board.map(c => c === ' ' ? '.' : c);
    return `${b[0]} ${b[1]} ${b[2]} | ${b[3]} ${b[4]} ${b[5]} | ${b[6]} ${b[7]} ${b[8]}`;
  }

  /**
   * Coordinate converter from action index to {row, col}.
   */
  actionToCoord(action) {
    return {
      row: Math.floor(action / 3),
      col: action % 3
    };
  }
}

// Export for module and window environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TicTacToeGame };
} else {
  window.TicTacToeGame = TicTacToeGame;
}
