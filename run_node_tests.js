/**
 * Node.js test runner for RL Lab
 * Executes the actual JavaScript implementations in js/agent.js, js/game.js, and js/tests.js.
 */
const { TicTacToeGame } = require('./js/game.js');
const { QLearningAgent } = require('./js/agent.js');
const { RLTestSuite } = require('./js/tests.js');

console.log('=======================================================');
console.log('RL Lab — JavaScript Test Suite (Node.js Engine)');
console.log('Running on Node.js:', process.version);
console.log('=======================================================\n');

const suite = new RLTestSuite();
const results = suite.runAllTests();

results.details.forEach(res => {
  const status = res.passed ? '[PASS]' : '[FAIL]';
  console.log(`  ${status} ${res.testName}: ${res.message}`);
});

console.log('\n=======================================================');
if (results.allPassed) {
  console.log(`ALL ${results.passedCount}/${results.totalCount} JAVASCRIPT INVARIANTS PASSED (100% SUCCESS)!`);
  console.log('=======================================================');
  process.exit(0);
} else {
  const failed = results.totalCount - results.passedCount;
  console.error(`TEST SUITE FAILED: ${failed} out of ${results.totalCount} tests failed.`);
  console.log('=======================================================');
  process.exit(1);
}
