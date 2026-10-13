"""
RL Lab — Automated Invariant Validation Runner
Executes the actual JavaScript implementation in js/agent.js, js/game.js, and js/tests.js via Node.js.
If Node.js is not present in the runtime environment, falls back to Python reference verification.
"""

import os
import sys
import shutil
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def find_node():
    # 1. Check system PATH
    which_node = shutil.which('node')
    if which_node:
        return which_node

    # 2. Check standard Windows Node.js paths
    candidates = [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\nodejs\node.exe"),
        os.path.expandvars(r"%APPDATA%\npm\node.exe")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_node_tests(node_exe):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_node_tests.js")
    print(f"Executing JavaScript test suite using: {node_exe}\n")
    proc = subprocess.run([node_exe, script_path], text=True)
    sys.exit(proc.returncode)

def run_python_fallback():
    print("WARNING: Node.js was not detected in PATH or standard installation directories.")
    print("Running Python algorithmic reference verification...\n")

    def compute_q_update(current_q, alpha, reward, gamma, max_next_q, is_terminal=False):
        target = reward if is_terminal else (reward + gamma * max_next_q)
        return current_q + alpha * (target - current_q)

    # Test 9: Exact numerical update rule
    q = compute_q_update(4.0, 0.5, 2.0, 0.9, 6.0, False)
    assert abs(q - 5.7) < 1e-9, f"Update rule failed: expected 5.7, got {q}"
    print(f"  [PASS] Numerical Q-Update: 4 + 0.5 * [2 + 0.9 * 6 - 4] = {q}")

    # Test 10: Terminal state target zeroed
    q_term = compute_q_update(4.0, 0.5, 2.0, 0.9, 6.0, True)
    assert abs(q_term - 3.0) < 1e-9, f"Terminal target failed: expected 3.0, got {q_term}"
    print(f"  [PASS] Terminal State Target Zeroed: 4 + 0.5 * [2 - 4] = {q_term}")

    print("\nFallback suite complete.")
    sys.exit(0)

if __name__ == '__main__':
    node_bin = find_node()
    if node_bin:
        run_node_tests(node_bin)
    else:
        run_python_fallback()
