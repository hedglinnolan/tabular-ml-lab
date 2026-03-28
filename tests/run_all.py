#!/usr/bin/env python
"""Run all test suites."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.workflow.test_regression_flow import run as run_regression
from tests.workflow.test_state_invalidation import run as run_invalidation
from tests.workflow.test_target_transform import run as run_transform

if __name__ == "__main__":
    results = []
    
    print("\n" + "🔬" * 30)
    print("TABULAR ML LAB — FULL TEST SUITE")
    print("🔬" * 30)
    
    results.append(("Regression Flow", run_regression()))
    results.append(("State Invalidation", run_invalidation()))
    results.append(("Target Transform", run_transform()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        if not passed:
            all_pass = False
    
    print(f"\n{'ALL SUITES PASSED' if all_pass else 'SOME SUITES FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
