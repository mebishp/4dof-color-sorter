#!/usr/bin/env python3
"""Unified Automated Testing & Verification Harness for Sorter Simulation.

Usage:
  python3 test_runner.py [all|unit|health|vision|e2e] [--json]

Features:
  - Fast offline unit test execution (< 1 sec)
  - Live simulation & transport health probes
  - Real-time vision-to-groundtruth calibration checks
  - End-to-end sorting policy and CSV metrics verification
  - Structured test_report.json generation for agent/CI pipelines
"""

import argparse
import json
import os
import subprocess
import sys
import time

TEST_DIR = "/home/hyprnon/ros_ws/tests"
REPORT_FILE = "/home/hyprnon/ros_ws/test_report.json"

# ANSI Colors
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"

SUITES = {
    'unit': [
        ('Kinematics & Tool Constraint', f"{TEST_DIR}/test_ik_kinematics.py"),
        ('Procedural Randomization & Clearance', f"{TEST_DIR}/test_randomization.py"),
    ],
    'health': [
        ('Simulation & DDS Transport Health', f"{TEST_DIR}/test_transport_health.py"),
    ],
    'vision': [
        ('Vision Detection & Groundtruth Accuracy', f"{TEST_DIR}/test_vision_groundtruth.py"),
    ],
    'e2e': [
        ('End-to-End Sorting Policy & CSV Metrics', f"{TEST_DIR}/test_sorting_e2e.py"),
    ],
}


def run_single_test(test_path: str, timeout: int = 25) -> dict:
    """Run a single test script and capture stdout, stderr, exit code, duration."""
    start = time.time()
    env = os.environ.copy()
    ros_python_paths = [
        "/opt/ros/humble/lib/python3.14/site-packages",
        "/opt/ros/humble/lib/python3.14/dist-packages",
        "/home/hyprnon/ros_ws/src/color_sorter_vision",
        TEST_DIR,
    ]
    cur_pythonpath = env.get('PYTHONPATH', '')
    all_paths = [p for p in ros_python_paths if os.path.exists(p)]
    if cur_pythonpath:
        all_paths.append(cur_pythonpath)
    env['PYTHONPATH'] = ":".join(all_paths)

    ros_lib_path = "/opt/ros/humble/lib"
    cur_ld = env.get('LD_LIBRARY_PATH', '')
    if os.path.exists(ros_lib_path) and ros_lib_path not in cur_ld:
        env['LD_LIBRARY_PATH'] = f"{ros_lib_path}:{cur_ld}" if cur_ld else ros_lib_path

    env.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', '/home/hyprnon/ros_ws/fastdds_udp.xml')
    env.setdefault('ROS_DOMAIN_ID', '0')
    env.setdefault('AMENT_PREFIX_PATH', '/opt/ros/humble')

    try:
        proc = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        duration = time.time() - start
        passed = (proc.returncode == 0)
        skipped = ("skipped=" in proc.stderr.lower() or "skip" in proc.stdout.lower()) and passed

        return {
            'passed': passed,
            'skipped': skipped,
            'duration_sec': round(duration, 3),
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            'passed': False,
            'skipped': False,
            'duration_sec': round(duration, 3),
            'stdout': '',
            'stderr': f"Test timed out after {timeout} seconds",
            'returncode': -1,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            'passed': False,
            'skipped': False,
            'duration_sec': round(duration, 3),
            'stdout': '',
            'stderr': str(e),
            'returncode': -2,
        }


def main():
    parser = argparse.ArgumentParser(description="Sorter Simulation Automated Test Harness")
    parser.add_argument('suite', nargs='?', default='all', choices=['all', 'unit', 'health', 'vision', 'e2e'],
                        help="Test suite to run (default: all)")
    parser.add_argument('--json', action='store_true', help="Output pure JSON report to stdout")
    parser.add_argument('--timeout', type=int, default=30, help="Per-test timeout in seconds")
    args = parser.parse_args()

    # Determine which tests to run
    if args.suite == 'all':
        active_tests = []
        for s in ['unit', 'health', 'vision', 'e2e']:
            active_tests.extend(SUITES[s])
    else:
        active_tests = SUITES[args.suite]

    if not args.json:
        print(f"\n{BOLD}{CYAN}======================================================{NC}")
        print(f"{BOLD}{CYAN}🚀 Sorter Simulation Automated Test Harness: [{args.suite.upper()}]{NC}")
        print(f"{BOLD}{CYAN}======================================================{NC}\n")

    results = []
    all_passed = True
    start_total = time.time()

    for name, test_path in active_tests:
        if not args.json:
            print(f"▶ Running {BOLD}{name}{NC}...", end=" ", flush=True)

        res = run_single_test(test_path, timeout=args.timeout)
        res['name'] = name
        res['script'] = os.path.basename(test_path)
        results.append(res)

        if not res['passed'] and not res['skipped']:
            all_passed = False
            if not args.json:
                print(f"{RED}✗ FAIL{NC} ({res['duration_sec']}s)")
                if res['stderr']:
                    print(f"  {RED}{res['stderr'].strip()}{NC}\n")
        elif res['skipped']:
            if not args.json:
                print(f"{YELLOW}○ SKIP{NC} ({res['duration_sec']}s)")
        else:
            if not args.json:
                print(f"{GREEN}✓ PASS{NC} ({res['duration_sec']}s)")

    total_duration = time.time() - start_total

    # Assemble report
    report = {
        'suite': args.suite,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'total_duration_sec': round(total_duration, 3),
        'all_passed': all_passed,
        'summary': {
            'total': len(results),
            'passed': sum(1 for r in results if r['passed'] and not r['skipped']),
            'failed': sum(1 for r in results if not r['passed']),
            'skipped': sum(1 for r in results if r['skipped']),
        },
        'tests': results,
    }

    # Save to disk
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{BOLD}------------------------------------------------------{NC}")
        print(f"Results: {GREEN}{report['summary']['passed']} Passed{NC}, "
              f"{RED}{report['summary']['failed']} Failed{NC}, "
              f"{YELLOW}{report['summary']['skipped']} Skipped{NC} "
              f"(Total Time: {total_duration:.2f}s)")
        print(f"Report written to: {REPORT_FILE}")
        print(f"{BOLD}------------------------------------------------------{NC}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
