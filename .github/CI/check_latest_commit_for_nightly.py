import sys
import traceback
from datetime import datetime, timedelta

import requests

# Author: Antoni Baum (Yard1)
# Helper script to see if it's worth pushing to nightly, to be used in a GitHub Workflow
# exit code 0 = push to nightly
# exit code 1 = don't push to nightly
# arguments: repo (eg. pycaret/pycaret), branch

BASE_URL = "https://api.github.com"
REPO = sys.argv[1]
BRANCH = sys.argv[2]
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def get_workflows_from_repo(repo: str):
    r = requests.get(f"{BASE_URL}/repos/{repo}/actions/workflows")
    if r.status_code == 200 and r.text:
        return r.json()
    else:
        r.raise_for_status()


def get_workflow_runs_from_repo(repo: str):
    r = requests.get(f"{BASE_URL}/repos/{repo}/actions/runs")
    if r.status_code == 200 and r.text:
        return r.json()
    else:
        r.raise_for_status()


def get_workflow_runs_for_id_branch(runs, workflow_id, branch):
    return [
        x
        for x in runs
        if x["head_branch"] == branch
        and x["workflow_id"] == workflow_id
        and x["event"] == "push"
    ]


def was_workflow_completed_in_last_day(run) -> bool:
    now = datetime.now()
    last_day = now - timedelta(days=1)
    last_workflow_date = run["updated_at"]
    last_workflow_date = datetime.strptime(last_workflow_date, DATE_FORMAT)
    print(
        f"Date of run {run['id']}: {last_workflow_date} - date 24h ago: {last_day}",
        file=sys.stderr,
    )
    return last_workflow_date >= last_day


def has_commit_passed_workflow(run) -> bool:
    return run["conclusion"] == "success"


def main():
    try:
        workflows = get_workflows_from_repo(REPO)["workflows"]
        runs = get_workflow_runs_from_repo(REPO)["workflow_runs"]
        test_workflows = [x for x in workflows if "test" in x["name"].lower()]

        for test_workflow in test_workflows:
            print(
                f"\"{test_workflow['name']}\" determined as test workflow",
                file=sys.stderr,
            )
            test_workflow_id = test_workflow["id"]
            latest_passing_run = next(
                run
                for run in get_workflow_runs_for_id_branch(
                    runs, test_workflow_id, BRANCH
                )
                if has_commit_passed_workflow(run)
            )
            print(
                f"Latest \"{test_workflow['name']}\" passing run for branch \"{BRANCH}\" is {latest_passing_run['id']} with conclusion \"{latest_passing_run['conclusion']}\"",
                file=sys.stderr,
            )
            if was_workflow_completed_in_last_day(latest_passing_run):
                print(latest_passing_run["head_sha"])
                print("Returning 0", file=sys.stderr)
                return 0
        print("Returning 1", file=sys.stderr)
        return 1
    except Exception:
        print(f"There was an exception", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    print("Returning 1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
