from datetime import datetime, timedelta
import sys
import traceback
import requests

# Author: Antoni Baum (Yard1)
# Helper script to see if it's worth pushing to nightly, to be used in a GitHub Workflow
# exit code 0 = push to nightly
# exit code 1 = don't push to nightly

BASE_URL = "https://api.github.com"
OWNER = "pycaret"
REPO = "pycaret"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def get_commits_from_repo(owner: str, repo: str):
    r = requests.get(f"{BASE_URL}/repos/{owner}/{repo}/commits")
    if r.status_code == 200 and r.text:
        return r.json()
    else:
        r.raise_for_status()
        return


def get_commit_from_repo_sha(owner: str, repo: str, sha: str):
    commit = {}
    commit_status = {}
    r = requests.get(f"{BASE_URL}/repos/{owner}/{repo}/commits/{sha}")
    if r.status_code == 200 and r.text:
        commit = r.json()
    else:
        r.raise_for_status()
    # get status too
    r = requests.get(f"{BASE_URL}/repos/{owner}/{repo}/commits/{sha}/status")
    if r.status_code == 200 and r.text:
        commit_status = r.json()
    else:
        r.raise_for_status()
    return (commit, commit_status)


def are_there_commits_in_last_day(commit) -> bool:
    now = datetime.now()
    last_day = now - timedelta(days=1)
    last_commit_date = commit["commit"]["committer"]["date"]
    last_commit_date = datetime.strptime(last_commit_date, DATE_FORMAT)
    return last_commit_date >= last_day


def has_commit_passed_ci(commit_status) -> bool:
    return commit_status["state"] == "success"


def main():
    try:
        commits = get_commits_from_repo(OWNER, REPO)
        latest_commit, latest_commit_status = get_commit_from_repo_sha(
            OWNER, REPO, commits[0]["sha"]
        )
        if are_there_commits_in_last_day(latest_commit) and has_commit_passed_ci(latest_commit_status):
            print(
                f"Latest commit {latest_commit['sha']} was made after 24h ago and passed CI, can push to nightly"
            )
            return 0
    except:
        print(f"There was an exception, push to nightly anyway")
        traceback.print_exc()
        return 0
    print(
        f"Latest commit {latest_commit['sha']} was either made before 24 ago or didn't pass CI, can't push to nightly"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
