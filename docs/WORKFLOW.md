# Workflow: tickets, branches, releases

Every change – feature, bug fix or chore – is a **ticket `HPAI-<number>`**. Numbers are
sequential and never reused; the log and the next free number live in
[`progress_context.md`](../progress_context.md#8-ticket-log-hpai-xx).

## Branches

| Branch | Purpose |
|---|---|
| `main` | **Production.** Only receives releases from `development`. |
| `development` | Integration. Every ticket branch merges here first. |
| `<type>/HPAI-<n>-<short-name>` | One ticket. Created from `development`. |

`<type>` is one of:

| Type | Use for |
|---|---|
| `feature` | new functionality |
| `bugfix` | fixing something broken |
| `hotfix` | urgent fix for `main` (branch from `main`, merge to `main` **and** `development`) |
| `chore` | tooling, docs, refactors, dependency upgrades |

Examples: `feature/HPAI-16-retrain-with-app-labels`, `bugfix/HPAI-17-upload-in-background`.
`<short-name>` is lowercase words joined by `-`.

## Day to day

```bash
# 1. take the next free number from progress_context.md and add a row to the ticket log
git switch development && git pull
git switch -c feature/HPAI-16-retrain-with-app-labels

# 2. commit – messages start with the ticket (the hook pre-fills it from the branch name)
git commit -m "HPAI-16: add app_labeled.csv to training data"

# 3. push and open a pull request into development (title starts with HPAI-16)
git push -u origin feature/HPAI-16-retrain-with-app-labels
```

Merge with a merge commit (`--no-ff`) so the ticket stays visible in history, then delete
the ticket branch. Update `progress_context.md` (status, decisions, results) in the same
pull request.

## Releases to production

When `development` is tested (server tests pass, app bundles, tried on a phone):

```bash
git switch main && git pull
git merge --ff-only development      # or a PR development → main
git push origin main
```

## Commit messages

```
HPAI-16: short imperative summary

Why the change was needed and what it does.
```

Merge commits, reverts and `fixup!`/`squash!` commits are exempt.

## Enforcement

- **Local hooks** (`scripts/git-hooks/`): `prepare-commit-msg` adds `HPAI-<n>: ` from the
  branch name, `commit-msg` rejects messages without a ticket. Install once per clone:
  ```bash
  sh scripts/git-hooks/install.sh
  ```
  (They are copied into `.git/hooks`, next to the Git LFS hooks, which stay untouched.)
- **GitHub Actions** (`.github/workflows/ticket-check.yml`): pull requests must come from
  a `<type>/HPAI-<n>-<name>` branch (or `development` → `main` for releases) and have a
  title starting with `HPAI-<n>`.
- **Templates**: `.github/ISSUE_TEMPLATE/` (feature, bug) and
  `.github/pull_request_template.md`.
