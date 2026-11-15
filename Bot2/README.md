# Autonomous GitHub Code Review and Quality Bot (Bot 2)

[![CI Test Suite](https://github.com/AP24110011763/cpu-gpu-canny-multi-agent/actions/workflows/bot2-tests.yml/badge.svg)](https://github.com/AP24110011763/cpu-gpu-canny-multi-agent/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An autonomous, security-first GitHub Code Review and Quality Bot built as **Bot 2** for our college multi-bot software engineering project.

---

## 📌 Project Context

This repository has two autonomous bots that work as teammates. You can talk to them, and they talk to each other (see [`team/README.md`](../team/README.md)):
- **Bot 1:** File manager and reporter. Files uploads from `inbox/`, moves/deletes/lists files on command, asks you when unsure where a file belongs, and posts repo activity to the Activity Log issue.
- **Bot 2 (This Bot):** Autonomous Code Review and Quality Bot. Inspects pull request changes, detects security vulnerabilities, logic errors, bug risks, test/documentation gaps, and posts actionable reviews automatically via GitHub Actions. It also reviews the code files Bot 1 files and reports its verdict back to Bot 1.

> The workflows live in the repository root `.github/workflows/` (`bot2-review.yml`, `bot2-tests.yml`) because GitHub only reads workflows from there. They run the bot with `working-directory: Bot2`.

---

## 🚀 Key Features of Bot 2

1. **Autonomous GitHub Actions Trigger:**
   Executes automatically whenever a Pull Request is **opened**, **reopened**, or **updated with new commits** (`synchronize`).

2. **Dual-Engine Review Architecture:**
   - **Static & Security Linter:** Detects hardcoded secrets, API tokens, SQL injection risks, arbitrary code execution (`eval`, `exec`, `os.system`, `subprocess(shell=True)`), bare `except:` blocks, and mutable default arguments.
   - **AI Semantic Reviewer (Google Gemini Flash):** Analyzes patch logic, concurrency risks, edge cases, off-by-one errors, and architectural maintainability.

3. **PR Hygiene & Coverage Verification:**
   Checks if modified production code has corresponding unit/integration test updates and documentation changes.

4. **Actionable Suggestions with Line Numbers:**
   Pins findings to exact file names and changed line numbers with drop-in code fixes.

5. **Spam Prevention & Comment Deduplication:**
   Embeds an identifier marker (`<!-- bot-2-code-review -->`). When contributors push additional commits, Bot 2 updates the existing PR review comment in-place instead of flooding the timeline with duplicate messages.

6. **Least-Privilege Security:**
   Only requires `contents: read`, `pull-requests: write`, and `issues: write`. Does not merge PRs, push commits, or alter project files.

7. **Zero Silent Failures:**
   If the optional `GEMINI_API_KEY` secret is missing, Bot 2 emits clear GitHub Actions runner warnings, falls back to the static analyzer, and provides explicit setup instructions right in the PR review report.

8. **Zero Runtime Dependencies:**
   Built entirely on the Python 3 standard library (`urllib`, `json`, `ast`, `re`). Runs instantly on GitHub Actions runners with no `pip install` overhead or supply-chain vulnerabilities.

---

## 🏗️ Architecture & Component Layout

```
coa-project/
├── .github/
│   └── workflows/
│       ├── code-review-bot.yml     # Automated PR review GitHub Actions workflow
│       └── bot-tests.yml           # CI workflow to run Bot 2 test suite
├── bot2/                           # Bot 2 Core Package
│   ├── __init__.py
│   ├── config.py                   # Environment configuration & secret validation
│   ├── diff_parser.py              # Unified git diff & patch hunk parser
│   ├── static_analyzer.py          # Rule-based AST & regex security scanner
│   ├── ai_reviewer.py              # Google Gemini Flash API integration
│   ├── formatter.py                # GitHub Markdown report formatter
│   ├── github_client.py            # GitHub REST API client (with deduplication)
│   ├── reviewer.py                 # Review orchestrator (merges & deduplicates)
│   └── main.py                     # CLI entrypoint for GitHub Actions
├── tests/                          # Automated Unit Test Suite
│   ├── test_diff_parser.py
│   ├── test_static_analyzer.py
│   ├── test_formatter.py
│   ├── test_ai_reviewer.py
│   ├── test_github_client.py
│   └── test_reviewer.py
├── sample_pr_diffs/                # Local demonstration fixtures
│   ├── buggy_change.patch          # Patch with SQL injection, secrets, bugs
│   └── clean_change.patch          # Patch with clean, tested, documented code
├── demo_local_review.py            # Interactive CLI tool for local demonstration
├── requirements.txt                # Optional dev/testing dependencies
└── README.md                       # Complete documentation
```

---

## ⚙️ GitHub Setup & Configuration

### 1. Workflow Permissions
The workflow `.github/workflows/code-review-bot.yml` runs with least-privilege permissions:
```yaml
permissions:
  contents: read          # Read PR diff and repository files
  pull-requests: write    # Post or update PR review comments
  issues: write           # Fallback for PR comment APIs
```

### 2. Configuring Repository Secrets
Bot 2 utilizes two tokens:
1. `GITHUB_TOKEN`: Provided **automatically** by GitHub Actions with no setup required.
2. `GEMINI_API_KEY` *(Optional for AI Semantic Review)*:
   - Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
   - In your GitHub repository, navigate to:
     `Settings` ➔ `Secrets and variables` ➔ `Actions` ➔ Click **New repository secret**.
   - Name: `GEMINI_API_KEY`
   - Value: Paste your Gemini API key and click **Add secret**.

> **Note on Secret Resilience:** If `GEMINI_API_KEY` is not provided, Bot 2 **never fails silently**. It outputs a clear warning in the GitHub Actions runner log, runs all static security/bug inspections, and displays a configuration notice on the PR review.

---

## 🧪 Local Demonstration & Testing

You can evaluate and demonstrate Bot 2 locally without opening a GitHub PR.

### Run Unit Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Run Review on Sample Diff (Buggy Code)
Inspects a sample patch containing a hardcoded token, SQL injection, `os.system` command injection, bare `except:`, and missing tests:
```bash
python3 demo_local_review.py --file sample_pr_diffs/buggy_change.patch
```

### Run Review on Sample Diff (Clean Code)
Inspects a sample patch containing safe functions, unit tests, and docstrings:
```bash
python3 demo_local_review.py --file sample_pr_diffs/clean_change.patch
```

### Run Review with Gemini AI Enabled Locally
```bash
python3 demo_local_review.py --file sample_pr_diffs/buggy_change.patch --api-key YOUR_GEMINI_API_KEY
```

### Run Review on Current Local Git Changes
```bash
python3 demo_local_review.py --git
```

---

## 📋 Example Review Comments

### Example 1: Issues Found (Buggy Code Detected)

```markdown
<!-- bot-2-code-review -->
## 🤖 Bot 2: Autonomous Code Review & Quality Report

| Review Status | Files Changed | Code Findings | Advisories | Engine Mode |
| :--- | :---: | :---: | :---: | :---: |
| 🔴 **Issues found** | 1 | 5 | 1 | Hybrid (Static + AI) |

### 📋 Executive Summary
Review completed with status **Issues found (Action required)**. Critical or high-severity problems were detected in the pull request changes. Please review the actionable suggestions below.

### 🔍 Detailed Findings
#### 1. 🔴 `CRITICAL` GitHub Personal Access Token Detected
- **Location:** `src/auth.py (Line 4)`
- **Category:** Security | **Detector:** `STATIC`
- **Problem:** A personal GitHub token was found directly in source code.
- **Actionable Suggestion:** Revoke the token in GitHub Settings and inject it via GitHub Actions Secrets.

#### 2. 🔴 `CRITICAL` Potential SQL Injection Risk
- **Location:** `src/auth.py (Line 11)`
- **Category:** Security | **Detector:** `STATIC`
- **Problem:** SQL query appears to be constructed using string formatting or interpolation instead of parameterized queries.
- **Actionable Suggestion:** Use parameterized SQL queries (e.g., `cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))`).

#### 3. 🟠 `HIGH` Unsafe Command Execution with `os.system`
- **Location:** `src/auth.py (Line 20)`
- **Category:** Security | **Detector:** `STATIC`
- **Problem:** `os.system` runs commands through the system shell without proper escaping.
- **Actionable Suggestion:** Replace `os.system()` with `subprocess.run([...], check=True)` passing argument lists.

#### 4. 🟠 `HIGH` Dynamic Code Execution (`eval`/`exec`)
- **Location:** `src/auth.py (Line 24)`
- **Category:** Security | **Detector:** `STATIC`
- **Problem:** Using `eval()` or `exec()` executes arbitrary string input, which can lead to remote code execution (RCE).
- **Actionable Suggestion:** Avoid dynamic code execution. Use `ast.literal_eval()` for safe literals or refactor to dedicated parsers.

#### 5. 🟡 `MEDIUM` Bare `except:` Swallowing Exceptions
- **Location:** `src/auth.py (Line 15)`
- **Category:** Bug Risk | **Detector:** `STATIC`
- **Problem:** A bare `except:` catches all exceptions (including SystemExit, KeyboardInterrupt) and silently ignores them.
- **Actionable Suggestion:** Catch specific exception classes (e.g., `except (ValueError, KeyError):`) and log or handle the error appropriately.

### 🧪 Testing & Documentation Hygiene
- ⚠️ **Missing Test Coverage:** This PR modifies/adds 1 source file(s) (src/auth.py), but no unit or integration tests were added or updated.
  - *Action:* Add corresponding test cases in the test suite to verify the new behavior and guard against regressions.

---
> ℹ️ *Bot 2 reviewed commit [`a1b2c3d`] at 2026-10-31 10:15:00 UTC. To prevent duplicate comments, this review updates automatically when new commits are pushed.*
```

---

### Example 2: Clean Code (No Significant Issues Found)

```markdown
<!-- bot-2-code-review -->
## 🤖 Bot 2: Autonomous Code Review & Quality Report

| Review Status | Files Changed | Code Findings | Advisories | Engine Mode |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **No significant issues found** | 2 | 0 | 0 | Hybrid (Static + AI) |

### 📋 Executive Summary
All changed files were inspected for potential bugs, logic errors, security vulnerabilities, and testing hygiene. **No significant issues were found.** Changes look clean and safe to merge.

### 🔍 Detailed Findings
✅ **No code defects, security smells, or logic errors were identified in the changed lines.**

### 🧪 Testing & Documentation Hygiene
- ✅ Test coverage and documentation requirements appear satisfied for this change.

---
> ℹ️ *Bot 2 reviewed commit [`e4f5a6b`] at 2026-10-31 10:20:00 UTC. To prevent duplicate comments, this review updates automatically when new commits are pushed.*
```

---

## 🔒 Security & Safe Operating Principles

- **Read-Only Code Analysis:** Bot 2 never pushes commits, modifies files in the repository, or automatically merges pull requests.
- **Credential Protection:** Secrets are masked from all console outputs. Never commit tokens to source code or git history.
- **Diff Size Guardrails:** Configured with a default limit (`150 KB`) to prevent API abuse, runaway memory consumption, or excessive latency on large PRs.
- **Non-blocking Execution:** Operates as a supportive reviewer. Unless `REQUIRE_AI_KEY="true"` is explicitly set, missing optional secrets will not block repository development.

---

## 💡 Limitations & Trade-offs

1. **Large Diffs:** Extremely large pull requests (>150 KB diffs, generated bundle files) have non-code assets filtered out and large diffs truncated to fit within LLM context windows.
2. **Third-Party Static Analysis Tools:** To keep Bot 2 lightweight with zero external runtime dependencies, it uses native Python AST parsing and regex rule engines rather than heavyweight tools like SonarQube or Semgrep.
