<p align="center">
  <img src="https://michaelgmunz.com/wp-content/uploads/2016/09/morbo-doom.png">
</p>

# Morbo

**Morbo** is an AI-powered GitHub PR assistant that automates pull request summaries using a large language model. Designed to integrate directly with your CI/CD pipeline, Morbo detects code changes introduced in a pull request, generates a concise summary of the diff, and posts it directly to the PR — so developers can focus on writing code, not explaining it.

---

## What It Does

Whenever a pull request is opened or updated, **Morbo**:

1. Pulls the diff from the PR using GitHub's API.
2. Sends the diff to a deployed **DeepSeek LLM** hosted on Google Cloud Platform.
3. Receives a short, human-readable summary of the changes.
4. Automatically posts the summary as a comment in the pull request.

---

## Tech Stack

-   **Python** – Core application logic
-   **Docker** – Containerized for deployment across CI environments
-   **DeepSeek LLM** – Custom AI model hosted on GCP for summarizing code diffs
-   **GitHub API** – For fetching diffs and posting PR comments
-   **CI Integration** – Built to run during your automated pipelines

---

## How It Works (Architecture)

```plaintext
           GitHub PR
               |
               v
        CI/CD Pipeline triggers
               |
               v
          Morbo CLI runs
               |
        ┌─────────────────────┐
        │  Fetch PR diff via  │
        │    GitHub API       │
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │  Send diff to LLM   │
        │ (DeepSeek on GCP)   │
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │ Receive summary text│
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │  Post comment to PR │
        └─────────────────────┘
```

---

## Quick Start

### 1. Clone the Repository

```
git clone https://github.com/Farnsworth-Enterprises/Morbo.git
cd Morbo
```

### 2. Build the Docker Container

```
docker build -t morbo .
```

### 3. Run Morbo (from CI or locally)

```
docker run \
  -e GITHUB_TOKEN=your_token \
  -e PR_URL=https://api.github.com/repos/user/repo/pulls/42 \
  -e LLM_API_URL=https://your-deepseek-endpoint.com/generate \
  morbo
```

Environment variable configuration is still in progress and will soon replace CLI input.
