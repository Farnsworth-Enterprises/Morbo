<p align="center">
  <img src="https://michaelgmunz.com/wp-content/uploads/2016/09/morbo-doom.png">
</p>

# Morbo - PR Description Generator

**Morbo** is an AI-powered GitHub Action that automatically generates descriptive Pull Request (PR) summaries using Ollama-powered LLM. Designed to integrate directly with your CI/CD pipeline, Morbo detects code changes introduced in a PR, generates a concise, informative summary utilizing the diff, and posts it directly to the PR — so developers can focus on writing code, not explaining it.

## Features

-   Automatic PR summary generation using Ollama LLM
-   Triggers on PR open and update events
-   Easy integration with any repository
-   Real-time PR updates

## What It Does

Whenever a pull request is opened or updated, **Morbo**:

1. Pulls the diff from the PR using GitHub's API
2. Sends the diff to your Ollama server for processing
3. Receives a short, human-readable summary of the changes
4. Automatically updates the PR with the generated summary

## Tech Stack

-   **Python** – Core application logic
-   **GitHub Actions** – For CI/CD integration
-   **Google Cloud Platform** – Hosting infrastructure
-   **Docker** – Containerized Ollama deployment
-   Ollama container for LLM serving
-   DeepSeek-R1:8B model for text generation
-   **GitHub API** – For fetching diffs and updating PRs

## How It Works (Architecture)

```plaintext
           GitHub PR
               |
               v
        CI/CD Pipeline triggers
               |
               v
          Morbo Action runs
               |
        ┌─────────────────────┐
        │  Fetch PR diff via  │
        │    GitHub API       │
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │  Send diff to LLM   │
        │    via Ollama       │
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │ Receive summary text│
        └─────────────────────┘
               |
               v
        ┌─────────────────────┐
        │  Update PR with     │
        │    new summary      │
        └─────────────────────┘
```

## Usage

Add this to your repository's `.github/workflows/pr-summary.yml`:

```yaml
name: Generate PR Description

on:
    pull_request:
        types: [opened, synchronize]

jobs:
    generate-pr-description:
        runs-on: ubuntu-latest
        steps:
            - name: Generate PR Description
              uses: Farnsworth-Enterprises/Morbo@latest
              with:
                  github-token: ${{ secrets.GITHUB_TOKEN }}
                  ollama-url: ${{ secrets.OLLAMA_URL }}
```

## Required Secrets

You need to set up the following secrets in your repository:

1. `OLLAMA_URL`: The URL of your Ollama server (e.g., 'http://localhost:11434')
2. `GITHUB_TOKEN`: This is automatically provided by GitHub Actions

To set up secrets:

1. Go to your repository settings
2. Navigate to "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Add the required secrets

## Inputs

| Input          | Required | Description                                                        |
| -------------- | -------- | ------------------------------------------------------------------ |
| `github-token` | Yes      | GitHub token for authentication. Use `${{ secrets.GITHUB_TOKEN }}` |
| `ollama-url`   | Yes      | URL of your Ollama server (e.g., 'http://localhost:11434')         |
| `model-name`   | No       | Name of the Ollama model to use. Defaults to 'deepseek-r1:8b'      |
| `temperature`  | No       | Temperature for the AI model (0.0 to 1.0). Defaults to '0.0'       |

## Requirements

-   An Ollama server running with the specified model
-   GitHub repository with pull request access

## Example Output

The action will generate a summary in this format:

```markdown
# Title: Add User Authentication System

## Description

-   Implemented JWT-based authentication
-   Added user registration endpoint
-   Created login functionality
-   Added password hashing
-   Updated API documentation
```

## Authors:

- [Luis Gonzalez](https://github.com/zluigon)
- [Jordan Biehl](https://github.com/jbiehl88)
