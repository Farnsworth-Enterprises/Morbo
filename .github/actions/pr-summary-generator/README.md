# PR Summary Generator Action

This GitHub Action automatically generates summaries for pull requests using Ollama LLM. It analyzes the changes in a PR and creates a concise summary of the modifications.

## Features

-   Automatically generates PR summaries when PRs are opened or updated
-   Uses Ollama LLM for intelligent summarization
-   Easy to integrate into any repository

## Usage

```yaml
name: PR Summary Generator

on:
    pull_request:
        types: [opened, synchronize]

jobs:
    generate-summary:
        runs-on: ubuntu-latest
        permissions:
            contents: read
            pull-requests: write

        steps:
            - name: Checkout repository
              uses: actions/checkout@v4

            - name: Generate PR Summary
              uses: Farnsworth-Enterprises/Morbo/.github/actions/pr-summary-generator@main
              with:
                  github-token: ${{ secrets.GITHUB_TOKEN }}
                  ollama-url: ${{ secrets.OLLAMA_URL }}
                  model-name: "deepseek-r1:8b" # Optional
```

## Inputs

| Input          | Required | Default          | Description                     |
| -------------- | -------- | ---------------- | ------------------------------- |
| `github-token` | Yes      | -                | GitHub token for authentication |
| `ollama-url`   | Yes      | -                | URL for Ollama service          |
| `model-name`   | No       | "deepseek-r1:8b" | Name of the Ollama model to use |

## Required Secrets

-   `OLLAMA_URL`: The URL of your Ollama service

## Example Output

The action will post a comment on the PR with a summary of the changes, including:

-   Overview of changes
-   Key modifications
-   Impact of the changes

## License

MIT License
