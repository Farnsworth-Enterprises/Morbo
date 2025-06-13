from github import Github, Auth
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
import logging
import requests
from requests.exceptions import RequestException


class GitHubConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GitHubConnection, cls).__new__(cls)
            cls._instance._github = None
        return cls._instance

    def __init__(self):
        if self._github is None:
            load_dotenv()
            auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
            self._github = Github(auth=auth)

    def get_github(self):
        return self._github

    def close(self):
        if self._github:
            self._github.close()
            self._github = None


# Create a global instance
github = GitHubConnection()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_pr_info(repo_name, pr_number):
    """
    Retrieves structured information about a pull request that can be used to generate a PR summary.

    Args:
        pr_number (int): The number of the pull request to analyze

    Returns:
        dict: A dictionary containing structured PR information including:
            - title: PR title
            - description: PR description
            - author: PR author
            - created_at: PR creation date
            - base_branch: Target branch
            - head_branch: Source branch
            - files_changed: List of files with their changes
            - total_changes: Summary of total changes
    """
    g = github.get_github()

    try:
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        files_changed = []
        for file in pr.get_files():
            file_info = {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch
            }
            files_changed.append(file_info)

        total_changes = {
            "additions": sum(f["additions"] for f in files_changed),
            "deletions": sum(f["deletions"] for f in files_changed),
            "files_changed": len(files_changed)
        }

        pr_info = {
            "title": pr.title,
            "description": pr.body,
            "author": pr.user.login,
            "created_at": pr.created_at.isoformat(),
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "files_changed": files_changed,
            "total_changes": total_changes
        }

        return pr_info
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        raise e


def generate_pr_summary(repo_name, pr_number):
    """
    Generates a PR summary using the LLM based on PR information.

    Args:
        pr_number (int): The number of the pull request to analyze

    Returns:
        dict: A dictionary containing:
            - title: The PR title (max 72 characters)
            - description: The PR description in markdown format
    """
    pr_info = get_pr_info(repo_name, pr_number)

    combined_diff = "\n\n".join([
        f"File: {file['filename']}\nStatus: {file['status']}\n{file['patch']}"
        for file in pr_info['files_changed']
    ])

    prompt = PromptTemplate(
        template="""You are a PR message generator. Your task is to analyze the PR changes and return ONLY a JSON object. Do not include any other text, explanations, or markdown formatting in your response.

Return this exact JSON structure:
{{
    "title": "PR Title (max 72 characters)",
    "description": "PR Description in markdown format"
}}

PR Context:
- Title: {title}
- Total Changes: {total_changes}

Changes:
```
{diff}
```

Guidelines:
1. Title: Be specific about the type of change, avoid generic titles
2. Description: Focus on why and how instead of what changed, use bullet points for changes with short descriptions, avoid using full sentences.
3. IMPORTANT: Return ONLY the JSON object, no additional text, no explanations, no markdown formatting outside the JSON""",
        input_variables=["title", "total_changes", "diff"]
    )

    ollama_url = os.getenv("OLLAMA_URL")
    logger.debug(f"Connecting to Ollama at URL: {ollama_url}")

    try:
        # Test connection to Ollama server with longer timeout and specific headers
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Python/3.x'
        }
        session = requests.Session()
        session.headers.update(headers)
        test_response = session.get(
            f"{ollama_url}/api/tags",
            timeout=30,  # Increased timeout
            verify=True
        )
        test_response.raise_for_status()
        logger.debug("Successfully connected to Ollama server")
    except RequestException as e:
        logger.error(f"Failed to connect to Ollama server: {str(e)}")
        raise

    model = OllamaLLM(
        base_url=ollama_url,
        model="deepseek-r1:8b",
        temperature=0.0,
        format="json",
        timeout=120,
        headers=headers  # Pass the same headers to the OllamaLLM
    )

    formatted_prompt = prompt.format(
        title=pr_info['title'],
        total_changes=f"{pr_info['total_changes']['additions']} additions, {pr_info['total_changes']['deletions']} deletions across {pr_info['total_changes']['files_changed']} files",
        diff=combined_diff
    )

    logger.debug("Sending request to Ollama")
    try:
        response = model.invoke(input=formatted_prompt)
        logger.debug("Received response from Ollama")
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {
                "title": pr_info['title'],
                "description": response
            }
    except Exception as e:
        logger.error(f"Error during Ollama request: {str(e)}")
        raise


def update_pr(repo_name, pr_number, summary):
    g = github.get_github()

    try:
        g.get_repo(repo_name).get_pull(pr_number).edit(
            title=summary['title'], body=summary['description'])

        print(f"PR {pr_number} updated successfully")
        return True
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <repo_name> <pr_number>")
        sys.exit(1)

    try:
        repo_name = sys.argv[1]
        pr_number = int(sys.argv[2])
        summary = generate_pr_summary(repo_name, pr_number)
        update_pr(repo_name, pr_number, summary)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    finally:
        github.close()
