from github import Github, Auth
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
import logging
import requests
from requests.exceptions import RequestException
import sys


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


github = GitHubConnection()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

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
        total_size = 0
        MAX_DIFF_SIZE = 100000  # 100KB limit for total diff size
        EXCLUDED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.ico',
                               '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mov', '.wav', '.mp3'}

        for file in pr.get_files():
            # Skip binary files and large files
            if any(file.filename.lower().endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                logger.info(f"Skipping binary file: {file.filename}")
                continue

            # Skip files without a patch (e.g., binary files)
            if not file.patch:
                logger.info(f"Skipping file without patch: {file.filename}")
                continue

            # Check if adding this file would exceed the size limit
            file_size = len(file.patch.encode('utf-8'))
            if total_size + file_size > MAX_DIFF_SIZE:
                logger.warning(
                    f"Diff size limit reached. Skipping remaining files.")
                break

            file_info = {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch
            }
            files_changed.append(file_info)
            total_size += file_size

        total_changes = {
            "additions": sum(f["additions"] for f in files_changed),
            "deletions": sum(f["deletions"] for f in files_changed),
            "files_changed": len(files_changed)
        }

        pr_info = {
            "title": pr.title,
            "description": pr.body,
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

    # Sanitize and format the diff content
    def sanitize_patch(patch):
        if not patch:
            return ""
        # Remove any null bytes and control characters
        patch = ''.join(char for char in patch if ord(char)
                        >= 32 or char == '\n')
        # Escape any special characters that could interfere with JSON
        patch = patch.replace('\\', '\\\\').replace(
            '"', '\\"').replace('\n', '\\n')
        return patch

    combined_diff = "\n\n".join([
        f"File: {file['filename']}\nStatus: {file['status']}\n{sanitize_patch(file['patch'])}"
        for file in pr_info['files_changed']
    ])

    # If no valid files were found, return a default response
    if not combined_diff.strip():
        return {
            "title": pr_info['title'],
            "description": "No text-based changes found in this PR. Please review manually."
        }

    prompt = PromptTemplate(
        template="""Analyze these PR changes and return a JSON object.

PR Context:
- Title: {title}
- Total Changes: {total_changes}

Changes:
```
{diff}
```

Return ONLY a JSON object with these exact fields:
{{
    "title": "PR Title (max 72 characters)",
    "description": "PR Description in markdown format"
}}

Example response:
{{
    "title": "Add user authentication system",
    "description": "- Implemented JWT-based authentication\\n- Added login and registration endpoints\\n- Created user model and database migrations"
}}

Guidelines:
1. Title: Be specific about the type of change
2. Description: Use bullet points for changes
3. Return ONLY the JSON object
4. Do not include any other text
5. Do not nest the response under any other fields
6. Do not use markdown headers or formatting
7. The response must be valid JSON""",
        input_variables=["title", "total_changes", "diff"]
    )

    ollama_url = os.getenv("OLLAMA_URL")

    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Python/3.x'
        }
        session = requests.Session()
        session.headers.update(headers)
        test_response = session.get(
            f"{ollama_url}/api/tags",
            timeout=30,
            verify=True
        )
        test_response.raise_for_status()
        logger.info("Successfully connected to Ollama server")
    except RequestException as e:
        logger.error("Failed to connect to Ollama server")
        raise

    model = OllamaLLM(
        base_url=ollama_url,
        model="deepseek-r1:8b",
        temperature=0.0,
        format="json",
        timeout=120,
        headers=headers,
        system="""You are a PR message generator that ONLY outputs valid JSON.
Your response must be a single JSON object with exactly these fields:
{
    "title": "string (max 72 chars)",
    "description": "string (markdown formatted)"
}
Do not include any other text, explanations, or fields.
Do not use markdown headers or formatting in the response.
The response must be parseable JSON.""",
        num_predict=1024
    )

    formatted_prompt = prompt.format(
        title=pr_info['title'],
        total_changes=f"{pr_info['total_changes']['additions']} additions, {pr_info['total_changes']['deletions']} deletions across {pr_info['total_changes']['files_changed']} files",
        diff=combined_diff
    )

    logger.info("Generating PR summary...")
    try:
        response = model.invoke(input=formatted_prompt)
        logger.info("Successfully generated PR summary")
        try:
            logger.debug(f"Raw response: {response}")

            summary = json.loads(response)
            if not isinstance(summary, dict):
                raise ValueError(
                    f"Response is not a dictionary. Raw response: {response}")

            if 'summary' in summary:
                summary = summary['summary']

            logger.debug(f"Processed summary: {summary}")

            if not summary:
                raise ValueError("Received empty response from Ollama")

            required_fields = ['title', 'description']
            missing_fields = [
                field for field in required_fields if field not in summary]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields: {missing_fields}. Raw response: {response}")

            if not isinstance(summary['title'], str) or not isinstance(summary['description'], str):
                raise ValueError(
                    f"Invalid field types in response. Raw response: {response}")

            return summary
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {
                "title": pr_info['title'],
                "description": f"Error parsing AI response. Original PR title preserved.\n\nRaw AI response:\n{response}"
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
    try:
        repo_name = os.environ.get('REPO_NAME')
        pr_number = int(os.environ.get('PR_NUMBER'))

        if not repo_name or not pr_number:
            logger.error(
                "Missing required environment variables: REPO_NAME and PR_NUMBER")
            sys.exit(1)

        github_token = os.environ.get('GITHUB_TOKEN')
        ollama_url = os.environ.get('OLLAMA_URL')

        if not github_token or not ollama_url:
            logger.error(
                "Missing required environment variables: GITHUB_TOKEN and OLLAMA_URL")
            sys.exit(1)

        summary = generate_pr_summary(repo_name, pr_number)
        update_pr(repo_name, pr_number, summary)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        github.close()
