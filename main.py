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
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Configuration constants
MAX_DIFF_SIZE = 100000
EXCLUDED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.ico',
                       '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mov', '.wav', '.mp3'}


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
            logger.info("Initializing GitHub connection...")
            auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
            self._github = Github(auth=auth)
            logger.info("GitHub connection established successfully")

    def get_github(self):
        return self._github

    def close(self):
        if self._github:
            logger.info("Closing GitHub connection...")
            self._github.close()
            self._github = None
            logger.info("GitHub connection closed")


github = GitHubConnection()


def get_pr_info(repo_name, pr_number):
    """
    Retrieves structured information about a pull request that can be used to generate a pull request summary.

    Args:
        pr_number (int): The number of the pull request to analyze

    Returns:
        dict: A dictionary containing structured pull request information including:
            - title: Pull request title
            - body: Pull request body
            - created_at: Pull request creation date
            - base_branch: Target branch
            - head_branch: Source branch
            - files_changed: List of files with their changes
            - total_changes: Summary of total changes
    """
    start_time = time.time()

    g = github.get_github()

    try:
        logger.info(f"Accessing repository: {repo_name}")
        repo = g.get_repo(repo_name)
        logger.info(f"Repository accessed successfully")

        logger.info(f"Retrieving pull request #{pr_number}")
        pr = repo.get_pull(pr_number)
        logger.info(f"Pull request #{pr_number} retrieved successfully")

        files_changed = []
        total_size = 0
        binary_files_skipped = 0
        files_without_patch = 0
        files_excluded_by_size = 0

        logger.info("Retrieving changed files...")
        all_files = list(pr.get_files())
        logger.info(f"Found {len(all_files)} total files in pull request")

        for file in all_files:
            if any(file.filename.lower().endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                logger.info(f"Skipping binary file: {file.filename}")
                binary_files_skipped += 1
                continue

            if not file.patch:
                logger.info(f"Skipping file without patch: {file.filename}")
                files_without_patch += 1
                continue

            file_size = len(file.patch.encode('utf-8'))
            if total_size + file_size > MAX_DIFF_SIZE:
                logger.warning(
                    f"Diff size limit reached. Skipping remaining files.")
                files_excluded_by_size += 1
                break

            logger.info(
                f"Processing {file.filename}: {file.status}, +{file.additions}/-{file.deletions} lines ({file_size} bytes)")

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

        elapsed_time = time.time() - start_time
        logger.info(
            f"Pull request info retrieval completed in {elapsed_time:.2f} seconds")
        logger.info(
            f"Processed {len(files_changed)} files with {total_changes['additions']} additions and {total_changes['deletions']} deletions")
        logger.info(f"Total diff size processed: {total_size} bytes")
        logger.info(f"Binary files skipped: {binary_files_skipped}")
        logger.info(f"Files without patch: {files_without_patch}")
        logger.info(f"Files excluded by size limit: {files_excluded_by_size}")

        if len(files_changed) == 0:
            logger.warning(
                "No text-based files were found in the pull request. This could be because:")
            logger.warning("1. All files are binary files")
            logger.warning("2. All files are too large")
            logger.warning("3. No files have patches")
            logger.warning("4. The pull request is empty")

        pr_info = {
            "title": pr.title,
            "body": pr.body,
            "files_changed": files_changed,
            "total_changes": total_changes
        }

        return pr_info
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Error retrieving pull request information after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        raise e


def generate_pr_summary(repo_name, pr_number):
    """
    Generates a pull request summary using the LLM based on pull request information.

    Args:
        pr_number (int): The number of the pull request to analyze

    Returns:
        dict: A dictionary containing:
            - title: The pull request title (max 72 characters)
            - body: The pull request body in markdown format
    """
    pr_info = get_pr_info(repo_name, pr_number)

    def sanitize_patch(patch):
        if not patch:
            return ""
        patch = ''.join(char for char in patch if ord(char)
                        >= 32 or char == '\n')

        patch = patch.replace('\\', '\\\\').replace(
            '"', '\\"').replace('\n', '\\n')
        return patch

    combined_diff = "\n\n".join([
        f"File: {file['filename']}\nStatus: {file['status']}\n{sanitize_patch(file['patch'])}"
        for file in pr_info['files_changed']
    ])

    if not combined_diff.strip():
        logger.warning("No diff content available for LLM processing")
        return {
            "title": pr_info['title'],
            "body": "No files were changed in this pull request. Please review manually."
        }

    prompt = PromptTemplate(
        template="""[SYSTEM: You are a Pull Request (PR) Title and body generator. Your task is to UPDATE the PR title and body based on the changes. Return ONLY a JSON object with exactly two fields: "title" and "body".]

PR Context:
- Current Title: {title}
- Total Changes: {total_changes}

Changes:
```
{diff}
```

[SYSTEM: You must generate a NEW title that is different from the current title. The new title should be more descriptive of the actual changes. Return ONLY this JSON object:]
{{
    "title": "string (max 72 chars, must be a new descriptive title)",
    "body": "string (markdown formatted)"
}}

[SYSTEM: Example of valid response, return ONLY the JSON object:]
{{
    "title": "Add user authentication system",
    "body": "- Implemented JWT-based authentication\\n- Added login and registration endpoints\\n- Created user model and database migrations"
}}

# [SYSTEM: IMPORTANT RULES:]
1. You MUST generate a NEW title that is different from the current title
2. The new title must be more descriptive of the actual changes
3. Do not include any notes or explanations
4. Do not include any text before or after the JSON
5. Return ONLY the JSON object
6. Do not include any comments about the title
7. Do not include any suggestions
""",
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
        model=model_name,
        temperature=0.0,
        format="json",
        timeout=120,
        headers=headers,

        system="""[SYSTEM: You are a PR message generator that ONLY outputs valid JSON.
Your task is to UPDATE the PR title and body based on the changes.

Your response must be a single JSON object with EXACTLY these two fields:

{
    "title": "string (max 72 chars, must be a new descriptive title)",
    "body": "string (markdown formatted)"
}


CRITICAL RULES:
1. You MUST generate a NEW title that is different from the current title
2. The new title must be more descriptive of the actual changes
3. Return ONLY the JSON object
4. Do not include any other fields
5. Do not include any other text
6. Do not include any explanations
7. Do not include any nested objects
8. The response must be parseable JSON
9. Both fields must be non-empty strings
10. Do not include any other fields like 'summary', 'output', or 'steps'
11. Do not include any text before or after the JSON object
12. Do not include any comments or explanations
13. Do not include any system messages
14. Do not include any user messages
15. Do not include any other JSON objects
16. Do not include any notes about the title
17. Do not include any suggestions
18. The title must be a new descriptive title, not the original title

Example of valid response:
{
    "title": "Add user authentication system",
    "body": "- Implemented JWT-based authentication\\n- Added login and registration endpoints\\n- Created user model and database migrations"
}""",

        num_predict=1024
    )

    formatted_prompt = prompt.format(
        title=pr_info['title'],
        total_changes=f"{pr_info['total_changes']['additions']} additions, {pr_info['total_changes']['deletions']} deletions across {pr_info['total_changes']['files_changed']} files",
        diff=combined_diff
    )

    logger.info("Generating pull request summary with LLM...")
    llm_start_time = time.time()
    try:
        response = model.invoke(input=formatted_prompt)
        llm_elapsed_time = time.time() - llm_start_time
        logger.info(f"LLM response received in {llm_elapsed_time:.2f} seconds")

        try:
            response = response.strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                response = response[start:end]

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

            required_fields = ['title', 'body']
            missing_fields = [
                field for field in required_fields if field not in summary]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields: {missing_fields}")

            if not isinstance(summary['title'], str) or not isinstance(summary['body'], str):
                raise ValueError(
                    f"Invalid field types in response")

            return summary
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error("---START OF FAILED RESPONSE---")
            logger.error(response)
            logger.error("---END OF FAILED RESPONSE---")
            logger.warning(
                f"Using original pull request title due to parsing error: {pr_info['title']}")
            return {
                "title": pr_info['title'],
                "body": f"Error parsing AI response. Original pull request title preserved.\n\nRaw AI response:\n{response}"
            }
    except Exception as e:
        llm_elapsed_time = time.time() - llm_start_time
        logger.error(
            f"Error during Ollama request after {llm_elapsed_time:.2f} seconds: {str(e)}")
        raise


def update_pr(repo_name, pr_number, summary):
    g = github.get_github()

    try:
        logger.info(
            f"Updating pull request #{pr_number} with generated title and body")
        logger.info(f"New title: {summary['title']}")
        logger.info(f"New body:\n{summary['body']}")

        pr = g.get_repo(repo_name).get_pull(pr_number)

        pr.edit(title=summary['title'], body=summary['body'])
        logger.info(f"Pull request {pr_number} updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating pull request: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    try:
        repo_name = os.environ.get('REPO_NAME')
        pr_number = os.environ.get('PR_NUMBER')

        if not repo_name or not pr_number:
            logger.error(
                "Missing required environment variables: REPO_NAME and PR_NUMBER")
            sys.exit(1)

        pr_number = int(pr_number)
        logger.info(
            f"Processing pull request #{pr_number} in repository {repo_name}")

        github_token = os.environ.get('GITHUB_TOKEN')
        ollama_url = os.environ.get('OLLAMA_URL')
        model_name = os.environ.get('MODEL_NAME', 'deepseek-r1:8b')

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
        logger.info(
            f"Pull request summary generation and update completed successfully for PR #{pr_number}")
