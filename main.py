from github import Github, Auth
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
import logging
import requests
from requests.exceptions import RequestException
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pr_info(github_token):
    """
    Retrieves structured information about the current pull request.
    """
    auth = Auth.Token(github_token)
    g = Github(auth=auth)

    try:
        # Get repository and PR information from environment variables
        repo_name = os.environ.get('REPO_NAME')
        pr_number = int(os.environ.get('PR_NUMBER'))

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
            "files_changed": files_changed,
            "total_changes": total_changes
        }

        return pr_info, g
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise e


def generate_pr_summary(pr_info, ollama_url, model_name, temperature):
    """
    Generates a PR summary using the LLM based on PR information.
    """
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
2. Description: Focus on what changed, you must use bullet points for changes with short descriptions, avoid using full sentences.
3. IMPORTANT: Return ONLY the JSON object, no additional text, no explanations, no markdown formatting outside the JSON""",
        input_variables=["title", "total_changes", "diff"]
    )

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
        logger.error(f"Failed to connect to Ollama server: {str(e)}")
        raise

    model = OllamaLLM(
        base_url=ollama_url,
        model=model_name,
        temperature=float(temperature),
        format="json",
        timeout=120,
        headers=headers
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


def update_pr(github, summary):
    """
    Updates the PR with the generated summary.
    """
    try:
        repo_name = os.environ.get('REPO_NAME')
        pr_number = int(os.environ.get('PR_NUMBER'))

        github.get_repo(repo_name).get_pull(pr_number).edit(
            title=summary['title'],
            body=summary['description']
        )

        logger.info(f"PR {pr_number} updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating PR: {str(e)}")
        raise e


def main():
    try:
        github_token = os.environ.get('GITHUB_TOKEN')
        ollama_url = os.environ.get('OLLAMA_URL')
        model_name = os.environ.get('MODEL_NAME', 'deepseek-r1:8b')
        temperature = os.environ.get('TEMPERATURE', '0.0')

        if not all([github_token, ollama_url]):
            logger.error("Missing required environment variables")
            sys.exit(1)

        pr_info, github = get_pr_info(github_token)
        summary = generate_pr_summary(
            pr_info, ollama_url, model_name, temperature)
        update_pr(github, summary)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'github' in locals():
            github.close()


if __name__ == "__main__":
    main()
