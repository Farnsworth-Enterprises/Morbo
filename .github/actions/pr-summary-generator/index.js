const core = require("@actions/core");
const { exec } = require("child_process");
const util = require("util");
const execPromise = util.promisify(exec);

async function run() {
	try {
		const githubToken = core.getInput("github-token");
		const ollamaUrl = core.getInput("ollama-url");
		const modelName = core.getInput("model-name") || "deepseek-r1:8b";
		const repoName = process.env.GITHUB_REPOSITORY;
		const prNumber = process.env.GITHUB_EVENT_PULL_REQUEST_NUMBER;

		// Set up Python 3.11
		await execPromise("sudo apt-get update");
		await execPromise("sudo apt-get install -y python3.11 python3.11-venv");
		await execPromise("python3.11 -m pip install --upgrade pip");

		// Create and activate virtual environment
		await execPromise("python3.11 -m venv venv");
		process.env.PATH = `${process.cwd()}/venv/bin:${process.env.PATH}`;

		// Install dependencies
		const dependencies = [
			"langchain==0.3.25",
			"PyGithub==2.6.1",
			"python-dotenv==1.1.0",
			"langchain-community==0.3.24",
			"langchain-ollama==0.3.3",
			"langchain-core==0.3.61",
			"langchain-text-splitters==0.3.8",
			"ollama==0.4.8",
		];
		await execPromise(`pip install ${dependencies.join(" ")}`);

		// Set environment variables
		process.env.GITHUB_TOKEN = githubToken;
		process.env.OLLAMA_URL = ollamaUrl;
		process.env.REPO_NAME = repoName;
		process.env.PR_NUMBER = prNumber;
		process.env.MODEL_NAME = modelName;

		// Run the Python script
		await execPromise(`python main.py "${repoName}" "${prNumber}"`);
	} catch (error) {
		core.setFailed(error.message);
	}
}

run();
