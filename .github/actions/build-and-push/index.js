const core = require("@actions/core");
const { exec } = require("child_process");
const util = require("util");
const execPromise = util.promisify(exec);

async function run() {
	try {
		const dockerfile = core.getInput("dockerfile");
		const githubToken = core.getInput("github-token");
		const imageName =
			core.getInput("image-name") || process.env.GITHUB_REPOSITORY;

		// Set up Docker Buildx
		await execPromise("docker buildx create --use");

		// Login to GitHub Container Registry
		await execPromise(
			`echo "${githubToken}" | docker login ghcr.io -u ${process.env.GITHUB_ACTOR} --password-stdin`
		);

		// Build and push the Docker image
		await execPromise(
			`docker buildx build --push -f ${dockerfile} -t ghcr.io/${imageName}:latest .`
		);

		core.setOutput("image", `ghcr.io/${imageName}:latest`);
	} catch (error) {
		core.setFailed(error.message);
	}
}

run();
