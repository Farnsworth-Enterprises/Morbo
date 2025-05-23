from github import Github
from dotenv import load_dotenv
import os
# Authentication is defined via github.Auth
from github import Auth

load_dotenv()
# using an access token
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

# First create a Github instance:

# Public Web Github
g = Github(auth=auth)

# Github Enterprise with custom hostname
# g = Github(base_url="https://github.com/Farnsworth-Enterprises/Morbo", auth=auth)

# Then play with your Github objects:
# for repo in g.get_user().get_orgs()
#     print(repo.name)

morbo = g.get_repo("Farnsworth-Enterprises/Morbo")

# print open PR
# print(list(morbo.get_pulls(state="open")))

# create a comment on a PR
# morbo.get_pull(14).create_issue_comment("Hello from Python!")


# prints all comments for a PR
# for comment in morbo.get_pull(14).get_issue_comments():
#     print(comment.body)

# prints files changed
# for file in morbo.get_pull(14).get_files():
#     print(file.raw_data)

# prints files changed
# print([file for file in morbo.get_pull(14).get_files()])

# prints the patch(diff) for a file
# print(morbo.get_pull(14).get_files()[2].status)

#  prints all files as objects with context data
for file in morbo.get_pull(14).get_files():
    file = {
        "filename": file.filename,
        "status": file.status,
        "additions": file.additions,
        "deletions": file.deletions,
        "changes": file.changes,
        "diff": file.patch
    }
    print(file, "\n\n")

# To close connections after use
g.close()
