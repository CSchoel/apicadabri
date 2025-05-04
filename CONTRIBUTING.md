# Contribution guide

Contributions to this project are always welcome.
Just message me, create an issue or directly open a PR on a forked version of the project.
Whatever works for you.

This document currently mainly contains small code snippets to remind myself how to do things, but if it turns out that other people want to contribute, it will be updated into a more comprehensive guide.

## Deploy a new version

1. Create a feature branch.
2. Open a PR.
3. Merge the PR.
4. Update the `CHANGELOG.md`.
   - Don't forget to also update the links at the bottom.
5. Update the version number in `pyproject.toml`
6. `git checkout main`
7. `git tag vX.Y.Z`
8. `git push vX.Y.Z`
9. Add your (Test)PyPI credentials to a `.env` file.
10. Execute the following:

    ```bash
    source .env
    uv build
    uv publish --index testpypi  # for testing
    uv publish                   # the real deal
    ```
