# Contribution guide

Contributions to this project are always welcome.
Just message me, create an issue or directly open a PR on a forked version of the project.
Whatever works for you.

This document currently mainly contains small code snippets to remind myself how to do things, but if it turns out that other people want to contribute, it will be updated into a more comprehensive guide.

## Deploy a new version

1. Create a feature branch.
2. Open a PR.
3. Update the `CHANGELOG.md`.
4. Update the version number.
   - Manually in `pyproject.toml`.
   - With `keepachangelog release X.Y.Z` for the changelog.
5. Merge the PR.
6. `git checkout main`
7. `git pull`
8. `git tag vX.Y.Z`
9. `git push origin vX.Y.Z`
10. Add your (Test)PyPI credentials to a `.env` file.
11. Execute the following:

    ```bash
    source .env
    uv build
    uv publish --index testpypi  # for testing
    uv publish                   # the real deal
    ```
12. Test that everything works with `uv run --with apicadabri --no-project -- python -c "import apicadabri; print('Everything okay.')"`