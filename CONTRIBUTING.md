# Contribution guide

Contributions to this project are always welcome.
Just message me, create an issue or directly open a PR on a forked version of the project.
Whatever works for you.

This document currently mainly contains small code snippets to remind myself how to do things, but if it turns out that other people want to contribute, it will be updated into a more comprehensive guide.

## Deploy a new version

First, add your (Test)PyPI credentials to a `.env` file.
Then execute the following:

```bash
source .env
uv build
uv publish --index testpypi  # for testing
uv publish                   # the real deal
```
