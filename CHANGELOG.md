# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-08-19

### Added

- `recursive_get` and `recursive_post` to call APIs recursively.
- Tests for `ApicadabriCallArguments` that just specifies `url` and no other field as regression test for the infinite generator (see fixes).

### Changed

- Bumps version of GH actions used in the CI.
- Validates that `ApicadabriCallArguments` sets either `url` or `urls`.

### Fixed

- Specifying an `ApicadabriCallArguments` instance that only provides single values for all fields with `zip` mode yielded an infinite iterator (because it zipped four `repeat()` generators).
- Moves away from yanked version `3.11.14` of `aiohttp`.

## [0.5.1] - 2025-05-19

### Fixed

- Type hints for `**kwargs` now properly use `Any` instead of `dict[str, Any]`.

## [0.5.0] - 2025-05-18

### Added

- Allows tracking progress through `tqdm` and the new `tee` method.
- Missing examples and overview of main features and design patterns in README.

### Fixed

- `AsyncRetrier` now tries for `max_retries + 1` times before failing to keep consistent with the phrasing of REtries.

## [0.4.0] - 2025-05-12

### Added

- Docstrings for all functions and classes.
- `bulk_post` function.
- Allows to forward kwargs to aiohttp functions.

### Changed

- `reduce` now raises its own `ApicadabriReduceError`.

### Fixed

- One reference to the `product` mode was called `multiply`.
- Uses `ConfigDict` instead of deprecated `class Config`.

### Removed

- Old `pipeline` mode that wasn't implemented anyway.

## [0.3.0] - 2025-05-08

### Added

- `AsyncRetrier` class that allows to specify retry settings and retry any async callable.
- `retrier` parameter for `bulk_call` and `bulk_get`.

## [0.2.0] - 2025-05-06

### Added

- `ApicadabriBulkResponse` base class that can be used to run Apicadabri for arbitrary async tasks that don't have to involve `aiohttp`.

## [0.1.0] - 2025-05-04

### Added

- First release of the project, containing base functionality to bulk call APIs asynchronously with basic error handling.

[Unreleased]: https://github.com/CSchoel/apicadabri/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/CSchoel/apicadabri/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/CSchoel/apicadabri/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/CSchoel/apicadabri/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/CSchoel/apicadabri/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CSchoel/apicadabri/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CSchoel/apicadabri/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CSchoel/apicadabri/releases/tag/v0.1.0
