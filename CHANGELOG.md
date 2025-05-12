# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/CSchoel/apicadabri/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/CSchoel/apicadabri/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CSchoel/apicadabri/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CSchoel/apicadabri/releases/tag/v0.1.0
