# Streaming Data ML

## Quickstart

- Ensure Python 3.10+ is available.
- Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- From this directory, install dependencies with `uv sync`.

## Project Goals

- Explore Kafka + Faust for near-real-time micro-trend detection in social media streams.
- Keep everything runnable locally first, then optionally containerize.

## Useful Commands

- `uv run run-faust` – start the Faust worker (topology to be implemented).
- `uv run run-generator` – launch the synthetic post generator (to be added).
- `uv run run-tests` – execute the test suite once implemented.
