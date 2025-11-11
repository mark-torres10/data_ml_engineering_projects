# Implementation Plan

## Step 1: Define project scaffold and dependencies

### Why are we doing this?
Set a consistent structure, lock dependencies with uv, and capture core settings so downstream modules import from one source of truth.

### What do we need to implement?

#### File 1: pyproject.toml
- What does this file do? Declares project metadata and uv-managed dependencies.
- What do you need to add? Project name, Python version (>=3.10), dependencies (`faust-streaming`, `aiokafka`, `pydantic`, `python-dotenv` optional, `pytest` for tests). Register uv scripts (`run-faust`, `run-generator`, `run-tests`).
- Things to consider for further exploration and enrichment. Add optional extras (e.g., `transformers` later). Configure tool settings (ruff, mypy) when ready.

#### File 2: streaming_data_ml/__init__.py
- What does this file do? Marks package, can expose version constants.
- What do you need to add? Empty file or `__all__`.
- Things to consider for further exploration and enrichment. Populate with `__version__` later.

#### File 3: streaming_data_ml/config.py
- What does this file do? Centralizes application settings.
- What do you need to add? `Settings` class (dataclass or pydantic) with broker URL, topic names, window sizes, thresholds; ability to load from env via `dotenv`.
- Things to consider for further exploration and enrichment. Support CLI overrides; add config validation.

#### File 4: README.md
- What does this file do? Onboarding instructions.
- What do you need to add? uv install instructions, quickstart, topology summary.
- Things to consider for further exploration and enrichment. Include architecture diagram, troubleshooting section.

### What does success look like?
`uv sync` installs dependencies, directories exist, config importable, README explains workflow.

### What tests do we need to add?
No automated tests yet; manual verification via `uv run` commands.


## Step 2: Model schemas and shared types

### Why are we doing this?
Consistent serialization across generator, processors, and tests; facilitates validation and static typing.

### What do we need to implement?

#### File 1: streaming_data_ml/schemas.py
- What does this file do? Defines data models for events.
- What do you need to add? `PostEvent`, `EnrichedPost`, `TrendAlert` via pydantic; helper to convert from Kafka messages (bytes) to model.
- Things to consider for further exploration and enrichment. Add sentiment enum, topic confidence scores.

#### File 2: streaming_data_ml/state.py (optional but recommended)
- What does this file do? Encapsulates rolling stats structures.
- What do you need to add? `RollingStat` dataclass/struct with methods `update(value)`, `z_score(value)`, TTL management.
- Things to consider for further exploration and enrichment. Persist state snapshot to disk; integrate Count-Min Sketch for scale.

### What does success look like?
Schemas validated in isolation; creating instances from sample payloads works; processors can import these classes.

### What tests do we need to add?
`tests/test_schemas.py` verifying serialization/deserialization, validation errors, rolling stats math.


## Step 3: Implement data generator

### Why are we doing this?
Need a controllable local stream to feed Kafka with synthetic social posts exhibiting trend bursts.

### What do we need to implement?

#### File 1: streaming_data_ml/generators/social_post_stream.py
- What does this file do? Produces synthetic posts and publishes to Kafka.
- What do you need to add? Async producer using `aiokafka`; load seed from `data/seed_posts.csv`; logic for topic spikes; CLI entry (uv script).
- Things to consider for further exploration and enrichment. Parameterize spike frequency; allow writing to stdout for dry-run; add CLI arguments.

#### File 2: data/seed_posts.csv
- What does this file do? Base data for generator.
- What do you need to add? Minimal dataset with timestamps, user ids, text, hashtags, sentiment.
- Things to consider for further exploration and enrichment. Expand languages, include ground-truth trending labels.

### What does success look like?
Running `uv run generate-posts` publishes events to `posts.raw`; logs confirm bursts.

### What tests do we need to add?
`tests/test_generator.py` with unit tests on deterministic helper functions (e.g., spike pattern) without hitting Kafka; integration test later with Kafka fixture.


## Step 4: Build Faust pipeline skeleton

### Why are we doing this?
Establish streaming topology connecting Kafka topics to Faust agents, ensuring modular structure.

### What do we need to implement?

#### File 1: streaming_data_ml/pipeline.py
- What does this file do? Instantiates Faust app, declares topics, wires agents.
- What do you need to add? `app = faust.App(...)`, topic definitions (`posts_raw`, `posts_enriched`, `posts_trends`), agent registrations referencing processor functions, table declarations.
- Things to consider for further exploration and enrichment. Introduce serializers, CLI options, error handling hooks.

#### File 2: streaming_data_ml/processors/__init__.py
- What does this file do? Provides package namespace.
- What do you need to add? Import convenience or leave blank.
- Things to consider for further exploration and enrichment. Export high-level functions for easier testing.

#### File 3: streaming_data_ml/processors/ingest.py
- What does this file do? Cleans and validates raw posts.
- What do you need to add? Functions to normalize language, ensure sentiment; agent function referencing Faust Table/streams.
- Things to consider for further exploration and enrichment. Integrate external sentiment service; quality metrics.

#### File 4: streaming_data_ml/processors/feature.py
- What does this file do? Extracts hashtags/topics, updates rolling stats.
- What do you need to add? Functions for hashtag parsing, table update (one table per hashtag).
- Things to consider for further exploration and enrichment. Advanced NLP, mention detection, cross-language support.

#### File 5: streaming_data_ml/processors/trend.py
- What does this file do? Calculates anomaly scores and emits alerts.
- What do you need to add? `score_anomaly` using rolling stats, threshold config, output to `TrendAlert`.
- Things to consider for further exploration and enrichment. Adaptive thresholds, sentiment drift detection.

### What does success look like?
Running Faust worker logs agent activity, moves messages through topics, and emits alerts when generator creates bursts.

### What tests do we need to add?
- `tests/test_ingest_processor.py`: unit tests for normalization.
- `tests/test_feature_processor.py`: unit tests for hashtag extraction and stats updates.
- `tests/test_trend_processor.py`: ensure scoring triggers in expected scenarios.
- Optional Faust app test using in-memory channels to validate pipeline wiring (`tests/test_pipeline_smoke.py`).


## Step 5: Support utilities and scripts

### Why are we doing this?
Improve ergonomics: easy topic creation, manual monitoring, reproducible command flow.

### What do we need to implement?

#### File 1: scripts/create_topics.py
- What does this file do? Idempotently creates Kafka topics.
- What do you need to add? KafkaAdminClient usage with settings from config; CLI arguments for partitions/replication.
- Things to consider for further exploration and enrichment. Add deletion or describe commands.

#### File 2: scripts/run_local_pipeline.py (optional)
- What does this file do? Orchestrates generator + worker for dev convenience.
- What do you need to add? CLI that sequentially ensures topics exist, optionally launches generator; instruct user to run Faust separately.
- Things to consider for further exploration and enrichment. Use `asyncio` to run generator and consumer side-by-side; integrate logging config.

#### File 3: scripts/read_trends.py
- What does this file do? Simple consumer to print alerts.
- What do you need to add? Kafka consumer reading from `posts.trends`, pretty-print alerts.
- Things to consider for further exploration and enrichment. Write to SQLite or serve via FastAPI.

### What does success look like?
Developers can run `uv run create-topics`, `uv run generate-posts`, `uv run read-trends` to observe pipeline without manual Kafka commands.

### What tests do we need to add?
Basic tests for helper functions where feasible (e.g., ensure topic list matches config); integration tests once Dockerized kafka is in CI eventually.


## Step 6: Testing and validation foundation

### Why are we doing this?
Ensure reliability via automated tests; establish fixtures and CI-ready structure before expanding.

### What do we need to implement?

#### File 1: tests/conftest.py
- What does this file do? Shared fixtures for sample events, rolling stats, config overrides.
- What do you need to add? Fixtures for `PostEvent`, `RollingStat`; optional fixture for temporary Faust app in unit scope.
- Things to consider for further exploration and enrichment. Add pytest markers for slow/integration.

#### File 2: tests/test_processors.py (aggregator or split per module)
- What does this file do? Validates processor logic end-to-end without Kafka.
- What do you need to add? Tests for ingest normalization, feature extraction, anomaly scoring.
- Things to consider for further exploration and enrichment. Add property-based tests (hypothesis) to catch edge cases; test TTL behavior.

### What does success look like?
`uv run pytest` passes; tests cover core deterministic logic.

### What tests do we need to add?
As above; later add integration tests with embedded Kafka (or Redpanda) and eventually load/latency checks.


## Step 7: Local execution docs and manual verification

### Why are we doing this?
Document the commands required to run everything locally (non-Docker), ensuring onboarding is smooth.

### What do we need to implement?

#### File 1: README.md (update)
- What does this file do? Guides users.
- What do you need to add? Step-by-step: start local Kafka (assume user uses established instructions or mention brew install), run uv commands, expected outputs, troubleshooting.
- Things to consider for further exploration and enrichment. Add diagrams, gif of alert output, Docker instructions later.

### What does success look like?
Someone can follow README to install deps, start Kafka, run generator + worker, and see alerts printed.

### What tests do we need to add?
Manual verification checklist:
- Sensor generator publishes events (check Kafka topic length via `kafka-console-consumer`).
- Faust worker emits alerts for synthetic bursts.
- Scripts handle missing topics gracefully.

(Automated acceptance tests can be added later with Dockerized Kafka in CI.)


## Step 8: Optional Dockerization (deferred)

### Why are we doing this?
Run same setup in containers for reproducibility, but only after local workflow is stable.

### What do we need to implement?

#### File 1: docker-compose.yml
- What does this file do? Orchestrates Kafka/ZooKeeper (or Redpanda) + Faust worker + generator.
- What do you need to add? Services definitions, environment variables, `depends_on`, volumes.
- Things to consider for further exploration and enrichment. Add UI tools (Kafka UI); handle graceful shutdown.

#### File 2: Dockerfile
- What does this file do? Builds project image.
- What do you need to add? Base Python image, `uv` install, `uv sync --frozen`, run command for Faust worker.
- Things to consider for further exploration and enrichment. Multi-stage builds, dev vs prod images.

### What does success look like?
`docker-compose up` spins brokers and worker; alerts visible via logs, matching local behavior.

### What tests do we need to add?
Manual smoke test via Docker; later add CI workflow running `docker compose run tests`.


## High-Detail Walkthrough

- **Raw events**: Synthetic `PostEvent` payloads generated every second, with occasional bursts that repeat hashtags over short windows to simulate emerging micro-trends.
- **Enriched events**: Validated and normalized posts (canonicalized hashtags, language normalization, fallback sentiment) emitted downstream once the ingest agent processes raw data.
- **Trend alerts**: Compact notifications describing the hashtag/topic, spike magnitude versus baseline, sentiment shift, and pointers to representative posts so downstream systems can act quickly.
- **Rolling statistics per hashtag**: Faust tables keyed by hashtag maintain rolling means, standard deviations, or EWMA counts and sentiment, enabling per-key anomaly detection.
- **Scoring**: Each enriched post updates the hashtag’s state; when the z-score or EWMA deviation exceeds configured thresholds (with cooldown throttling), an alert is emitted to `posts.trends`.
- **Windowing**: State tables can implement tumbling windows (e.g., 60-second buckets) or decayed counters to focus on recent activity while discarding stale data.
- **Kafka role**: `posts.raw`, `posts.enriched`, and `posts.trends` topics provide durable, partitioned streams; keying by hashtag keeps all relevant events on the same partition, ensuring consistent per-key state and replayability.
- **Faust role**: Declares the topology, manages consumer groups, and persists tables via changelog topics; agents (`ingest`, `feature`, `trend`) are coroutine-based processors that maintain state and emit downstream events; scaling is achieved by running additional workers which Kafka rebalances across partitions.
- **Learning focus**: Observe consumer offsets, partition assignment, table recovery after restarts, exactly-once vs at-least-once semantics, and how stateful stream processors differ from batch pipelines.


## Step-by-Step: Local Kafka (macOS/Linux)

1. **Install Java**  
   - Ensure Java 11+ exists via `java -version`; install (e.g., `brew install openjdk@21`) if absent, then re-check.

2. **Download Kafka binary**  
   - Fetch Kafka (e.g., 3.7.0) using `curl -O ...`, extract with `tar -xzf`, and move to `~/kafka`.

3. **Initialize KRaft storage**  
   - Generate a cluster ID: `KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"`.  
   - Format logs: `bin/kafka-storage.sh format -t "$KAFKA_CLUSTER_ID" -c config/kraft/server.properties`.

4. **Start the broker**  
   - Run `bin/kafka-server-start.sh config/kraft/server.properties` and leave the terminal open.

5. **Create required topics**  
   - In a new terminal:  
     ```
     bin/kafka-topics.sh --create --topic posts.raw --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
     bin/kafka-topics.sh --create --topic posts.enriched --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
     bin/kafka-topics.sh --create --topic posts.trends --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
     ```

6. **Smoke test produce/consume**  
   - Producer: `bin/kafka-console-producer.sh --topic posts.raw --bootstrap-server localhost:9092` and send a sample line.  
   - Consumer: `bin/kafka-console-consumer.sh --topic posts.raw --bootstrap-server localhost:9092 --from-beginning` to verify.

7. **Integrate with Python stack**  
   - Use `localhost:9092` as the bootstrap server.  
   - Run generator via `uv run generators/social_post_stream.py`.  
   - Start Faust worker: `uv run faust -A streaming_data_ml.pipeline worker -l info`.

8. **Operational notes**  
   - Stop Kafka with `Ctrl+C`; reuse the same command to restart.  
   - Reformat storage only if you delete logs.  
   - Adjust ports or log directories by editing `config/kraft/server.properties`.

9. **Next-level options**  
   - Experiment with Redpanda for a single-binary Kafka alternative.  
   - Add a Kafka UI (e.g., provectuslabs/kafka-ui) once comfortable.  
   - Later explore security (SSL/SASL) and multi-broker setups.
