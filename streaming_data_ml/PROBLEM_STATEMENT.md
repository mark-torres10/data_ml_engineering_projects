# Problem Statement

- Monitor live social media posts to detect micro-trends—short-lived surges in hashtag activity—so content or moderation teams can act quickly.
- Streaming approach is necessary because trends emerge within minutes; Kafka provides durable, replayable streams, and Faust delivers stateful, per-hashtag anomaly detection in real time.

## Functional Requirements

- Generate and ingest synthetic post events with controllable burst patterns.
- Validate and enrich posts (normalize language, ensure sentiment) before downstream processing.
- Maintain rolling statistics per hashtag and flag anomalies when activity exceeds thresholds.
- Publish trend alerts with metadata (hashtag, spike magnitude, sentiment change) for consumers.
- Enable replay and reprocessing by retaining raw and enriched streams in Kafka topics.

## Nonfunctional Requirements

- Run entirely on a local developer machine without Docker; later support containerized deployment.
- Favor simplicity and observability for learning: clear logs, minimal dependencies, testable pure functions.
- Support restart resilience via state recovery (Faust tables) and replay capability (Kafka offsets).
- Keep latency low enough to detect spikes within seconds while balancing resource usage on modest hardware.
