# Turbofan Engine Predictive Maintenance System

Predicting Remaining Useful life (RUL) for aircraft turbofan engines from sensor
telemetry, and surfacing those predictions in a fleet-monitoring dashboard that ranks engine by
urgency to support maintenance scheduling decisions.

Built on NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
turbofan degradation dataset.

---

## Problem Statement

Aircraft engines degrades gradually over their operating life, and unplanned in-serivce
failures are far more expensive and dangerous than scheduled maintenance. The C-MAPSS
dataset simulates this degradation: each engine units runs from healthy
operation to failure,. recording 21 sensor channels and 3 operational settings at every
operating cycle.

**What is being predicted:** the Remaining Useful Life (RUL) of an engine, the number
of operating cycles left before failure, guessed from the current and historical sensor readings.
Another task would be a binary classification: Would this engine need maintenance soon
(i.e., is the RUL below an acceptable operational threshold?)

**Why it matters:** in a real fleet-maintenance settings, a model like this supports a shift
from fixed-interval maintenance (replace parts on a set schedule regardless of actual wear)
to condition-based maintenance (replacing parts where data suggests they're actually close
to failing). This redues unnecessary part replacements while also reducing the risk on
in-service failure.

---

## Dataset
- Source: [NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)
- Subset used: FD001 (single operating condition, single fault mode). FD002 - FD004 introduces multiple operating conditions and/or faults modes and are stretch goals
- 100 training engines run to failure,, 100 test engines truncated before failure
- 21 sensor channels + 3 operational settings per cycle
- Test set ground truth: `RUL_FDOOX.txt` gives the true RUL at the last recorded cycle of each test engine only.
---
