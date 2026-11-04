# Known Limitations

This project is an ML research/demo pipeline, not a production IDS. The
models score well on held-out UNSW-NB15 traffic, but several properties
limit what those numbers mean in the real world. Numbers below refer to
the headline protocol (`official_novel`, see `reports/rigorous_eval.md`)
unless stated otherwise.

## 0. The original headline numbers were inflated by the split

The v1 README reported RF accuracy 0.951 / macro-F1 0.948 and LSTM AUC
0.998. Both came from a random re-split of the combined UNSW-NB15 table:

* ~40% of the combined rows are exact duplicates, so many "test" flows
  had byte-identical twins in training;
* the official training and testing files were captured separately; a
  random re-split hides that distribution shift;
* the LSTM's windows overlap (stride 5 < window 10) and were split at
  random, so the same flows sat in training *and* test windows.

Re-running the same model under leakage-aware protocols
(`src/evaluation/rigorous_eval.py`) gives macro-F1 0.948 -> 0.917
(dedup) -> 0.862 (official split) -> **0.818** (official split, unseen
flows only). FPR at the 0.5 threshold goes from 5% to 26.5%. The new
numbers are the honest ones; `--split random` still reproduces the old.

Related finding: a threshold chosen on validation for FPR <= 1% yields
**5% FPR on test**. Validation comes from the training capture, test
from another, and the operating point does not transfer. In a real
deployment, calibrate thresholds on traffic from the network you defend.

## 1. Adversarial evasion risk (transparency cuts both ways)

Random Forest feature importance is part of this repo's value:
`reports/feature_importances.csv` tells *you* which features drive
detection - and it tells an attacker exactly the same thing. The most
influential features (Sload, sbytes, dbytes, sttl, ...) are all
attributes an attacker can manipulate within a crafted session. An
adaptive adversary can infer "lower Sload and sbytes, keep sttl small,
use common ports" and drive attack flows toward the normal class:

* classical evasion via feature perturbation,
* transfer attacks: a separate surrogate model on public UNSW-NB15
  mirrors this classifier's decision boundary closely enough to mine
  evasion samples offline,
* the LSTM is a black box, which complicates *interpretation*, but the
  same input-space attack works: perturb flow features to shift window
  embeddings.

This is now measured rather than asserted: `src/attacks/evasion.py`
runs a black-box, budget-constrained evasion against the deployed RF, a
no-TTL variant and an adversarially trained variant
(`reports/evasion.md`). Headline: choosing the initial TTL alone evades
the deployed RF on 100% of detected attacks; without TTL features,
padding/slowing still evades 21-64%; adversarial training holds it to
<= 4% for this perturbation family. Mitigations: feature-difficulty hardening (drop
features an attacker trivially controls), adversarial training, and
never relying on a single ML model at the boundary.

## 2. Fuzzers and Analysis evade detection (stable, real weak spots)

At the 0.5 threshold every family but one is caught >= 0.97 of the
time, but 0.5 costs a 26.5% false-positive rate. At the deployed
operating threshold (0.854, validation FPR <= 1%) the weak spots show:
**Fuzzers 0.27** and **Analysis 0.80** recall, against >= 0.88 for every
other family. The v1 random-split evaluation showed the same two
families as weakest (0.72 / 0.88). They survived every fix, so they are
a real property of per-flow UNSW-NB15 classifiers, not a bug. Do not
hide them: they are the most honest part of the evaluation.

Both families are instructive as adversarial-adjacent cases:

* **Fuzzers** generate malformed/mutated traffic whose statistical
  profile is engineered to resemble benign sessions (small packets,
  modest byte counts, common ports). Fuzzing is designed to look benign,
  and an anomaly-flagging detector is exactly the wrong tool for a
  family built to be unremarkable - this is benign mimicry, the same
  tactic an adaptive adversary uses against deployed ML IDS. Closing
  the gap needs signals beyond per-flow statistics: payload analysis,
  protocol conformance checking, and mutation correlation across
  sessions.
* **Analysis** (port scans, spam probes, HTML/iframe injection) is
  typically sparse and low-volume, so individual probe flows look
  "mostly normal" to a per-flow model. Detection improves with window/
  sequence context (the LSTM's strength) and cross-flow correlation
  rather than per-flow classification.

Practical takeaway: reported aggregate accuracy of ~0.95 on UNSW-NB15
must always be read alongside family-level recall, because the families
an attacker actually deploys (fuzzers, stealthy probes) are the ones the
aggregate number hides.

## 3. Pipeline-version pinning (the RF/pipeline desync incident)

An audit of this repo's retraining guard in Aug 2026 uncovered a
deployment-class bug. It is documented here because it is exactly the
kind of failure that happens in production systems:

**What happened (timeline):**

1. Preprocessing was re-run several times during development (feature
   fixes, leakage removal). Each run regenerates `data/processed/*`,
   the scaler and the metadata - the pipeline's canonical state.
2. The Random Forest was *not* re-run after the last preprocessing
   change, so `models/random_forest_best.joblib` kept a model trained
   on an older `X_train.csv` while every other artifact in the repo
   described the newer pipeline.
3. The mismatch is silent (no file hashes, no version stamp), so
   evaluation, the API, the dashboard and the README kept reporting
   the stale model as current: RF test accuracy 0.9372 instead of the
   fair 0.9514.
4. The retraining script exposed the problem: with zero new data it
   still proposed replacing the model, because its compare-and-replace
   guard compared a freshly trained model (new pipeline) against the
   stale incumbent (old pipeline). A secondary bug found in the same
   audit - the guard's parameter reconstruction dropped the grid-search
   result `max_depth=None` and silently substituted `max_depth=25` -
   would have corrupted any future replacement.

**What it implies:** a model artifact is meaningless without the
pipeline version that produced it. In a real deployment:

* record the pipeline git SHA + input dataset hash in
  `preprocess_metadata.json` (or a model registry) at every training
  run, and refuse to load a model whose pipeline stamp mismatches the
  serving pipeline;
* re-validate the model whenever preprocessing changes - a short
  re-fit on the current splits catches the desync before it reaches
  any report;
* keep the retrain guard honest: with zero new rows it must refuse,
  bit-for-bit (this is now covered by a regression test in this repo).

Implemented since: every training step writes `models/manifest.json`
(SHA-256 of each artifact, split protocol, git commit) and the API
refuses to start if any artifact it loads does not match
(`src/models/manifest.py`, `tests/test_serving.py::test_tampered_model_refuses_to_load`).

## 4. Concept drift

The models are static snapshots of UNSW-NB15 (2015 synthetic captures).
Real networks drift: new protocols (QUIC by default, ECH/TLS 1.3),
work-from-home IP mixes, cloud-native ephemeral ports, and IoT traffic
change the feature distribution the model was calibrated on. Symptoms:

* rising false-positive rate on legitimate traffic (the classifier
  flags "unusual" flows that are merely new),
* silent recall loss on novel attack tooling (novel *tools* reuse old
  *patterns*, but novel *patterns* - e.g. DNS-over-HTTPS exfil - are
  seen as normal).

`src/models/retrain.py` is the manual response: harvest live
predictions (confirmed labels or high-confidence pseudo-labels), retrain,
and only swap the model when it beats the incumbent on the fixed
held-out test set. For a production deployment, pair this with drift
detection (e.g. monitoring the served confidence distribution and
per-feature KS statistics) and a scheduled retrain cadence.

## 5. ML complements, it does not replace, signatures

Everything in this repo is anomaly/behavior learning. Signature- and
rule-based systems (Suricata/Snort rules, YARA, IOC lists) remain the
right tool for *known* threats: exact C2 domains, hashes, exploit
patterns, GeoIP/ASN blocking. ML is strongest where signatures are
weakest (novel or polymorphic behavior), but:

* the model cannot name the threat precisely: the family classifier
  reaches macro-F1 0.63 (DoS / Analysis / Backdoor / Shellcode are
  routinely confused), so the ATT&CK mapping is a triage hint, not an
  attribution;
* false alarms on legitimate-but-rare traffic are inherent to
  unsupervised-adjacent learning and cannot be tuned away by signatures;
* UNSW-NB15 itself is synthetic; real-world traffic differs in ways the
  test set cannot expose.

Recommended posture: **signatures first, ML as the second layer**, with
the ML output feeding analyst workflows (below), not auto-blocking.

## 6. From a /predict alert to SOC action

The API returns `{prediction_id, prediction, p_attack, threshold,
model_version, alert}`; `alert` (see `src/soc/alerts.py`) already
carries severity, the family classifier's guess, the ATT&CK mapping and
the top drivers. A realistic SOC integration would use it as follows:

1. **Enrichment** - join the flow record with asset inventory, user
   identity (via NetFlow/Zeek sidecar), and reputation/GeoIP lookups;
   attach `attack_family_probs` and `method` (rf vs lstm) for context.
2. **Severity mapping from P(attack)** (implemented in
   `src/soc/alerts.py::severity`). v1 documented this table against
   `confidence = max(p, 1-p)`, under which a *confidently benign* flow
   would have mapped to HIGH. Severity must key off the attack
   probability:

   | p_attack | suggested severity |
   |---|---|
   | threshold-0.80 | LOW - info/triage queue |
   | 0.80-0.95 | MEDIUM - ticket |
   | >= 0.95 | HIGH - immediate ticket + page |
   | lstm window | route to the window's constituent flows for review |

3. **Ticket creation** - the alert becomes a ticket in the SIEM/SOAR
   (e.g. TheHive/Jira), with the raw flow record attached so the analyst
   has the evidence without re-querying the sensor.
4. **Human-in-the-loop review** - an analyst confirms or dismisses the
   alert. The confirmation feeds back: confirmed attacks and dismissed
   false positives are *labels*, which (a) train the severity
   thresholds and (b) become ground truth rows in the prediction log
   via the analyst-only `/feedback` endpoint, the only label path
   `retrain.py` trusts. This closes the loop:
   alerts -> labels -> better model -> fewer false alarms.
5. **Escalation** - only confirmed high-confidence alerts should trigger
   blocking actions; everything below the review threshold stays
   advisory, because a wrong auto-block is a self-inflicted DoS.

## Data caveats specific to this repo

* The downloadable "cleaned" UNSW-NB15 variant has no srcip/dstip/ports/
  timestamps, so the LSTM groups flows by `(proto, service)` in file
  order rather than by real connection identity - the sequence signal is
  weaker than a canonical-data deployment would provide.
* The traffic simulator replays held-out test flows; retraining dedupes
  them against the base dataset, but in production, ensure logged flows
  carry a flow ID so label feedback can never leak the test set.
* Per-family "precision" in `reports/model_comparison.md` is
  one-vs-rest precision: rare families always look poor because most
  attack alarms belong to Generic. Use the **recall** column for
  detection coverage and the confusion matrix for alarm quality.
