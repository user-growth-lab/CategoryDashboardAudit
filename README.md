# CategoryDashboardAudit

Code and frozen aggregate evidence for **When Category Means Move: A
Paired-Support Audit of Recommendation Logs with Random Insertions**, accepted
at WSDM 2027.

**Authors:** Yu Zhang, Zhihan Wang, Yuhang Lu, and Shuai Li.
**Corresponding author:** Zhihan Wang.

- Paper DOI: <https://doi.org/10.1145/3862322.3863593>
- Versioned release: <https://github.com/user-growth-lab/CategoryDashboardAudit/releases/tag/v1.0.0>

The paper audits category-level engagement estimates in recommendation logs
using random insertions as a reference on shared user and video support. This
repository provides the analysis code, frozen aggregate outputs, and technical
appendix needed to inspect that evidence. The main paper is self-contained:
no claim requires a reader to visit this repository, run code, or obtain
external data. The included technical appendix is explanatory rather than a
source of assumptions needed by the main paper; its original anonymous review
version is preserved unchanged.

## Focused scope

The repository is limited to the P7--P14 evidence track:

- P7--P11 and P13--P14 code, tests, configurations, and frozen aggregate
  KuaiRand audit outputs;
- the P12 data-free known-truth stress test and its aggregate outputs;
- figures and machine-readable summaries derived from those stages; and
- `technical_materials/technical_appendix.pdf`, containing extended derivations,
  protocols, and aggregate audit tables.

The repository does not contain the earlier P0--P6 or KuaiRec exploration
branch.  It also excludes the main-paper PDF and source, raw interaction rows,
row-level derivatives, user or video identifier lists, model weights,
embeddings, caches, logs, local repository history, internal reviews, and
release-administration records.

## Install the focused environment

Create the environment outside the repository so that virtual-environment and
cache files cannot alter the released tree:

```bash
python3 -m venv ../category-audit-env
source ../category-audit-env/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip check
```

## Verify code and frozen-evidence logic

First verify the exact public file closure and every recorded SHA-256 using the
standard-library verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 verify_repository.py
```

The focused public tests then exercise the paired-support estimands and the
P7--P14 accounting and inference logic:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

## Data-free smoke run

P12 does not use an external dataset.  A small execution check can write only
to a disposable directory outside the repository:

```bash
XDG_CACHE_HOME=/tmp/category-audit-cache \
MPLCONFIGDIR=/tmp/category-audit-matplotlib \
PYTHONDONTWRITEBYTECODE=1 python3 code/p12_known_truth_benchmark.py \
  --replications 2 \
  --output ../p12-smoke.json --figure ../p12-smoke.png
```

This smoke run checks executability, not the precision or Monte Carlo scale of
the frozen paper benchmark.

## KuaiRand rerun boundary

P7--P11 and P13--P14 require independently downloaded and verified
KuaiRand-Pure logs plus the official KuaiRand category supplement.  Set the two
variables documented in `config.example.env`; no dataset is bundled here.

A full rerun is substantially more expensive than the tests and may replace
default result paths.  Run it only in a disposable copy of the repository after
verifying the upstream archive and category file:

```bash
set -a
source config.example.env
set +a
PYTHONDONTWRITEBYTECODE=1 python3 code/p7_overlap_diagnostics.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p8_paired_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p9_error_ledger.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p10_prospective_calibration.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p11_decision_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p12_known_truth_benchmark.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p13_dependence_sensitivity.py
PYTHONDONTWRITEBYTECODE=1 python3 code/p14_split_reference_decision.py
```

The frozen aggregate files preserve the paper's review snapshot. Cross-platform reruns
should reproduce the documented estimands and conclusions; byte identity of
new numerical outputs is not asserted across operating systems or numerical
library builds.

## Evidence and licensing boundary

KuaiRand random insertions provide a reference on the retained realized
category--video mix.  They do not randomize videos into categories, identify an
intrinsic category effect, or make the standard and random logs matched trials.
The P12 finite worlds are synthetic failure probes rather than a second
platform validation.

Original code and documentation are released under the MIT License in
`LICENSE`. Third-party data, dependencies, conference materials, and fonts
retain their own terms; see
`THIRD_PARTY_NOTICES.md` and `DATA_ACCESS_AND_RELEASE_BOUNDARY.md`.
