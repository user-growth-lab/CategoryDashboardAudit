# Data Access and Release Boundary

KuaiRand-Pure and the predicted-category supplement are publicly obtainable
from their upstream records.  Their public availability does not grant this
public repository permission to republish them under the license for original
project code and documentation.

## Included in the focused repository

- P7--P11 and P13--P14 analysis code, public tests, configurations, and frozen
  aggregate KuaiRand JSON, tables, and figures.
- P12 known-truth simulation code, tests, and aggregate outputs.
- Aggregate provenance and omission records needed to interpret the frozen
  evidence.
- One prebuilt anonymous technical-appendix PDF with derivations, protocols,
  diagnostics, and aggregate tables.

## Not redistributed

- KuaiRand or KuaiRec raw interaction logs, metadata tables, archives, or
  caption files.
- The complete KuaiRand category supplement or its derived Pure-ID row subset.
- User IDs, video IDs, row-level outcomes, row-level weights or predictions,
  bootstrap multiplicity arrays, embeddings, fitted models, or model weights.
- The P0--P6 exploratory branch and all KuaiRec-dependent result files.
- The main-paper PDF or source, LaTeX build products, runtime logs, caches,
  local paths, repository history, internal reviews, or release-administration
  records.

## Verification levels

1. **Data-free inspection:** readers can read the paper and technical
   appendix, inspect every distributed aggregate, and run the focused unit tests
   without downloading data.
2. **Data-free known-truth rerun:** P12 can be rerun independently because its
   finite worlds are generated from declared parameters.
3. **KuaiRand rerun:** P7--P11 and P13--P14 require the reader to download the
   official KuaiRand inputs, verify the complete published digests before
   parsing, and configure the paths locally.

The KuaiRand archive MD5 is the upstream trust anchor because official
member-level checksums are not published.  Project-recorded member SHA-256
values identify the analyzed inputs but do not replace verification of the
official archive.  A checksum mismatch is a hard stop: do not extract, analyze,
or substitute the mismatched bytes.

Downloaded data and rerun outputs must remain outside the released repository.
This omission does not narrow what a reader can read in the self-contained
paper and technical appendix; it limits only optional end-to-end execution.
