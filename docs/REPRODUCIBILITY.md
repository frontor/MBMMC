# Reproducibility checklist

## Before training

- Freeze cohort definitions, labels, exclusions, and splitting unit.
- Define patient, study, platform, batch, or simulation grouping.
- Register the candidate set and primary endpoint.
- Record data accessions, versions, and SHA-256 checksums.
- Keep the independent test set inaccessible during tuning.

## During training

- Run from a clean Git commit.
- Preserve commit hash and working-tree status.
- Preserve commands, resolved arguments, logs, and run manifests.
- Record hardware, operating system, Python, dependencies, threads, seeds, and runtime.
- Record failed or excluded runs and reasons.

## Reporting and release

- Separate development/CV and independent-test results.
- Report per-class and macro metrics, calibration, and no-call behavior where relevant.
- Report the number of candidates and selection rules.
- Do not tune thresholds on the final test cohort.
- Replace publication placeholders and run release checks.
- Create an immutable release and archive it with a DOI.
- Cite repository URL, release number, commit hash, DOI, and access date.

## Data governance

Do not publish protected health information, linkable identifiers, credentials,
consent-restricted molecular data, or controlled-access files.
