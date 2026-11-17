#!/usr/bin/env python3
"""Verify the focused anonymous reviewer repository using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Dict, Optional, Sequence, Set


TECHNICAL_PDF_PATH = "technical_materials/technical_appendix.pdf"
PRIMARY_SCOPE_ID = "4a262302e95474af4ac6d5c17b93625a943ad1e2f0692ae664a5fc76d75ac9f0"
EXPECTED_SCIENTIFIC_SNAPSHOT = {
    "primary_scope_id": PRIMARY_SCOPE_ID,
    "paired_users": 12472,
    "common_videos": 5486,
    "categories": 38,
    "p8_bootstrap_draws": 1000,
    "p9_bootstrap_draws": 1000,
    "p10_frozen_fit_users": 14912,
    "p10_returning_evaluation_users": 12342,
    "p10_bootstrap_draws": 1000,
    "p11_bootstrap_draws": 1000,
    "p12_replications": 100,
    "p12_users_per_replication": 1000,
    "p12_exposures_per_user": 60,
    "p12_categories": 38,
    "p13_observed_dates": 17,
    "p13_video_multiplier_draws": 500,
    "p14_bootstrap_draws": 1000,
    "p14_selection_rows": 233793,
    "p14_evaluation_rows": 234126,
}

EXPECTED_PAYLOAD: Set[str] = {
    ".gitignore",
    "pytest.ini",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "DATA_ACCESS_AND_RELEASE_BOUNDARY.md",
    "requirements.txt",
    "config.example.env",
    "verify_repository.py",
    *{
        f"code/{name}.py"
        for name in (
            "audit_metrics",
            "category_labels",
            "causal_baselines",
            "data_utils",
            "p7_overlap_diagnostics",
            "p8_paired_audit",
            "p9_error_ledger",
            "p10_prospective_calibration",
            "p11_decision_audit",
            "p12_known_truth_benchmark",
            "p13_dependence_sensitivity",
            "p14_split_reference_decision",
            "prepare_kuairand_categories",
        )
    },
    *{
        f"tests/{name}.py"
        for name in (
            "test_audit_metrics",
            "test_causal_baselines",
            "test_data_utils",
            "test_p8_paired_audit",
            "test_p9_error_ledger",
            "test_p10_prospective_calibration",
            "test_p11_decision_audit",
            "test_p12_known_truth_benchmark",
            "test_p13_dependence_sensitivity",
            "test_p14_split_reference_decision",
            "test_prepare_kuairand_categories",
        )
    },
    *{
        f"results/p{stage}_{suffix}.json"
        for stage, suffix in (
            (7, "overlap_diagnostics"),
            (8, "paired_audit"),
            (9, "error_ledger"),
            (10, "prospective_calibration"),
            (11, "decision_audit"),
            (12, "known_truth_benchmark"),
            (13, "dependence_sensitivity"),
            (14, "split_reference_decision"),
        )
    },
    *{
        f"figures/{name}.png"
        for name in (
            "fig8_paired_decomposition",
            "fig9_error_ledger",
            "fig10_prospective_calibration",
            "fig11_decision_audit",
            "fig12_known_truth_benchmark",
        )
    },
    TECHNICAL_PDF_PATH,
}
EXPECTED_FILES = EXPECTED_PAYLOAD | {"artifact_manifest.json", "MANIFEST.sha256"}

FORBIDDEN_DIRECTORY_PARTS = {".git", "data", "private", "raw", "review", "paper", "output"}
FORBIDDEN_SUFFIXES = {
    ".aux", ".bbl", ".blg", ".csv", ".fls", ".gz", ".log", ".npy",
    ".npz", ".out", ".tar", ".tex", ".tgz", ".zip",
}
TEXT_SUFFIXES = {"", ".env", ".json", ".md", ".py", ".txt"}
PDF_ACTIVE_TOKENS = (
    b"/javascript", b"/js ", b"/launch", b"/embeddedfile", b"/richmedia",
    b"/submitform", b"/importdata", b"/gotor",
)


def fail(message: str) -> None:
    raise SystemExit(f"verification failed: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative_path(name: str) -> PurePosixPath:
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        fail(f"unsafe relative path: {name}")
    return pure


def validate_forbidden_path(name: str) -> None:
    pure = validate_relative_path(name)
    folded_parts = tuple(part.casefold() for part in pure.parts)
    if any(part in FORBIDDEN_DIRECTORY_PARTS for part in folded_parts[:-1]):
        fail(f"forbidden directory entered closure: {name}")
    if pure.suffix.casefold() in FORBIDDEN_SUFFIXES:
        fail(f"forbidden suffix entered closure: {name}")
    if pure.name.casefold() == "main.pdf" or "kuairec" in name.casefold():
        fail(f"forbidden manuscript or KuaiRec path entered closure: {name}")
    historical = re.search(r"(?:^|_)p([0-6])(?:_|$)", pure.stem.casefold())
    if historical:
        fail(f"historical P0--P6 path entered closure: {name}")


def actual_files(root: Path) -> Set[str]:
    observed: Set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        validate_relative_path(relative)
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail(f"symlink is forbidden: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            fail(f"special file is forbidden: {relative}")
        observed.add(relative)
    return observed


def parse_manifest(root: Path, observed: Set[str]) -> Dict[str, object]:
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse artifact_manifest.json: {exc}")
    if manifest.get("schema_version") != 1:
        fail("unsupported manifest schema")
    if manifest.get("artifact_type") != "wsdm27_focused_anonymous_reviewer_repository":
        fail("unexpected artifact type")
    if manifest.get("allowed_files") != sorted(observed):
        fail("manifest allowlist does not equal actual closure")
    if manifest.get("file_count") != len(observed):
        fail("manifest file count drifted")
    if set(manifest.get("allowed_files", [])) != EXPECTED_FILES:
        fail("released file closure differs from the focused expected set")
    payload = manifest.get("payload_sha256")
    if not isinstance(payload, dict) or set(payload) != EXPECTED_PAYLOAD:
        fail("manifest payload hash closure drifted")
    for name, expected in payload.items():
        if not isinstance(expected, str) or len(expected) != 64:
            fail(f"malformed payload hash: {name}")
        if sha256_file(root / name) != expected:
            fail(f"payload hash mismatch: {name}")
    scope = manifest.get("content_scope")
    if not isinstance(scope, dict) or scope.get("stages_included") != [
        f"P{stage}" for stage in range(7, 15)
    ]:
        fail("P7--P14 content scope drifted")
    if any(
        scope.get(key) is not False
        for key in (
            "main_manuscript_pdf_included",
            "latex_sources_included",
            "raw_data_included",
        )
    ):
        fail("manifest privacy/content boundary drifted")
    return manifest


def verify_checksum_file(root: Path, observed: Set[str]) -> None:
    try:
        lines = (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"cannot read MANIFEST.sha256: {exc}")
    parsed: Dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            fail("malformed MANIFEST.sha256 line")
        digest, name = line.split("  ", 1)
        if name in parsed or not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail("duplicate or malformed checksum entry")
        validate_relative_path(name)
        parsed[name] = digest
    expected_names = observed - {"MANIFEST.sha256"}
    if set(parsed) != expected_names:
        fail("MANIFEST.sha256 closure drifted")
    for name, expected in parsed.items():
        if sha256_file(root / name) != expected:
            fail(f"checksum mismatch: {name}")


def verify_public_text(root: Path) -> None:
    patterns = (
        r"(?<![A-Za-z0-9:])/(?:Users|home)/[^\s\"'<>]+",
        r"[A-Za-z]:[\\/](?:Users)[\\/][^\s\"'<>]+",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b0000-000[0-9]-[0-9]{4}-[0-9]{3}[0-9X]\b",
    )
    for name in sorted(EXPECTED_PAYLOAD):
        if PurePosixPath(name).suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = (root / name).read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            fail(f"cannot decode public text {name}: {exc}")
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            fail(f"identity or local path found in public text: {name}")


def verify_pdf(root: Path) -> Dict[str, object]:
    raw = (root / TECHNICAL_PDF_PATH).read_bytes()
    if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-4096:]:
        fail("technical appendix is not a complete PDF")
    folded = raw.lower()
    if b"/encrypt" in folded:
        fail("technical appendix is encrypted")
    if any(token in folded for token in PDF_ACTIVE_TOKENS):
        fail("technical appendix contains active or external content")
    if b"anonymous" not in folded:
        fail("technical appendix lacks an anonymous author marker")
    for match in re.finditer(rb"/Author\s*\(([^)]*)\)", raw, flags=re.IGNORECASE):
        author = match.group(1).strip().lower()
        if author and b"anonymous" not in author:
            fail("technical appendix PDF metadata reveals an author")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "author_marker_anonymous": True,
        "encrypted": False,
        "active_or_external_content": False,
    }


def verify_structured_payload(root: Path) -> None:
    for name in sorted(EXPECTED_PAYLOAD):
        suffix = PurePosixPath(name).suffix.casefold()
        if name.startswith("results/") and suffix == ".json":
            try:
                json.loads((root / name).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"invalid result JSON {name}: {exc}")
        elif name.startswith("figures/"):
            if not (root / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                fail(f"invalid PNG signature: {name}")


def verify_scientific_snapshot(root: Path, manifest: Dict[str, object]) -> None:
    def load(name: str) -> Dict[str, object]:
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot read focused result {name}: {exc}")
        if not isinstance(value, dict):
            fail(f"focused result root is not an object: {name}")
        return value

    p7 = load("results/p7_overlap_diagnostics.json")
    p8 = load("results/p8_paired_audit.json")
    p9 = load("results/p9_error_ledger.json")
    p10 = load("results/p10_prospective_calibration.json")
    p11 = load("results/p11_decision_audit.json")
    p12 = load("results/p12_known_truth_benchmark.json")
    p13 = load("results/p13_dependence_sensitivity.json")
    p14 = load("results/p14_split_reference_decision.json")
    scopes = (
        p7["primary_paired_scope"],
        p8["paired_scope"],
        p11["paired_scope"],
        p14["paired_scope"],
    )
    ids = {scope["primary_scope_id"] for scope in scopes}
    ids.update(
        {
            p9["ledger_scope"]["canonical_paired_scope_id"],
            p13["scope"]["primary_scope_id"],
        }
    )
    if ids != {PRIMARY_SCOPE_ID}:
        fail("P7--P14 primary scope IDs drifted")
    final = scopes[1]["final"]
    if any(scope["final"] != final for scope in scopes):
        fail("paired scope records disagree across public results")
    snapshot = {
        "primary_scope_id": PRIMARY_SCOPE_ID,
        "paired_users": final["random"]["users"],
        "common_videos": final["random"]["videos"],
        "categories": final["random"]["categories"],
        "p8_bootstrap_draws": p8["main_paired_metrics_and_bootstrap"]["design"]["n_bootstrap"],
        "p9_bootstrap_draws": p9["paired_positive_multiplier_bootstrap"]["design"]["n_bootstrap"],
        "p10_frozen_fit_users": p10["primary"]["scope"]["frozen"]["users"],
        "p10_returning_evaluation_users": p10["primary"]["scope"]["evaluation"]["paired_users"],
        "p10_bootstrap_draws": p10["primary"]["paired_bootstrap"]["design"]["n_bootstrap"],
        "p11_bootstrap_draws": p11["design"]["n_bootstrap"],
        "p12_replications": p12["n_replications"],
        "p12_users_per_replication": p12["n_users_per_replication"],
        "p12_exposures_per_user": p12["n_exposures_per_user"],
        "p12_categories": p12["n_categories"],
        "p13_observed_dates": p13["scope"]["n_dates"],
        "p13_video_multiplier_draws": p13["shared_video_positive_multiplier_sensitivity"]["n_draws"],
        "p14_bootstrap_draws": p14["design"]["n_bootstrap"],
        "p14_selection_rows": p14["split_design"]["selection"]["rows"],
        "p14_evaluation_rows": p14["split_design"]["evaluation"]["rows"],
    }
    if snapshot != EXPECTED_SCIENTIFIC_SNAPSHOT:
        fail("focused scientific snapshot drifted")
    if manifest.get("scientific_snapshot") != snapshot:
        fail("manifest scientific snapshot drifted")
    if float(p9["maximum_point_path_reconstruction_error"]) > 1e-12:
        fail("P9 path reconstruction invariant failed")


def verify_repository(root: Path) -> Dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        fail(f"repository root is not a directory: {root}")
    observed = actual_files(root)
    if observed != EXPECTED_FILES:
        missing = sorted(EXPECTED_FILES - observed)
        extra = sorted(observed - EXPECTED_FILES)
        fail(f"focused closure drifted; missing={missing}, extra={extra}")
    for name in observed:
        validate_forbidden_path(name)
    manifest = parse_manifest(root, observed)
    verify_checksum_file(root, observed)
    verify_public_text(root)
    verify_structured_payload(root)
    verify_scientific_snapshot(root, manifest)
    pdf = verify_pdf(root)
    if manifest.get("technical_appendix") != pdf:
        fail("technical appendix manifest facts drifted")
    return {
        "status": "verified",
        "artifact_type": manifest["artifact_type"],
        "file_count": len(observed),
        "technical_appendix_sha256": pdf["sha256"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = verify_repository(parse_args(argv).root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
