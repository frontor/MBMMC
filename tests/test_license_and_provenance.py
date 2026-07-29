from pathlib import Path


def test_polyform_license_metadata() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "PolyForm Noncommercial License 1.0.0" in license_text
    assert "Required Notice:" in license_text
    assert "PolyForm-Noncommercial-1.0.0" in citation
    assert "noncommercial" in readme.lower()
    assert "BSD 3-Clause" not in license_text


def test_crossnn_provenance_files() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    provenance = Path("docs/CROSSNN_METHOD_PROVENANCE.md").read_text(encoding="utf-8")
    trainer = Path("mbmmc/train_crossnn.py").read_text(encoding="utf-8")

    assert "10.1038/s43018-025-00976-5" in notices
    assert "independent" in provenance.lower()
    assert "not an official release" in trainer.lower()
