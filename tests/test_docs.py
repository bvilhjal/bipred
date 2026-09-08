"""Integrity checks for the documentation, in Markdown and in comments.

Every local link and anchor in the primary Markdown must resolve. Source
files must cite symbols rather than line numbers, which rot silently.
"""

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DOCS = (
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    *sorted((ROOT / "docs").rglob("*.md")),
)
REPOSITORY_DOCS = (
    ROOT / "benchmarks" / "README.md",
    ROOT / "benchmarks" / "RESULTS.md",
    ROOT / "research" / "cross_corr_estimation" / "RESULTS_REGIONAL.md",
)
PRIMARY_DOCS = tuple(p for p in PACKAGE_DOCS + REPOSITORY_DOCS if p.exists())
# Archived benchmark runs snapshot the tree as it was; they are records, not
# maintained sources, and are excluded along with the caches beside them.
ARCHIVED = ("results", ".runs", ".work", ".ld_cache")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
EXPLICIT_ANCHOR_RE = re.compile(r'<a\s+(?:id|name)=["\']([^"\']+)["\']')
# A cross-reference into a source file by line number. Written as a pattern so
# this module is not itself an instance of what it forbids.
SOURCE_LINE_REF_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*" + r"\.py" + r":"
                                + r"[0-9]+(?:-[0-9]+)?")


def _read(path):
    """Read a repository document as UTF-8, whatever the platform codepage is."""
    return path.read_text(encoding="utf-8")


def _markdown_anchors(path):
    """Return the GitHub-style heading anchors used by these documents."""
    counts = {}
    text = _read(path)
    anchors = set(EXPLICIT_ANCHOR_RE.findall(text))
    for heading in HEADING_RE.findall(text):
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"[`*~]", "", heading).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).strip().replace(" ", "-")
        occurrence = counts.get(slug, 0)
        counts[slug] = occurrence + 1
        anchors.add(slug if occurrence == 0 else f"{slug}-{occurrence}")
    return anchors


_SELF_REPO_RE = re.compile(
    r"^https://github\.com/bvilhjal/bipred/(?:blob|tree)/"
    r"(?:master|main|v[\d.]+)/(.*)$")


def _resolve_self_repo(target):
    """Repo-relative path for a link into this repository, else None."""
    match = _SELF_REPO_RE.match(target)
    return match.group(1) if match else None


def test_primary_markdown_links_and_anchors_resolve():
    failures = []
    checked = 0
    for source in PRIMARY_DOCS:
        for raw_target in LINK_RE.findall(_read(source)):
            target = raw_target.strip().strip("<>")
            self_repo = _resolve_self_repo(target)
            if self_repo is not None:
                if not (ROOT / ".git").exists():
                    continue
                target = self_repo
                source_dir = ROOT
            elif target.startswith(("http://", "https://", "mailto:")):
                continue
            else:
                source_dir = source.parent
            checked += 1
            target = unquote(target.split()[0])
            path_text, _, fragment = target.partition("#")
            destination = (
                (source_dir / path_text).resolve() if path_text else source
            )
            if not destination.exists():
                failures.append(f"{source.relative_to(ROOT)} -> {target}")
            elif fragment and destination.suffix.lower() == ".md":
                if fragment not in _markdown_anchors(destination):
                    failures.append(
                        f"{source.relative_to(ROOT)} -> {target} (missing anchor)"
                    )
    assert not failures, "broken primary documentation links:\n" + "\n".join(failures)
    minimum = 2 * len(PRIMARY_DOCS)
    assert checked >= minimum, (
        f"link checker reached {checked} links; expected at least {minimum}")


def test_source_comments_cite_symbols_rather_than_line_numbers():
    """No source file points into another one by line number.

    A reference that names a file and a line number is unverifiable, and it is
    silently wrong the moment anything above that line changes. Name the
    function or attribute instead.
    """
    offenders = []
    for root in ("bipred", "tests", "benchmarks"):
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.relative_to(ROOT).parts:
                continue
            if any(part in ARCHIVED for part in path.relative_to(ROOT).parts):
                continue
            for number, line in enumerate(_read(path).splitlines(), 1):
                for hit in SOURCE_LINE_REF_RE.findall(line):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{number} cites {hit}")

    assert not offenders, (
        "source files cite a line number instead of a symbol:\n"
        + "\n".join(offenders))
