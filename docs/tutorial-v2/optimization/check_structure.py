"""Read-only, standard-library checks for the reorganized tutorial; no model imports."""

from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote, urlsplit


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parents[2]
    mapping = json.loads((folder / "coverage.json").read_text())
    errors = []
    chapters = mapping["chapters"]
    lessons = mapping["lessons"]
    if sorted(item["chapter"] for item in chapters) != list(range(16, 46)):
        errors.append("Original chapter IDs must be exactly 16 through 45, once each.")
    names = [item["file"] for item in lessons]
    expected = [f"{part}{n}" for part, count in zip("ABCDEF", [2, 3, 3, 3, 4, 3])
                for n in range(1, count + 1)]
    if [item["id"] for item in lessons] != expected or len(set(names)) != 18:
        errors.append("Expected 18 unique lessons, A1 through F3 with E1–E4.")
    referenced = set()
    originals = 0
    for chapter in chapters:
        referenced.update(chapter["lessons"])
        if not chapter["teaching_points"] or not chapter["details_retained_in_original"]:
            errors.append(f"Missing content disposition: {chapter['chapter']}")
        for source in chapter["sources"]:
            originals += 1
            try:
                old = subprocess.check_output(
                    ["git", "show", source["revision"] + ":" + source["path"]], cwd=root)
                if hashlib.sha256(old).hexdigest() != source["sha256"]:
                    errors.append(f"Baseline hash differs: {source['path']}")
                if "tag" in source:
                    revision = subprocess.check_output(
                        ["git", "rev-parse", source["tag"] + "^{commit}"],
                        cwd=root, text=True).strip()
                    if revision != source["revision"]:
                        errors.append(f"Historical tag moved: {source['tag']}")
                elif (root / source["path"]).read_bytes() != old:
                    errors.append(f"Original text changed: {source['path']}")
            except (OSError, subprocess.CalledProcessError) as exc:
                errors.append(f"Cannot read source {source['path']}: {exc}")
    if referenced != set(names):
        errors.append("Every lesson must be mapped, and every mapped lesson must exist.")
    for name in names:
        if not (folder / name).is_file():
            errors.append(f"Missing lesson: {name}")
    links = 0
    documents = sorted(folder.glob("*.md"))
    for doc in documents:
        fence = None
        for line_number, line in enumerate(doc.read_text().splitlines(), 1):
            marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
            if marker:
                run, suffix = marker.groups()
                if fence is None:
                    fence = run
                elif run[0] == fence[0] and len(run) >= len(fence) and not suffix.strip():
                    fence = None
                continue
            if fence:
                continue
            # The new pages use inline Markdown links; not a general Markdown parser.
            for href in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", line):
                target = urlsplit(href)
                if target.scheme or target.netloc or not target.path:
                    continue
                links += 1
                path = Path(unquote(target.path))
                if path.is_absolute() or not (doc.parent / path).exists():
                    errors.append(f"Bad local link {doc.name}:{line_number}: {href}")
        if fence:
            errors.append(f"Unclosed fence: {doc.name}")
    result = {
        "status": "PASS_STATIC" if not errors else "FAIL",
        "original_chapter_ids": len(chapters),
        "original_documents_verified": originals,
        "short_lessons": len(lessons),
        "markdown_documents": len(documents),
        "relative_file_links_checked": links,
        "scope": "CPU file/link/fence preservation checks; no semantic proof, external link fetch, native rendering, model loading or performance test.",
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
