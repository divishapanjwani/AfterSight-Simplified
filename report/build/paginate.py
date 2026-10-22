"""Two-pass pagination: render the .docx to PDF, find each heading/caption's page, rebuild until stable.

Usage (repo root): .venv/bin/python report/build/paginate.py
Writes report/build/pages.json and the final report/AfterSight_Mini_Project_Report.{docx,pdf}.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
REPORT = BUILD.parent
DOCX = REPORT / "AfterSight_Mini_Project_Report.docx"
PDF = REPORT / "AfterSight_Mini_Project_Report.pdf"
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def render() -> list[str]:
    subprocess.run(["node", "build_report.js"], cwd=BUILD, check=True, capture_output=True)
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(REPORT), str(DOCX)], check=True, capture_output=True)
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], check=True, capture_output=True, text=True).stdout
    return text.split("\f")


def locate(pages: list[str]) -> dict[str, int]:
    reg = json.loads((BUILD / "registry.json").read_text())
    body_start = next(i for i, p in enumerate(pages)
                      if any(norm(l) == "1. Introduction" for l in p.splitlines()) and "Table of Contents" not in p and ".........." not in p)
    found: dict[str, int] = {}
    entries = [e["text"] for e in reg["toc"]] + reg["figures"] + reg["tables"]
    for entry in entries:
        key = norm(entry)[:48]
        for i in range(body_start, len(pages)):
            joined = norm(pages[i])
            lines = [norm(l) for l in pages[i].splitlines()]
            if any(l.startswith(key) for l in lines) or (len(key) > 30 and key in joined):
                found[entry] = i - body_start + 1
                break
        else:
            print(f"  NOT FOUND: {entry}", file=sys.stderr)
    return found


def main() -> int:
    previous = None
    for attempt in range(1, 5):
        pages = render()
        found = locate(pages)
        (BUILD / "pages.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
        print(f"pass {attempt}: {len(pages) - 1} PDF pages, {len(found)} entries located")
        if found == previous:
            print("page numbers stable")
            return 0
        previous = found
    print("page numbers did not stabilise", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
