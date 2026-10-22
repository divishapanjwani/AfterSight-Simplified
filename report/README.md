# Mini project report

- `AfterSight_Mini_Project_Report.docx`: editable report in the college format. Fill in the student
  table and the guide name on the cover, and the contributors in Table 8.1.
- `AfterSight_Mini_Project_Report.pdf`: PDF export of the same file (LibreOffice).

Every number in the report is read from saved artifacts (`build/make_assets.py` writes `build/data.json`).
Figures are screenshots and pipeline outputs copied into `assets/`.

## Rebuild

```zsh
source .venv/bin/activate
python report/build/make_assets.py           # data.json + figures from artifacts/ and outputs/
(cd report/build && npm ci)                  # docx builder dependency
python report/build/paginate.py              # build .docx → render PDF → fill TOC page numbers → repeat until stable
```

The Table of Contents and the lists of figures and tables use **static** page numbers, computed from the
PDF render. If you add text in Word that shifts pages, update those numbers by hand or rebuild.

## Licence note

Satellite imagery in the figures is derived from xBD (CC BY-NC-SA 4.0; Gupta et al., 2019). The report text
and figures are shared for academic, non-commercial use.
