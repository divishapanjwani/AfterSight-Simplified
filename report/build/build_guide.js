// Builds report/Understanding_AfterSight.pdf from docs/UNDERSTANDING_AFTERSIGHT.md.
// Markdown -> styled HTML (Mermaid rendered in the browser) -> PDF via headless Chrome.
// Usage (from report/build): node build_guide.js   (page numbers: ../../.venv/bin/python stamp_pages.py)

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { marked } = require('marked');

const ROOT = path.resolve(__dirname, '..', '..');
const MD = path.join(ROOT, 'docs', 'UNDERSTANDING_AFTERSIGHT.md');
const HTML = path.join(__dirname, 'guide.html');
const PDF = path.join(ROOT, 'report', 'Understanding_AfterSight.pdf');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const MERMAID = path.join(__dirname, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js');

// GitHub-style heading slugs so the in-document contents links work in the PDF too
const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\w\- ]+/g, '').trim().replace(/ /g, '-');

const renderer = {
  heading({ tokens, depth, text }) {
    const inner = this.parser.parseInline(tokens);
    const cls = text.trim() === 'Answers' ? ' class="answers"' : '';
    return `<h${depth} id="${slug(text)}"${cls}>${inner}</h${depth}>\n`;
  },
  code({ text, lang }) {
    if (lang === 'mermaid') return `<div class="mermaid">${text}</div>\n`;
    return false; // default rendering
  },
  image({ href, title, text }) {
    return `<img src="${href}" alt="${text}"${title ? ` title="${title}"` : ''}>`;
  },
};
marked.use({ renderer, gfm: true });

const body = marked.parse(fs.readFileSync(MD, 'utf8'));

const css = `
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
:root { --ink: #1d1b22; --ink2: #4a4652; --rule: #d9d4de; --accent: #5b2a6e; --tint: #f6f3f8; }
html { background: #fff; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.2pt; line-height: 1.45; color: var(--ink); background: #fff; margin: 0; }
h1 { font-size: 24pt; color: var(--accent); margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 16pt; color: var(--accent); margin: 0 0 8pt; padding-bottom: 4pt; border-bottom: 1.5pt solid var(--accent); break-before: page; }
h3 { font-size: 12.5pt; margin: 14pt 0 4pt; break-after: avoid; }
h2 + p, h3 + p { margin-top: 2pt; }
h2.answers, h3.answers { break-before: page; }
p { margin: 5pt 0; }
ul, ol { margin: 4pt 0; padding-left: 20pt; }
li { margin: 2pt 0; }
a { color: var(--accent); text-decoration: none; }
code { font-family: Menlo, 'Courier New', monospace; font-size: 8.6pt; background: var(--tint); padding: 0 2pt; border-radius: 2pt; }
pre { background: var(--tint); padding: 6pt; font-size: 8.6pt; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9pt; break-inside: auto; }
tr { break-inside: avoid; }
th { background: var(--tint); text-align: left; font-weight: bold; }
th, td { border: 0.6pt solid var(--rule); padding: 3pt 5pt; vertical-align: top; }
td code, th code { font-size: 8pt; }
img { display: block; max-width: 100%; max-height: 100mm; margin: 8pt auto 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
p:has(> em:only-child) { font-size: 9pt; color: var(--ink2); margin-top: 0; }
hr { border: none; margin: 0; }
.mermaid { display: flex; justify-content: center; margin: 8pt 0; break-inside: avoid; }
.mermaid svg { max-width: 100% !important; max-height: 125mm; height: auto; }
blockquote { border-left: 3pt solid var(--accent); margin: 6pt 0; padding: 2pt 10pt; color: var(--ink2); }
`;

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Understanding AfterSight</title>
<base href="file://${path.join(ROOT, 'docs')}/">
<style>${css}</style>
</head><body>
${body}
<script src="file://${MERMAID}"></script>
<script>
  mermaid.initialize({ startOnLoad: false, theme: 'neutral', fontFamily: 'Helvetica, Arial, sans-serif',
                       flowchart: { useMaxWidth: true }, sequence: { useMaxWidth: true } });
  mermaid.run({ querySelector: '.mermaid' }).then(() => { document.body.dataset.ready = '1'; });
</script>
</body></html>`;

fs.writeFileSync(HTML, html);
execFileSync(CHROME, [
  '--headless=new', '--disable-gpu', '--allow-file-access-from-files', '--no-pdf-header-footer',
  '--virtual-time-budget=20000', `--print-to-pdf=${PDF}`, `file://${HTML}`,
], { stdio: 'ignore' });
console.log(`wrote ${path.relative(ROOT, PDF)}`);
