// Builds report/AfterSight_Mini_Project_Report.docx from data.json (all numbers) and ../assets (figures).
// Page numbers for the TOC / lists come from pages.json, written by paginate.py after a first render.
const fs = require('fs')
const path = require('path')
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  BorderStyle, ImageRun, PageBreak, Footer, PageNumber, NumberFormat, ShadingType, LevelFormat, PositionalTab,
  PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader, VerticalAlign, TableLayoutType, Tab, TabStopType, LeaderType,
} = require('docx')

const D = require('./data.json')
const SIZES = require('./image_sizes.json')
const PAGES = fs.existsSync(path.join(__dirname, 'pages.json')) ? require('./pages.json') : {}
const ASSETS = path.join(__dirname, '..', 'assets')
const FONT = 'Times New Roman'
const CONTENT_W = 9026 // A4 width 11906 - 2 x 1440 margins (DXA)

// ---------------------------------------------------------------- number formatting (values only from D)
const f3 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(3))
const f2 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(2))
const int = (v) => Number(v).toLocaleString('en-GB')
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`
const CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
const MODEL_NAME = { baseline: 'Threshold baseline', logistic_regression: 'Logistic Regression', knn: 'K-Nearest Neighbours', decision_tree: 'Decision Tree', random_forest: 'Random Forest (optional)' }

// ---------------------------------------------------------------- text helpers
function runs(text, base = {}) {
  // **bold**, _italic_ (only at word boundaries, so feature names like pre_mean_r stay intact) and `code`
  const out = []
  const re = /(\*\*[^*]+\*\*|(?<![A-Za-z0-9])_[^_]+_(?![A-Za-z0-9])|`[^`]+`)/g
  let last = 0
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }))
    const t = m[0]
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }))
    else if (t.startsWith('`')) out.push(new TextRun({ text: t.slice(1, -1), font: 'Courier New', ...base, size: Math.round((base.size ?? 24) * 0.9) }))
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }))
    last = m.index + t.length
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }))
  return out
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o.run), alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: o.line ?? 340 }, keepNext: o.keepNext })
const bullet = (text) => new Paragraph({ children: runs(text), numbering: { reference: 'bullets', level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 320 } })
const blank = () => new Paragraph({ children: [new TextRun({ text: '', font: FONT })] })
const pageBreak = () => new Paragraph({ children: [new PageBreak()] })

// ---------------------------------------------------------------- headings, captions, registries
const toc = []
const figures = []
const tables = []
let chapter = 0
let figN = 0
let tabN = 0
function H1(title, newPage = true) {
  chapter += 1
  figN = 0
  tabN = 0
  const text = `${chapter}. ${title}`
  toc.push({ level: 1, text })
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT, bold: true, size: 32 })], spacing: { before: newPage ? 120 : 360, after: 200 }, pageBreakBefore: newPage, keepNext: true })
}
function H2(num, title) {
  const text = `${chapter}.${num} ${title}`
  toc.push({ level: 2, text })
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT, bold: true, size: 26 })], spacing: { before: 220, after: 120 }, keepNext: true })
}
function tableCaption(title) {
  tabN += 1
  const text = `Table ${chapter}.${tabN}: ${title}`
  tables.push(text)
  return new Paragraph({ children: [new TextRun({ text, font: FONT, bold: true, size: 22 })], alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true })
}
function figure(file, title, widthIn = 6.2, maxHIn = 8.3) {
  figN += 1
  const text = `Figure ${chapter}.${figN}: ${title}`
  figures.push(text)
  const [pw, ph] = SIZES[file]
  let w = widthIn * 96
  let h = (w * ph) / pw
  const maxH = maxHIn * 96 // keep figures short enough to share a page
  if (h > maxH) { h = maxH; w = (h * pw) / ph }
  return [
    new Paragraph({ children: [new ImageRun({ type: 'png', data: fs.readFileSync(path.join(ASSETS, file)), transformation: { width: Math.round(w), height: Math.round(h) },
      altText: { title, description: title, name: file } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text, font: FONT, italics: true, size: 21 })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
  ]
}

// ---------------------------------------------------------------- tables
const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' }
const borders = { top: border, bottom: border, left: border, right: border }
function cell(text, width, o = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: o.head ? { fill: 'E7E3D9', type: ShadingType.CLEAR, color: 'auto' } : o.shade ? { fill: 'F4F2EC', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: (o.size ?? 19) <= 17 ? { top: 25, bottom: 25, left: 70, right: 70 } : { top: 45, bottom: 45, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(String(text), { size: o.size ?? 19, bold: o.head || o.bold }), alignment: o.align ?? AlignmentType.LEFT, spacing: { after: 0, line: 250 } })],
  })
}
function table(headers, rows, widths, o = {}) {
  const sum = widths.reduce((a, b) => a + b, 0)
  const scaled = widths.map((w) => Math.floor((w * CONTENT_W) / sum))
  scaled[scaled.length - 1] += CONTENT_W - scaled.reduce((a, b) => a + b, 0)
  const size = o.size ?? 19
  const align = (i) => (o.numeric?.includes(i) ? AlignmentType.RIGHT : AlignmentType.LEFT)
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: scaled, layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, scaled[i], { head: true, size, align: align(i) })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(v, scaled[i], { size, align: align(i), bold: o.boldRows?.includes(ri), shade: o.shadeRows?.includes(ri) })) })),
    ],
  })
}
const after = () => new Paragraph({ children: [], spacing: { after: 120 } })

// ---------------------------------------------------------------- front matter
function cover() {
  const c = (text, size, o = {}) => new Paragraph({ children: [new TextRun({ text, font: FONT, size, bold: o.bold, italics: o.italics })], alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 80 } })
  const studentTable = new Table({
    width: { size: 8200, type: WidthType.DXA }, columnWidths: [3000, 1700, 2000, 1500], alignment: AlignmentType.CENTER, layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ children: ['Student Name', 'U. No.', 'Class / Division', 'Roll No.'].map((h, i) => cell(h, [3000, 1700, 2000, 1500][i], { head: true, size: 22, align: AlignmentType.CENTER })) }),
      ...[1, 2, 3, 4].map(() => new TableRow({ height: { value: 520, rule: 'atLeast' }, children: [3000, 1700, 2000, 1500].map((w) => cell(' ', w, { size: 22 })) })),
    ],
  })
  return [
    c('MKSSS’s', 28, { bold: true }),
    c('Cummins College of Engineering for Women, Pune', 32, { bold: true }),
    c('(An Autonomous Institute Affiliated to Savitribai Phule Pune University)', 22, { italics: true, after: 360 }),
    c('Third Year B. Tech (Computer Engineering)', 26, { bold: true, after: 360 }),
    c('MINI PROJECT REPORT', 34, { bold: true }),
    c('On', 24, { after: 160 }),
    c('“AfterSight: Explainable Building Damage Triage from Pre/Post-Disaster Satellite Imagery using Classical Machine Learning”', 28, { bold: true, after: 480 }),
    c('Submitted by', 24, { bold: true, after: 160 }),
    studentTable,
    c(' ', 24, { after: 360 }),
    c('Under the Guidance of', 24, { bold: true, after: 160 }),
    c('Guide Name: ______________________________', 24, { after: 480 }),
    c('Department of Computer Engineering', 26, { bold: true }),
    c('Academic Year: 2026–27', 24, { bold: true }),
  ]
}

function listPage(title, entries, key) {
  // right-aligned tab stop with a dot leader (rendered the same in Word and LibreOffice)
  const lines = entries.map((e) => new Paragraph({
    indent: { left: e.level === 2 ? 480 : 0, right: 0 },
    tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: LeaderType.DOT }],
    spacing: { after: e.level === 2 ? 40 : 80, before: e.level === 1 ? 80 : 0 },
    children: [
      new TextRun({ text: e.text, font: FONT, size: e.level === 1 ? 24 : 22, bold: e.level === 1 }),
      new TextRun({ children: [new Tab(), String(PAGES[e.text] ?? '00')], font: FONT, size: e.level === 1 ? 24 : 22, bold: e.level === 1 }),
    ],
  }))
  return [new Paragraph({ children: [new TextRun({ text: title, font: FONT, bold: true, size: 32 })], alignment: AlignmentType.CENTER, spacing: { after: 240 } }), ...lines]
}

// ---------------------------------------------------------------- body
const C = D.comparisons
const M = D.models
const within = C['within_event/four_class']
const cross = C['cross_event/four_class']
const row = (comp, model) => comp.models.find((m) => m.model === model)
const lrW = row(within, 'logistic_regression')
const lrX = row(cross, 'logistic_regression')
const ins = D.inspection
const fm = D.features_meta
const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0)
const share = (c) => fm.class_counts[c] / fm.n_buildings
const conf = D.confounding
const km = D.kmeans
const best = (comp) => comp.models.reduce((a, b) => (b.test_macro_f1 > a.test_macro_f1 ? b : a))

function body() {
  const out = []
  // 1 ------------------------------------------------------------ Introduction
  out.push(H1('Introduction'))
  out.push(P('When a hurricane, flood, wildfire or tsunami strikes, emergency managers need to know quickly **which buildings are damaged and how badly**. Ground surveys are slow, expensive and often dangerous in the first days after an event. High-resolution satellite images taken before and after the disaster cover an entire city within hours, but inspecting tens of thousands of roofs by eye is not feasible. Automatic building damage assessment from satellite imagery is therefore an important problem in remote sensing and disaster response.'))
  out.push(P('The **xBD** dataset (Gupta et al., 2019) made this problem measurable at scale. It provides pairs of pre- and post-disaster satellite images with building footprints and a four-level damage scale: _no damage_, _minor damage_, _major damage_ and _destroyed_ [1]. Most published approaches to xBD use deep convolutional networks, which are accurate but difficult to explain and expensive to train.'))
  out.push(P(`**AfterSight** takes a deliberately different route. It classifies each building with **classical, explainable machine learning** (a threshold baseline, Logistic Regression, K-Nearest Neighbours, a Decision Tree and, as a reference, a Random Forest) using **${D.feature_rows.length} hand-crafted features** per building: colour, change, texture, the surroundings of the building and its shape. Every prediction comes with an explanation: the features that pushed a Logistic Regression decision, the exact path through the Decision Tree, or the most similar training buildings for KNN.`))
  out.push(P(`The project is delivered as a complete web application. A FastAPI backend serves saved models and results, and a React interface provides a **Triage Desk**, a **Building Case File**, a **Model Arena**, a **Feature Lab**, a **Cluster Atlas**, a **Review Queue** and **Reports**. The Triage Desk has a before/after swipe viewer with building overlays. The scope is stated on every page: _“Damage classification on building footprints derived from xBD annotations.”_ The system does not detect buildings itself and gives no rescue or routing advice.`))
  out.push(P(`A central aim of the project is **honest evaluation**. Models are tested in two ways: on new scenes of the five disasters they were trained on (within-event), and on a disaster they never saw (cross-event, hurricane-michael held out). The default model reaches a test macro-F1 of **${f3(lrW.test_macro_f1)}** within-event, but **${f3(lrX.test_macro_f1)}** on the unseen disaster. Chapter 12 shows why this gap exists.`))

  // 2 ------------------------------------------------------------ Motivation
  out.push(H1('Motivation', false))
  out.push(P('Three observations motivated AfterSight.'))
  out.push(bullet('**Speed matters after a disaster.** A first, approximate damage map helps prioritise ground teams. Even an imperfect model that ranks buildings for human review can save time.'))
  out.push(bullet('**Explanations build trust.** Responders and examiners need to know why a building is marked destroyed. Classical models on interpretable features can show their reasoning directly (“the roof texture became much rougher”). A deep network cannot do this without extra tools.'))
  out.push(bullet('**Scores are often over-optimistic.** Buildings in the same image share lighting, sensor and often damage. Random splits leak this information, and evaluating only on known disasters hides how a model behaves on a new event. We wanted to measure performance honestly and show the limits clearly.'))
  out.push(P('The project also had a teaching motivation: to apply the classical machine-learning syllabus (feature engineering, linear models, nearest neighbours, decision trees, clustering, class imbalance and evaluation) end to end on a real, messy, large dataset, with reproducible code and a usable interface.'))

  // 3 ------------------------------------------------------------ Literature survey
  out.push(H1('Literature Survey'))
  out.push(P('Table 3.1 summarises the five works most relevant to this project. Every reference was checked against its publisher or arXiv record.'))
  out.push(tableCaption('Literature survey'))
  out.push(table(['Sr. No.', 'Author(s)', 'Year', 'Title / Method', 'Dataset', 'Technique', 'Major Findings', 'Limitation'], [
    ['1', 'R. Gupta, R. Hosfelt, S. Sajeev et al. [1]', '2019', 'xBD: A Dataset for Assessing Building Damage from Satellite Imagery', 'xBD: pre/post satellite pairs, building polygons, 4-level damage scale', 'Dataset creation with a joint damage scale; baseline models for the xView2 challenge', 'Released 850,736 building annotations over 45,362 km² of imagery, enabling large-scale damage assessment research', 'Strong class imbalance (most buildings undamaged); baseline models are not designed for explanation'],
    ['2', 'R. M. Haralick, K. Shanmugam, I. Dinstein [2]', '1973', 'Textural Features for Image Classification', 'Photo-micrographs of sandstone, aerial photographs, satellite imagery', 'Grey-level co-occurrence (spatial dependency) texture statistics', 'Texture features gave about 89%, 82% and 83% accuracy on the three image sets', 'Hand-crafted; depends on grey-level quantisation and pixel offset; texture only'],
    ['3', 'A. J. Cooner, Y. Shao, J. B. Campbell [3]', '2016', 'Detection of Urban Damage Using Remote Sensing and Machine Learning Algorithms: Revisiting the 2010 Haiti Earthquake', 'High-resolution imagery of Port-au-Prince; UNITAR/UNOSAT damage data', 'Feed-forward and radial-basis neural networks, Random Forest; texture and structure features', 'Each algorithm reached about 90% agreement with the UNOSAT validation data', 'Single event; transfer to other disaster types not tested'],
    ['4', 'E. Weber, H. Kané [4]', '2020', 'Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion', 'xBD', 'Shared-weight ResNet50 backbone on pre and post images, fused features, segmentation into damage classes', 'Substantial improvement over the xBD baseline; among the top results on the xView2 leaderboard', 'Deep CNN: needs GPU training, decisions hard to explain'],
    ['5', 'R. Gupta, M. Shah [5]', '2020', 'RescueNet: Joint Building Segmentation and Damage Assessment from Satellite Imagery', 'xBD', 'Single end-to-end network for building segmentation and damage classification with a combined loss', 'Better segmentation and damage classification than earlier multi-stage methods; reported generalisation across regions and disaster types', 'Deep model with high compute cost; per-building reasoning not exposed'],
  ], [430, 1250, 500, 1480, 1300, 1330, 1450, 1286], { size: 16 }))
  out.push(after())
  out.push(H2(1, 'Gap addressed by this project'))
  out.push(P('Work on xBD concentrates on deep networks optimised for leaderboard scores, while classical studies such as [3] evaluate a single disaster. Two questions remain open for a student-scale, explainable system:'))
  out.push(bullet('**How far can classical models on interpretable features go** on xBD, with every decision explained per building?'))
  out.push(bullet('**How much of the score survives on a disaster the model has never seen**, and is the gap caused by the model recognising the disaster rather than the damage?'))
  out.push(P('AfterSight answers both. It uses texture ideas from [2] and the Canny edge detector [6], scene-grouped and event-held-out splits, an explicit test of event confounding, and an interface that shows every explanation. All models are implemented with scikit-learn [7].'))

  // 4 ------------------------------------------------------------ Dataset
  out.push(H1('Dataset Used'))
  out.push(H2(1, 'Source and format'))
  out.push(P(`The project uses the training split of **xBD** [1], published by the xView2 organisers under the CC BY-NC-SA 4.0 licence [11]. The copy available to us was a Hugging Face-style **Parquet mirror** of ${ins.shards} shards holding ${int(ins.rows_all_events)} scene pairs from 10 disaster events. Each row contains the pre-disaster image, the post-disaster image and two annotation masks (Table 4.1). The data is not redistributed with the code.`))
  out.push(tableCaption('Columns of the Parquet mirror (verified by reading the files)'))
  out.push(table(['Column', 'Content'], [
    ['t1_image', 'pre-disaster image, PNG, 1024 × 1024 RGB'],
    ['t2_image', 'post-disaster image, PNG, 1024 × 1024 RGB'],
    ['t1_mask', 'building footprint mask: 0 background, 1 building'],
    ['t2_mask', 'damage mask: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed'],
    ['image_name', 'e.g. train/images/socal-fire_00000106 (event and scene number)'],
  ], [2000, 7026]))
  out.push(after())
  out.push(H2(2, 'Events selected'))
  out.push(P('A first inspection of all ten events showed that some, such as mexico-earthquake and midwest-flooding, contain almost no major-damage or destroyed buildings. Five events covering four disaster types were selected (Table 4.2). The disaster type is not stored in the mirror and is taken from the project configuration.'))
  out.push(tableCaption('Selected events: scenes, buildings and ground-truth class counts'))
  const evRows = D.per_event.map((e) => [e.event, e.type, int(e.scenes), int(e.empty), int(e.buildings), int(e['no-damage']), int(e['minor-damage']), int(e['major-damage']), int(e.destroyed)])
  evRows.push(['Total', '—', int(sum(Object.fromEntries(D.per_event.map((e) => [e.event, e.scenes])))), int(sum(Object.fromEntries(D.per_event.map((e) => [e.event, e.empty])))), int(fm.n_buildings),
    int(fm.class_counts['no-damage']), int(fm.class_counts['minor-damage']), int(fm.class_counts['major-damage']), int(fm.class_counts.destroyed)])
  out.push(table(['Event', 'Type', 'Scenes', 'Empty', 'Buildings', 'No damage', 'Minor', 'Major', 'Destroyed'], evRows, [1900, 900, 750, 700, 950, 1000, 850, 850, 1000], { numeric: [2, 3, 4, 5, 6, 7, 8], boldRows: [evRows.length - 1], shadeRows: [evRows.length - 1] }))
  out.push(after())
  out.push(H2(3, 'Building footprints derived from masks'))
  out.push(P(`The mirror has no polygons, so buildings are reconstructed. Each **connected blob** of the footprint mask (4-connectivity) becomes one building, and its **label** is the majority damage value inside the blob. A full scan of all scenes found only the damage values 0–4. In the five events, ${int(ins.totals.raw_footprints)} blobs were found and ${int(ins.totals.buildings_kept)} were kept. Excluded blobs were logged: ${int(ins.drop_reasons.label_missing ?? 0)} had no damage value (most likely xBD’s “un-classified” buildings), ${int(ins.drop_reasons.too_small ?? 0)} were smaller than 16 px and ${int(ins.drop_reasons.label_tie ?? 0)} had a tied vote. ${int(ins.totals.mixed_label_blobs)} blobs contained more than one damage value and were labelled by majority. Each blob outline is traced into a polygon in image pixel coordinates (Figure 4.1).`))
  out.push(...figure('fig_sample_fire.png', 'Footprints derived from the masks, drawn on a pre/post pair from the Santa Rosa wildfire', 5.6, 3.4))
  out.push(H2(4, 'Feature description'))
  out.push(P('For every building, 37 numeric features are computed on its exact pixels in both images (Table 4.3). Most are region statistics such as means, spreads and proportions, which tolerate small misalignment between the pre and post images. The target is the damage label. Identifiers (scene, event, disaster type) and quality flags are stored separately and are **never** model inputs.'))
  out.push(tableCaption('Feature dictionary: 37 input features and the target'))
  const fRows = D.feature_rows.map((f) => [f.name, `numeric (${f.group})`, f.formula, 'input feature'])
  fRows.push(['label', 'categorical (4 classes)', 'majority damage-mask value inside the footprint: no-damage, minor-damage, major-damage, destroyed', 'target'])
  out.push(table(['Feature', 'Type', 'Description', 'Role'], fRows, [2150, 1450, 4300, 1126], { size: 16, boldRows: [fRows.length - 1], shadeRows: [fRows.length - 1] }))
  out.push(after())
  out.push(H2(5, 'Missing values and quality checks'))
  out.push(bullet(`**Missing / infinite values:** ${Object.keys(fm.nan_counts).length === 0 ? 'none' : JSON.stringify(fm.nan_counts)} and ${Object.keys(fm.inf_counts).length === 0 ? 'none' : JSON.stringify(fm.inf_counts)} in the ${int(fm.n_buildings)}-row feature table. Median imputation is still kept in every pipeline as a safeguard.`))
  out.push(bullet(`**Image checks:** every one of the ${int(ins.totals.scenes_usable)} selected scenes decoded correctly as 1024 × 1024 RGB. A phase-correlation check flagged ${int(ins.alignment_flagged)} scenes with an estimated pre/post shift above 8 px, which is shown as a data-quality warning in the app.`))
  out.push(bullet(`**Footprint size:** blob areas range from ${int(ins.footprint_size.all_blobs_before_filter.min)} to ${int(ins.footprint_size.all_blobs_before_filter.max)} px, with a median of ${int(ins.footprint_size.all_blobs_before_filter.median)} px (Figure 4.2). The 16 px minimum sits in a natural gap in the histogram.`))
  out.push(bullet(`**Feature table:** ${int(fm.n_buildings)} rows × 37 features, ${(fm.table_bytes / 1e6).toFixed(1)} MB in Parquet, extracted in ${fm.timing.wall_seconds} s with ${fm.timing.workers} workers.`))
  out.push(...figure('fig_footprint_sizes.png', 'Distribution of footprint sizes with the 16 px minimum marked', 5.0, 2.2))
  out.push(H2(6, 'Class imbalance'))
  out.push(P(`The classes are imbalanced: no-damage makes up ${pct(share('no-damage'))} of buildings, minor ${pct(share('minor-damage'))}, major ${pct(share('major-damage'))} and destroyed ${pct(share('destroyed'))}. The mix also differs strongly between events (Figure 4.3): the fire and tsunami events are dominated by no-damage and destroyed, while the hurricanes contain most of the minor and major damage. This is handled with balanced class weights and macro-F1 as the primary metric (Section 7.6).`))
  out.push(...figure('fig_class_distribution.png', 'Ground-truth class counts and class share per selected event', 6.3, 3.0))

  // 5 ------------------------------------------------------------ Flow diagram
  out.push(H1('Flow Diagram / Block Diagram'))
  out.push(P('Figure 5.1 shows the complete pipeline from the raw Parquet shards to the web application. Each stage writes files that the next stage reads, so every number in the interface can be traced back to a saved artifact.'))
  out.push(...figure('fig_pipeline.png', 'AfterSight end-to-end pipeline', 5.6, 3.0))
  for (const b of [
    '**Parquet mirror:** the shards are streamed in small batches, so memory stays at about one row group regardless of dataset size.',
    '**Masks → buildings:** connected components of the footprint mask become buildings, labelled by the majority damage value. Every dropped blob is logged with its reason.',
    '**Scene cache:** only the five selected events are extracted, as original PNGs, thumbnails and polygons, so later stages open single scenes quickly.',
    '**Feature extraction:** 37 features per building, cached in a versioned Parquet table with a fingerprint of the settings.',
    '**Splits:** scene-grouped manifests for the within-event and cross-event designs, with automatic leakage checks.',
    '**Model training:** scikit-learn pipelines tuned on validation. Each final model is evaluated on its test set exactly once, and explanations and figures are produced.',
    '**Artifacts:** every model is saved with joblib plus a metadata JSON (hyperparameters, schema version, split hash, metrics, library version).',
    '**FastAPI backend:** a file-based store that validates every ID, caches analyses, and serves explanations, images, CSV/PDF reports and the review queue.',
    '**React web app:** the Triage Desk, Case File, Model Arena, Feature Lab, Cluster Atlas, Review Queue and Reports pages.',
  ]) out.push(bullet(b))

  // 6 ------------------------------------------------------------ Tech stack
  out.push(H1('Methods Used / Technology Stack'))
  out.push(H2(1, 'Programming languages'))
  out.push(bullet('**Python 3.11:** data pipeline, feature extraction, machine learning, evaluation and the REST API.'))
  out.push(bullet('**TypeScript (React):** the web interface, including the hand-built SVG charts and the scene viewer.'))
  out.push(bullet('**Bash:** the one-command pipeline (`run_pipeline.sh`) and application launcher (`start.sh`).'))
  out.push(H2(2, 'Libraries'))
  const ver = (name) => (D.requirements.find((r) => r.toLowerCase().startsWith(name.toLowerCase() + '==')) ?? '').split('==')[1]?.split(' ')[0] ?? '—'
  const npm = (name) => D.npm[name] ?? '—'
  out.push(tableCaption('Libraries and their purpose'))
  out.push(table(['Library (version)', 'Purpose', 'Main functions used'], [
    [`scikit-learn ${ver('scikit-learn')}`, 'models, preprocessing, splits, metrics', 'LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, KMeans, PCA, Pipeline, StandardScaler, SimpleImputer, StratifiedGroupKFold, permutation_importance, silhouette_score, adjusted_rand_score'],
    [`NumPy ${ver('numpy')}`, 'array computation', 'vectorised statistics, histograms, percentiles'],
    [`pandas ${ver('pandas')}`, 'tables', 'DataFrame, groupby, pivot_table, read/write Parquet'],
    [`PyArrow ${ver('pyarrow')}`, 'Parquet I/O', 'ParquetFile.iter_batches (streaming), write_table'],
    [`SciPy ${ver('scipy')}`, 'image labelling, statistics', 'ndimage.label, find_objects, binary_dilation; stats.skew, spearmanr'],
    [`scikit-image ${ver('scikit-image')}`, 'image features', 'feature.canny, registration.phase_cross_correlation'],
    [`OpenCV ${ver('opencv-python-headless')}`, 'contours', 'findContours (blob outline → polygon)'],
    [`Shapely ${ver('shapely')}`, 'geometry', 'Polygon, make_valid, union_all, WKT'],
    [`Pillow ${ver('pillow')}`, 'images', 'decode PNG, thumbnails, JPEG crops'],
    [`FastAPI ${ver('fastapi')} / Pydantic ${ver('pydantic')}`, 'REST API and schemas', 'APIRouter, dependency injection, response models, validation'],
    [`Matplotlib ${ver('matplotlib')} / ReportLab ${ver('reportlab')}`, 'figures and PDF reports', 'confusion matrices, charts; SimpleDocTemplate'],
    [`joblib ${ver('joblib')}`, 'model persistence', 'dump / load of fitted pipelines'],
    [`pytest ${ver('pytest')}`, 'backend tests', 'fixtures, parametrised tests, TestClient'],
    [`React ${npm('react')} / Vite ${npm('vite')}`, 'web interface and build', 'components, hooks, dev server with API proxy'],
    [`TanStack Query ${npm('@tanstack/react-query')} / React Router ${npm('react-router-dom')}`, 'data fetching, routing', 'useQuery, URL state'],
    [`Tailwind CSS ${npm('tailwindcss')}`, 'styling', 'design tokens, light/dark themes'],
    [`Vitest ${npm('vitest')} / Testing Library ${npm('@testing-library/react')}`, 'frontend tests', 'render, user events, mocked API'],
  ], [2500, 2100, 4426], { size: 16 }))
  out.push(after())
  out.push(H2(3, 'Development environment'))
  out.push(tableCaption('Development environment'))
  out.push(table(['Item', 'Details'], [
    ['Operating system', 'macOS 15.6'],
    ['Hardware', 'Apple Silicon (Apple M4, arm64)'],
    ['Python', 'Python 3.11.17 in a virtual environment at the repository root (.venv)'],
    ['Node.js / npm', 'Node.js v26.0.0, npm 11.12.1'],
    ['Editor', 'Visual Studio Code'],
    ['Version control', 'Git 2.39.5, GitHub'],
    ['Browser', 'Google Chrome (development and headless screenshots)'],
  ], [2500, 6526]))

  // 7 ------------------------------------------------------------ Model development
  out.push(H1('Model Development'))
  out.push(H2(1, 'Data preprocessing'))
  const logf = M['within_event/logistic_regression'].log1p
  out.push(P('All preprocessing happens **inside each scikit-learn Pipeline** and is fitted on the training split only, so validation and test data never influence it:'))
  out.push(bullet(`**log1p** on features whose training-set skewness exceeds 2. Seven features qualified: ${logf.join(', ')}. Footprint area, for example, has a very long tail.`))
  out.push(bullet('**Median imputation** as a safeguard; the real table has no missing values.'))
  out.push(bullet('**StandardScaler**, so that Logistic Regression and KNN treat all features on the same scale. The threshold baseline works in raw units and needs no preprocessing.'))
  out.push(H2(2, 'Exploratory data analysis'))
  out.push(P(`The Feature Lab (Figure 7.1) ranks every feature by its **separation score** η², the share of the feature’s variance explained by the damage class, computed on the training split. Table 7.1 lists the strongest features. Texture features dominate, led by the post-disaster co-occurrence homogeneity (η² = ${f3(D.feature_lab_top[0].eta2)}). The weakest on their own are ${D.feature_lab_bottom.map((f) => `${f.feature} (η² ${f3(f.eta2)})`).join(', ')}. Even the best single feature explains under a fifth of the variance, so the models must combine many; a feature that is weak alone, such as gray_std_post, can still matter in combination (it is among the most relied-on features of Logistic Regression).`))
  out.push(tableCaption('Strongest features by separation score (training split)'))
  out.push(table(['Feature', 'Group', 'η²', 'Median: no', 'minor', 'major', 'destroyed'], D.feature_lab_top.map((f) => [f.feature, f.group, f3(f.eta2), ...CLASSES.map((c) => (Math.abs(f.median[c]) >= 100 ? f.median[c].toFixed(0) : f.median[c].toFixed(3)))]), [2700, 1000, 800, 1100, 1100, 1100, 1226], { numeric: [2, 3, 4, 5, 6], size: 17 }))
  out.push(...figure('r07_feature_lab.png', 'Feature Lab: separation ranking, per-class box plot and two-feature scatter (training split)', 6.3, 4.2))
  out.push(H2(3, 'Feature engineering'))
  out.push(P('The 37 features (Table 4.3) were designed around how damage looks from above. Burnt or collapsed roofs change colour and brightness, so the colour and change groups capture that. Rubble is rough and patchy: grey-level spread, Canny edge density and co-occurrence contrast and homogeneity [2], [6] measure texture before and after. The **context ring** (ground within 8 px of the footprint, excluding all buildings) separates a change of the building from a change of the whole area, such as flood water or shadows. Two shape features (area, compactness) let the model normalise for building size. A separate experiment (Phase 3b, Section 12.5) tested change-only and scene-normalised variants of these features.'))
  out.push(H2(4, 'Model selection'))
  out.push(tableCaption('Models and their important parameters (tuned on validation)'))
  const hp = (m) => JSON.stringify(M[`within_event/${m}`].hyperparameters).replace(/[{}"]/g, '').replace(/,/g, ', ')
  out.push(table(['Model', 'Why included', 'Important parameters', 'Selected (within-event)'], [
    ['Threshold baseline (non-ML)', 'floor: one feature with ordered thresholds', 'candidate change features; thresholds from training quantiles (5% steps), chosen on validation', `feature: ${M['within_event/baseline'].hyperparameters.feature}`],
    ['Logistic Regression', 'linear, fast, coefficients explain decisions (app default)', `multinomial, class_weight=balanced, C ∈ {${D.grids.logistic_regression.C.join(', ')}}`, hp('logistic_regression')],
    ['K-Nearest Neighbours', 'explanation by similar examples', `k ∈ {${D.grids.knn.n_neighbors.join(', ')}}, weights ∈ {${D.grids.knn.weights.join(', ')}}`, hp('knn')],
    ['Decision Tree', 'human-readable rules', `max_depth ∈ {${D.grids.decision_tree.max_depth.join(', ')}}, min_samples_leaf ∈ {${D.grids.decision_tree.min_samples_leaf.join(', ')}}, balanced`, hp('decision_tree')],
    ['Random Forest (optional)', 'stronger classical reference [8]', `${D.grids.rf_trees} trees, max_depth ∈ {${D.grids.random_forest.max_depth.map((v) => v ?? 'None').join(', ')}}, min_samples_leaf ∈ {${D.grids.random_forest.min_samples_leaf.join(', ')}}, balanced_subsample`, hp('random_forest')],
    ['K-Means (analysis only)', 'is there natural structure in the features?', 'k = 2 … 10, n_init = 10', `k = ${km.chosen} (best silhouette)`],
  ], [1800, 2100, 3100, 2026], { size: 17 }))
  out.push(after())
  out.push(H2(5, 'Training'))
  const sw = D.splits.within_event.counts
  const sx = D.splits.cross_event.counts
  out.push(tableCaption('Training settings'))
  out.push(table(['Setting', 'Value'], [
    ['Split unit', 'whole scenes (buildings of one scene never cross splits); leakage test on the real manifests'],
    ['Within-event design', `class-stratified, scene-grouped (StratifiedGroupKFold, 20 folds of 5%) → train ${int(sw.train.buildings)} / val ${int(sw.val.buildings)} / test ${int(sw.test.buildings)} buildings; hash ${D.splits.within_event.hash}`],
    ['Cross-event design', `hurricane-michael held out as test (${int(sx.test.buildings)} buildings); train ${int(sx.train.buildings)} / val ${int(sx.val.buildings)} from the other four events; hash ${D.splits.cross_event.hash}`],
    ['Class weights', 'balanced (LR, DT), balanced_subsample (RF); KNN unweighted by class'],
    ['Model selection', 'best validation macro-F1 over the grid in Table 7.2'],
    ['Test-set use', 'exactly once per final model, enforced by a TestSetGuard object in code'],
    ['Random seed', '42 everywhere'],
    ['Label schemes', '4-class (primary); 3-class experiment no-damage / damaged / destroyed'],
  ], [2300, 6726], { size: 18 }))
  out.push(after())
  out.push(H2(6, 'Evaluation'))
  out.push(P(`Because ${pct(share('no-damage'), 0)} of buildings are undamaged, accuracy alone would reward a model that ignores the rare classes. The **primary metric is macro-F1**: the F1 score is computed per class and averaged with equal weight. Accuracy, weighted-F1, per-class precision and recall, confusion matrices, training time and inference time per 1,000 buildings are also reported. Interpretation is supported by permutation importance on validation, Logistic Regression coefficients, Decision Tree importances and paths, and KNN neighbours. Two extra analyses test the limits of the results: per-event scores, and a classifier that predicts the **event** from the same features (the confounding check).`))

  // 8 ------------------------------------------------------------ Version control
  out.push(H1('Version Control'))
  out.push(P('The project was developed in phases, each finished with a tested commit. Table 8.1 gives the version-control details, Table 8.2 lists the commit milestones and Figure 8.1 shows the history graph.'))
  out.push(tableCaption('Version control details'))
  out.push(table(['Item', 'Details'], [
    ['Tool', 'Git 2.39.5'], ['Platform', 'GitHub'], ['Repository name', 'aftersight'], ['Branch', 'main (linear history, one commit per phase)'],
    ['Contributors', '______________________________ (to be filled in by the group)'],
    ['Ignored from version control', 'dataset, scene cache, feature table, trained models, outputs, virtual environment, node_modules, secrets (.env)'],
  ], [2600, 6426]))
  out.push(after())
  out.push(tableCaption('Commit milestones'))
  out.push(table(['Commit', 'Date', 'Milestone'], D.commits.map((c) => [c.hash, c.date, c.subject]), [1100, 1400, 6526], { size: 17 }))
  out.push(...figure('fig_git_graph.png', 'git log --graph --oneline of the repository at the time of writing', 6.3, 2.6))

  // 9 ------------------------------------------------------------ Experiment tracking
  out.push(H1('Experiment Tracking'))
  out.push(P('No external tracking service was used. Every run is tracked with files kept next to the models:'))
  out.push(bullet('**Model metadata JSON:** for each saved model, the hyperparameters, the full validation search log, the preprocessing chosen, the feature-schema version, the split-manifest hash, the scikit-learn version, validation, test and per-event metrics, timings and the number of test evaluations (always 1).'))
  out.push(bullet('**Comparison files and RESULTS.md:** a comparison table per design and label scheme is written by the training run, and docs/RESULTS.md records the results with the commands that reproduce them.'))
  out.push(bullet('**Experiments kept separate:** the Phase 3b feature-variant models live in their own folder with their own test evaluations, so they can never overwrite the main models.'))
  out.push(P('Table 9.1 lists the tracked experiments for the 4-class task. The Model Arena page (Figure 9.1) presents the same files visually.'))
  out.push(tableCaption('Tracked experiments (4-class, test set evaluated once per model)'))
  const expRows = []
  for (const d of ['within_event', 'cross_event']) for (const m of ['baseline', 'logistic_regression', 'knn', 'decision_tree', 'random_forest']) {
    const x = M[`${d}/${m}`]
    expRows.push([MODEL_NAME[m], d.replace('_', '-'), 'full (37)', f3(x.val.macro_f1), f3(x.test.accuracy), f3(x.test.macro_f1)])
  }
  for (const r of D.phase3b) expRows.push([`${MODEL_NAME[r.model]} · Phase 3b`, r.design.replace('_', '-'), `${r.variant} (${r.n_features})`, f3(r.val_macro_f1), '—', f3(r.test_macro_f1)])
  out.push(table(['Model', 'Design', 'Feature set', 'Val macro-F1', 'Test accuracy', 'Test macro-F1'], expRows, [2600, 1300, 2300, 950, 950, 926], { numeric: [3, 4, 5], size: 16 }))
  out.push(P('Phase 3b runs record macro-F1 and recall; their accuracy is not part of the saved summary, so it is marked “—”.', { run: { size: 20, italics: true } }))
  out.push(...figure('r06_arena_top.png', 'Model Arena: every model on known vs unseen disasters, and the comparison table', 6.3, 4.2))

  // 10 ------------------------------------------------------------ AutoML / MLOps
  out.push(H1('AutoML / MLOps', false))
  out.push(P('**No AutoML was used.** All models, grids and features were chosen by hand so that every decision can be explained in the viva. Hyperparameters were tuned with small, explicit grids on the validation split. The project does follow several MLOps practices:'))
  for (const b of [
    `**Automated tests:** ${123} backend tests (pytest), ${26} frontend tests (Vitest) and an end-to-end check with 22 assertions. They cover the parser, features with known answers, split leakage, saving and reloading models, the API (including path-traversal attempts) and the user interface.`,
    '**Reproducible pipeline:** one command (`scripts/run_pipeline.sh`) rebuilds everything from the raw shards. A fresh clone following only the README reproduced every test score exactly.',
    '**Model metadata and loading checks:** a model refuses to load if the feature schema, the scikit-learn version or the split manifest differs from what it was trained with.',
    '**Test-set guard:** a code object raises an error if any model’s test set is evaluated a second time.',
    '**Versioned, fingerprinted data products:** the feature table records a fingerprint of its settings and is rebuilt only when they change. Split manifests carry a hash that detects hand edits.',
    '**Pinned dependencies:** exact versions in requirements.txt and package-lock.json.',
  ]) out.push(bullet(b))
  out.push(P('Not done: continuous integration, model monitoring in production and automated retraining. The system runs locally and is not deployed for operational use.'))

  // 11 ------------------------------------------------------------ Deployment
  out.push(H1('Deployment'))
  out.push(P('AfterSight is deployed **locally** on a laptop and works fully offline after installation. Two scripts cover the whole life cycle: `scripts/run_pipeline.sh` builds every data product and model in order (inspect → features → splits → train → review thresholds → experiments → demo cache), and `./start.sh` frees the two ports, starts the API and the web app, prints the URL and stops both cleanly. Table 11.1 lists the components. Figures 11.1–11.6 show the main pages; the Cluster Atlas and Reports pages are shown through their content in Chapter 12.'))
  out.push(tableCaption('Deployment components'))
  out.push(table(['Component', 'Technology', 'Port / location', 'Role'], [
    ['Data pipeline', 'Python scripts', 'scripts/, data/, artifacts/', 'builds the cache, features, splits, models, thresholds and demo cache'],
    ['REST API', 'FastAPI + Uvicorn', 'localhost:8020 (/docs for Swagger)', 'serves scenes, images, analyses, explanations, reports; validates IDs'],
    ['Web app', 'React + Vite', 'localhost:5200', 'user interface; proxies /api to the backend'],
    ['Artifact store', 'files (JSON, Parquet, joblib, JPEG)', 'artifacts/, data/processed/', 'no database; every number traces to a file'],
    ['Launcher', 'Bash', 'start.sh', 'frees ports, starts and stops both servers'],
  ], [1700, 1900, 2300, 3126], { size: 17 }))
  out.push(after())
  const shots = [
    ['r01_readiness.png', 'Readiness board: build status, honest headline results and the served models'],
    ['r02_library.png', 'Scene Library, opening on held-out test scenes with ground-truth damage bars'],
    ['r03_triage.png', 'Triage Desk: before/after swipe viewer with predicted damage overlays and building list'],
    ['r04_casefile.png', 'Building Case File with a Logistic Regression explanation'],
    ['r05_case_knn.png', 'Case File with a KNN explanation: the most similar training buildings'],
    ['r09_review.png', 'Review Queue: buildings suggested for human review'],
  ]
  for (const [file, cap] of shots) out.push(...figure(file, cap, 5.6, 4.15))

  // 12 ------------------------------------------------------------ Results
  out.push(H1('Results and Discussions'))
  const resRows = (comp, d) => comp.models.map((m) => [MODEL_NAME[m.model], f3(m.test_accuracy), f3(m.test_macro_f1), f3(m.test_weighted_f1), ...CLASSES.map((c) => f3(m.test_recall[c])), f2(m.train_seconds), f2(1000 * m.inference_seconds_per_1000)])
  out.push(H2(1, 'Within-event results'))
  out.push(P(`Test set: ${int(sw.test.buildings)} buildings from new scenes of the five training disasters (Table 12.1). Logistic Regression reaches a macro-F1 of ${f3(lrW.test_macro_f1)} and the optional Random Forest ${f3(row(within, 'random_forest').test_macro_f1)}, against ${f3(row(within, 'baseline').test_macro_f1)} for the single-feature baseline. **Destroyed** is the easiest class (LR recall ${f3(lrW.test_recall.destroyed)}). Minor and major damage are harder. KNN is about two orders of magnitude slower at inference than LR because it compares every building with all ${int(sw.train.buildings)} training buildings.`))
  out.push(tableCaption('Within-event test results (4-class)'))
  out.push(table(['Model', 'Acc.', 'Macro F1', 'Wtd. F1', 'Rec. none', 'Rec. minor', 'Rec. major', 'Rec. destr.', 'Train s', 'ms/1k'], resRows(within), [1900, 750, 750, 800, 750, 750, 750, 850, 650, 726], { numeric: [1, 2, 3, 4, 5, 6, 7, 8, 9], size: 16 }))
  out.push(H2(2, 'Cross-event results (unseen disaster)'))
  out.push(P(`Test set: all ${int(sx.test.buildings)} hurricane-michael buildings, a disaster never seen in training (Table 12.2). **Every model drops sharply**: the best macro-F1 is ${f3(best(cross).test_macro_f1)} (${MODEL_NAME[best(cross).model]}), and LR scores ${f3(lrX.test_macro_f1)}, although the same models reached ${f3(Math.min(...cross.models.filter((m) => m.model !== 'baseline').map((m) => m.val_macro_f1)))}–${f3(Math.max(...cross.models.filter((m) => m.model !== 'baseline').map((m) => m.val_macro_f1)))} on validation scenes from the four training events. Major-damage recall nearly vanishes (Figure 12.1, right).`))
  out.push(tableCaption('Cross-event test results (4-class, hurricane-michael held out)'))
  out.push(table(['Model', 'Acc.', 'Macro F1', 'Wtd. F1', 'Rec. none', 'Rec. minor', 'Rec. major', 'Rec. destr.', 'Train s', 'ms/1k'], resRows(cross), [1900, 750, 750, 800, 750, 750, 750, 850, 650, 726], { numeric: [1, 2, 3, 4, 5, 6, 7, 8, 9], size: 16 }))
  out.push(...figure('fig_cm_lr_pair.png', 'Confusion matrices, Logistic Regression: within-event test set (left) and cross-event test set (right); row % = recall', 6.3, 3.4))
  out.push(H2(3, 'Per-event scores'))
  const pe = M['within_event/logistic_regression'].per_event
  out.push(P(`Splitting the within-event test set by disaster (Table 12.3) shows that **every single event scores below the pooled ${f3(lrW.test_macro_f1)}**. Part of the pooled score therefore comes from telling the disasters apart, since each has its own typical damage mix, rather than from reading damage on each roof.`))
  out.push(tableCaption('Logistic Regression macro-F1 per event (within-event test)'))
  out.push(table(['Event', 'Macro-F1'], [...Object.entries(pe).map(([e, v]) => [e, f3(v)]), ['All events pooled', f3(lrW.test_macro_f1)]], [5500, 3526], { numeric: [1], boldRows: [Object.keys(pe).length], shadeRows: [Object.keys(pe).length] }))
  out.push(after())
  out.push(H2(4, 'Event-confounding check'))
  out.push(P(`A Logistic Regression was trained to predict the **event** (not the damage) from the same 37 features. It reaches **${pct(conf.accuracy)}** test accuracy (macro-F1 ${f3(conf.macro_f1)}), against ${pct(conf.majority_class_accuracy)} for always guessing the most common event and ${pct(conf.chance_accuracy_uniform)} uniform chance (Figure 12.2). The features carry a strong signature of where a building comes from: sensor, region, vegetation and roof style. This explains the gap between Tables 12.1 and 12.2. Event is never an input to any damage model.`))
  out.push(...figure('fig_confounding.png', 'Confounding check: confusion matrix for predicting the event from the features (test set)', 4.6, 3.3))
  out.push(H2(5, 'Phase 3b: reducing the event signature'))
  out.push(P('Three feature variants were tested to remove that signature: **change-only** (17 features with no absolute colour or texture levels), **scene-normalised** (each feature minus its scene median) and both combined. New LR and Decision Tree models were trained and tested once each (Table 12.4).'))
  out.push(tableCaption('Phase 3b results (separate models; Phase 3 models untouched)'))
  const p3 = D.phase3b.map((r) => [r.variant, r.design.replace('_', '-'), MODEL_NAME[r.model], f3(r.test_macro_f1), f3(r.test_recall_major), f3(r.test_recall_destroyed), r.design === 'within_event' ? f3(r.event_prediction_accuracy_within_event) : ''])
  out.push(table(['Variant', 'Design', 'Model', 'Test macro-F1', 'Recall major', 'Recall destroyed', 'Event-pred. accuracy'], p3, [2500, 1150, 1700, 950, 900, 950, 876], { numeric: [3, 4, 5, 6], size: 16 }))
  const p3x = D.phase3b.filter((r) => r.design === 'cross_event').map((r) => r.test_macro_f1)
  const p3e = D.phase3b.map((r) => r.event_prediction_accuracy_within_event).filter((v) => v !== null)
  out.push(P(`Event prediction fell from ${pct(conf.accuracy)} to as low as ${pct(Math.min(...p3e))}, but **cross-event macro-F1 did not improve** (${f3(Math.min(...p3x))}–${f3(Math.max(...p3x))} against ${f3(lrX.test_macro_f1)}), and within-event scores fell. The poor transfer is therefore not mainly the model exploiting event identity. It is a genuine **domain shift** between disasters that these features cannot bridge. The full 37-feature models remain the default.`))
  out.push(H2(6, 'K-Means clustering'))
  out.push(P(`K-Means was run on the training buildings for k = 2 … 10 (Figure 12.3). The silhouette is highest at k = ${km.chosen} (${f3(km.per_k[String(km.chosen)].silhouette)}) and the elbow heuristic suggests k = ${km.elbow}. Agreement with the damage labels is weak: the adjusted Rand index [10] is ${f3(km.per_k['2'].ari)} at k = 2 and ${f3(km.per_k['4'].ari)} at k = 4 (Figure 12.4). The strongest natural structure in the features is therefore not damage. Clusters are an analysis only and are never used as predictions [9].`))
  out.push(...figure('fig_kmeans_curves.png', 'K-Means: inertia (elbow) and sampled silhouette for k = 2 … 10', 6.0, 2.2))
  out.push(...figure('fig_kmeans_contingency.png', 'K-Means (k = 4) clusters against ground-truth damage classes', 3.6, 3.0))
  out.push(H2(7, '3-class experiment'))
  const w3 = C['within_event/three_class']
  const x3 = C['cross_event/three_class']
  out.push(P('Merging minor and major damage into one _damaged_ class (Table 12.5) raises scores, because the hardest distinction is removed. The drop on the unseen disaster remains. The 4-class task stays primary.'))
  out.push(tableCaption('3-class experiment: test macro-F1 (no-damage / damaged / destroyed)'))
  out.push(table(['Model', 'Within-event', 'Cross-event'], w3.models.map((m) => [MODEL_NAME[m.model], f3(m.test_macro_f1), f3(row(x3, m.model).test_macro_f1)]), [4000, 2513, 2513], { numeric: [1, 2] }))
  out.push(after())
  out.push(H2(8, 'Demo scenes and review suggestions'))
  out.push(P('Three held-out test scenes of different disaster types were chosen automatically for the demonstration, picking scenes with a clear mix of damage classes (Table 12.6). Scene-level scores are small samples and vary widely: the fire scene is predicted well, while the flooding scene, mostly minor damage under water, is difficult.'))
  out.push(tableCaption('Demo scenes (test split): ground truth and scene macro-F1 per model'))
  out.push(table(['Scene', 'Type', 'Buildings', 'GT none · minor · major · destroyed', 'LR', 'KNN', 'DT', 'RF'], D.demo.map((s) => [s.scene_id, s.disaster_type, String(s.n_buildings), CLASSES.map((c) => s.ground_truth_counts[c]).join(' · '),
    ...['logistic_regression', 'knn', 'decision_tree', 'random_forest'].map((m) => f2(s.analyses[m]?.scene_macro_f1))]), [2700, 900, 900, 2000, 630, 630, 630, 636], { numeric: [2, 4, 5, 6, 7], size: 16 }))
  out.push(P(`For the Review Queue, each model’s confidence threshold was chosen on validation as the lowest value at which confident buildings are at least ${pct(D.review.target_accuracy, 0)} correct (Table 12.7). Buildings below it are presented neutrally as _suggested for human review_.`))
  out.push(tableCaption('Review thresholds chosen on validation'))
  out.push(table(['Model', 'Threshold', 'Share flagged', 'Accuracy ≥ threshold', 'Accuracy < threshold'], Object.entries(D.review.models).map(([m, v]) => [MODEL_NAME[m], f3(v.threshold), pct(v.flagged_share_val, 0), f3(v.accuracy_at_or_above_val), f3(v.accuracy_below_val)]), [2700, 1400, 1500, 1700, 1726], { numeric: [1, 2, 3, 4] }))
  out.push(after())
  out.push(H2(9, 'Discussion'))
  out.push(bullet(`**${f3(lrW.test_macro_f1)} is an upper bound.** Within-event scores include help from recognising the disaster (per-event scores are all lower, and the event is predictable at ${pct(conf.accuracy)}).`))
  out.push(bullet(`**${f3(lrX.test_macro_f1)} is the realistic figure for a new disaster.** All learned models converge to about this value on hurricane-michael, still above the ${f3(row(cross, 'baseline').test_macro_f1)} baseline.`))
  out.push(bullet(`**Domain shift, not just a shortcut.** Removing the event signature did not help transfer. Roofs, imagery and the appearance of damage differ between disasters, and hurricane-michael has a different class mix: ${pct(sx.test.classes['minor-damage'] / sum(sx.test.classes), 0)} of its buildings are minor damage, against ${pct(sx.train.classes['minor-damage'] / sum(sx.train.classes), 0)} in the training events.`))
  out.push(bullet('**What works:** destroyed buildings are recognised well within-event, texture change is the most informative signal across all models, and every model beats the non-ML baseline.'))
  out.push(bullet('**What does not:** separating minor from major damage at about 0.5 m per pixel, and transferring to new events. Balanced class weights also push some undamaged buildings into damaged classes.'))
  out.push(bullet('**Explainability is a real benefit.** Every prediction can be traced to features, tree questions or similar examples, which makes these failure modes visible instead of hidden.'))

  // 13 ------------------------------------------------------------ Future scope
  out.push(H1('Future Scope', false))
  for (const b of [
    'Compare learned features (CNN or Siamese networks on pre/post crops) against this classical baseline on the same honest, scene-grouped and event-held-out splits.',
    'Use the original xBD JSON polygons to avoid merged buildings, and add more events and the official test and holdout splits.',
    'Domain adaptation or per-event normalisation that keeps scene-wide damage information, since the scene-median normalisation removed it.',
    'Calibrate the model scores (for example with isotonic regression) so that confidence can be read as a probability.',
    'Georeferenced output for map overlays and integration with real response workflows, with human review built in.',
  ]) out.push(bullet(b))

  // 14 ------------------------------------------------------------ Conclusion
  out.push(H1('Conclusion', false))
  out.push(P(`AfterSight shows that classical, explainable machine learning can perform meaningful building damage triage on xBD. On ${int(fm.n_buildings)} buildings from five disasters, Logistic Regression on 37 hand-crafted features reaches a test macro-F1 of ${f3(lrW.test_macro_f1)} on new scenes of known disasters and explains every decision. The project's main finding is an honest one: on an unseen disaster the score falls to about ${f3(lrX.test_macro_f1)}. The features reveal which disaster a building comes from, but removing that signature does not restore performance, so the gap reflects a genuine domain shift.`))
  out.push(P('Beyond the models, the project delivers a reproducible pipeline, leakage-checked splits, a test-set guard, 171 automated checks in total and a complete web application. Its Triage Desk, Case File and Model Arena make both the strengths and the limits of the approach visible to users and examiners.'))

  // 15 ------------------------------------------------------------ References
  out.push(H1('References', false))
  for (const r of [
    '[1] R. Gupta, R. Hosfelt, S. Sajeev, N. Patel, B. Goodman, J. Doshi, E. Heim, H. Choset and M. Gaston, “xBD: A Dataset for Assessing Building Damage from Satellite Imagery,” arXiv:1911.09296, 2019.',
    '[2] R. M. Haralick, K. Shanmugam and I. Dinstein, “Textural Features for Image Classification,” IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973, doi: 10.1109/TSMC.1973.4309314.',
    '[3] A. J. Cooner, Y. Shao and J. B. Campbell, “Detection of Urban Damage Using Remote Sensing and Machine Learning Algorithms: Revisiting the 2010 Haiti Earthquake,” Remote Sensing, vol. 8, no. 10, art. 868, 2016, doi: 10.3390/rs8100868.',
    '[4] E. Weber and H. Kané, “Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion,” ICLR 2020 Workshop on AI for Earth Sciences, arXiv:2004.05525, 2020.',
    '[5] R. Gupta and M. Shah, “RescueNet: Joint Building Segmentation and Damage Assessment from Satellite Imagery,” in Proc. 25th International Conference on Pattern Recognition (ICPR), 2020, arXiv:2004.07312.',
    '[6] J. Canny, “A Computational Approach to Edge Detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679–698, 1986, doi: 10.1109/TPAMI.1986.4767851.',
    '[7] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
    '[8] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.',
    '[9] P. J. Rousseeuw, “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987, doi: 10.1016/0377-0427(87)90125-7.',
    '[10] L. Hubert and P. Arabie, “Comparing Partitions,” Journal of Classification, vol. 2, no. 1, pp. 193–218, 1985.',
    '[11] xView2 Challenge, “xBD Dataset,” [Online]. Available: https://xview2.org/dataset. Licence: CC BY-NC-SA 4.0.',
  ]) out.push(new Paragraph({ children: runs(r, { size: 22 }), indent: { left: 560, hanging: 560 }, spacing: { after: 120, line: 300 }, alignment: AlignmentType.LEFT }))
  return out
}

// ---------------------------------------------------------------- assemble
const bodyChildren = body() // fills toc / figures / tables registries first
const footer = (fmt) => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] })
const doc = new Document({
  creator: 'AfterSight project group',
  title: 'AfterSight — Mini Project Report',
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, font: FONT }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] }] },
  sections: [
    { properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: cover() },
    {
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer() },
      children: [...listPage('Table of Contents', toc), pageBreak(), ...listPage('List of Figures', figures.map((t) => ({ level: 2, text: t }))), pageBreak(),
        ...listPage('List of Tables', tables.map((t) => ({ level: 2, text: t })))],
    },
    {
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: footer() },
      children: bodyChildren,
    },
  ],
})
const out = path.join(__dirname, '..', 'AfterSight_Mini_Project_Report.docx')
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf)
  fs.writeFileSync(path.join(__dirname, 'registry.json'), JSON.stringify({ toc, figures, tables }, null, 1))
  console.log(`wrote ${out} · ${toc.length} headings · ${figures.length} figures · ${tables.length} tables · pages known: ${Object.keys(PAGES).length}`)
})
