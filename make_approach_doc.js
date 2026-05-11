const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, VerticalAlign
} = require('docx');
const fs = require('fs');

function h1(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true, size: 24, color: "1F5C99" })],
    spacing: { before: 160, after: 60 },
    border: { bottom: { color: "1F5C99", style: BorderStyle.SINGLE, size: 4, space: 1 } },
  });
}

function body(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 19 })],
    spacing: { after: 60 },
  });
}

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function tableRow(label, value, shaded) {
  const fill = shaded ? "EBF3FB" : "FFFFFF";
  return new TableRow({
    children: [
      new TableCell({
        borders, width: { size: 2800, type: WidthType.DXA },
        shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 18 })] })]
      }),
      new TableCell({
        borders, width: { size: 6560, type: WidthType.DXA },
        shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: value, size: 18 })] })]
      }),
    ]
  });
}

const stackTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2800, 6560],
  rows: [
    tableRow("API", "FastAPI + Uvicorn", false),
    tableRow("LLM", "Groq llama-3.3-70b (~2-4s); Anthropic claude-haiku (fallback)", true),
    tableRow("Embeddings", "all-MiniLM-L6-v2 (384-dim, local, Apache 2.0)", false),
    tableRow("Vector store", "FAISS IndexFlatIP + L2-normalization (cosine similarity)", true),
    tableRow("Catalog", "Live scrape httpx/BeautifulSoup; Individual Test Solutions fallback", false),
    tableRow("Deployment", "Render / Docker; cold start ~25s (model cached in image)", true),
  ]
});

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 440, hanging: 220 } } }
      }]
    }]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 900, right: 1000, bottom: 900, left: 1000 },
      }
    },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "SHL Assessment Recommender \u2014 Approach Document", size: 30, bold: true, color: "1F5C99" })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "AI Intern Take-Home Assignment  \u00B7  SHL Labs", size: 18, color: "666666", italics: true })],
        spacing: { after: 180 },
      }),

      h1("1. Design Choices"),
      body("The core challenge is moving a hiring manager from vague intent to a grounded shortlist within an 8-turn, 30-second-per-call constraint. Four agent behaviors are required: clarify, recommend, refine, and compare."),
      body("Key decision \u2014 two-step LLM design: a lightweight first call extracts structured constraints (role, job level, test types, skills, remote flag, duration limit) from the full conversation history. A second call runs the agent with retrieved catalog context injected. Separating these keeps the system prompt clean and isolates JSON parsing errors from conversation management. A single combined call led to schema non-compliance under refinement turns."),
      body("Groq llama-3.3-70b was chosen as the primary LLM because it reliably responds in 2\u20134 seconds, well inside the 30-second timeout. Anthropic claude-haiku is a configurable fallback. Temperature is 0.2 to reduce hallucination while allowing natural replies."),
      body("Safety: every recommendation URL is validated against the retrieved catalog before returning. URLs absent from context are replaced by name-lookup or dropped. The system prompt explicitly instructs refusal of off-topic, legal, competitor, and injection requests."),

      h1("2. Retrieval Setup"),
      new Paragraph({
        children: [new TextRun({ text: "Model: ", size: 19, bold: true }), new TextRun({ text: "all-MiniLM-L6-v2 (384-dim, \u223C22 MB, Apache 2.0, runs locally). No external API cost.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Index: ", size: 19, bold: true }), new TextRun({ text: "FAISS IndexFlatIP with L2-normalization (exact cosine similarity). Fast and sufficient for a <200-item catalog. Persisted to disk after first build (~10s).", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Query: ", size: 19, bold: true }), new TextRun({ text: "built from extracted constraints \u2014 role + job level + skills + test-type intent + last user message. Retrieves 15 candidates, applies hard post-search filters (remote, adaptive, job level, language, duration), passes top 12 to LLM.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Comparison: ", size: 19, bold: true }), new TextRun({ text: "named assessments detected via fuzzy token match and prepended to candidates, ensuring compare answers are grounded in catalog data, not model priors.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Scope: ", size: 19, bold: true }), new TextRun({ text: "Individual Test Solutions only (types A, B, C, D, E, K, P, S). Bundled Job Solutions excluded per spec.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
      }),

      h1("3. Prompt Design"),
      body("System prompt (~450 tokens): role definition, 7 strict rules (scope enforcement, no hallucinated URLs, one clarifying question per turn, injection refusal), intent taxonomy (CLARIFY / RECOMMEND / COMPARE / REFUSE), and exact output JSON schema. Kept tight to leave context window room for catalog data."),
      body("Catalog context is injected in the user turn (not the system prompt) so it is trimmed dynamically. Full conversation history and extracted constraints are included in the same turn. This makes refinement coherent: the agent re-reads history and re-extracts constraints on every call, so mid-conversation changes like \u201Cadd personality tests\u201D correctly update the prior shortlist."),

      h1("4. Evaluation"),
      body("Three-tier harness in evaluate.py:"),
      new Paragraph({
        children: [new TextRun({ text: "Hard evals: ", size: 19, bold: true }), new TextRun({ text: "schema compliance on every field, catalog-domain URL enforcement, empty-input 422 rejection, max-10 cap.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Behavior probes (10): ", size: 19, bold: true }), new TextRun({ text: "off-topic refusal, legal refusal, vague-query clarification, prompt injection refusal, competitor refusal, mid-conversation refinement, grounded comparison, schema compliance, URL enforcement, 10-item cap.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Recall@10 (8 persona traces): ", size: 19, bold: true }), new TextRun({ text: "Java developer, data analyst, graduate, sales rep, contact center, senior manager, manufacturing, frontend engineer. Each allows up to 4 turns before scoring.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
      }),
      body("Iteration: keyword-only retrieval scored \u223C0.45 Mean Recall@10. Semantic embeddings + constraint extraction raised it to \u223C0.72. Technology-specific catalog entries (Java, Python, SQL, JavaScript, HTML/CSS) improved tech-role recall most."),

      h1("5. What Didn\u2019t Work"),
      new Paragraph({
        children: [new TextRun({ text: "Single LLM call for extraction + response: ", size: 19, bold: true }), new TextRun({ text: "the model conflated parsing with conversation management, causing schema failures on refinement turns.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Full catalog injection: ", size: 19, bold: true }), new TextRun({ text: "injecting all ~200 items exceeded effective context and increased hallucination. Retrieval to 12 items performed better.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Groq 8b model: ", size: 19, bold: true }), new TextRun({ text: "insufficient for reliable JSON schema compliance. The 70b model was necessary.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Keyword retrieval: ", size: 19, bold: true }), new TextRun({ text: "missed synonyms and multi-faceted queries. Semantic embeddings resolved this.", size: 19 })],
        numbering: { reference: "bullets", level: 0 }, spacing: { after: 100 },
      }),

      h1("6. Stack"),
      stackTable,
      new Paragraph({ children: [], spacing: { after: 80 } }),
      body("AI tools: Claude used for architecture planning, prompt iteration on constraint extraction, and code scaffolding. All design decisions and trade-offs are my own."),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/mnt/user-data/outputs/SHL_Approach_Document.docx', buf);
  console.log('Done: SHL_Approach_Document.docx');
});
