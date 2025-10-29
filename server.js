// server.js — RAG + Deterministic KB Answer + Short General Fallback

const path = require("path");
const fs = require("fs");
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const glob = require("glob");
const removeMd = require("remove-markdown");
const natural = require("natural");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

const PORT = process.env.PORT || 3000;
const LM_BASE_URL = process.env.LM_BASE_URL || "http://127.0.0.1:1234/v1";
const LM_MODEL = process.env.LM_MODEL || "llama-3.2-1b-instruct";

const KB_DIR = process.env.KB_DIR || "kb";
const TOP_K = parseInt(process.env.TOP_K || "4", 10);
const STRICT_THRESHOLD = parseInt(process.env.STRICT_THRESHOLD || "1", 10);

// ---------- Utils ----------
function normalizeTH(s = "") {
  return s.replace(/[\u200B-\u200D\uFEFF]/g, "").normalize("NFC");
}
function tokenizeTH(s = "") {
  return (normalizeTH(s).match(/[\p{L}\p{N}_\-]+/gu) || [])
    .map(t => t.trim())
    .filter(t => t.length >= 2);
}

// ตัดประโยคที่เกี่ยวข้องที่สุดจากเอกสาร
function extractSnippet(query, text) {
  const toks = tokenizeTH(query);
  const hay = normalizeTH(text);
  const sentences = hay
    .split(/[\n\r]+|(?<=[\.\!\?])\s+|(?<=[\u0E2F\u0E46])\s*/g)
    .filter(Boolean);

  let best = "";
  let bestScore = -1;
  for (const s of sentences) {
    let score = 0;
    for (const w of toks) if (s.includes(w)) score++;
    if (score > bestScore) {
      bestScore = score; best = s.trim();
    }
  }
  if (!best) best = hay.slice(0, 220);
  return best;
}

// ---------- Simple intents (ตอบเองแบบสั้นๆ ไม่เรียกโมเดล) ----------
function isGreeting(msg = "") {
  const t = msg.trim().toLowerCase();
  return /^(สวัสดี|ดีจ้า|ดีครับ|ดีค่ะ|หวัดดี|hello|hi|hey|yo)\!?$/.test(t);
}
function isThanks(msg = "") {
  const t = msg.trim().toLowerCase();
  return /^(ขอบคุณ|ขอบคุณครับ|ขอบคุณค่ะ|ขอบใจ|thanks|thx|thank you)\!?$/.test(t);
}
function isBye(msg = "") {
  const t = msg.trim().toLowerCase();
  return /^(บ๊ายบาย|ลาแล้ว|ลาละ|goodbye|bye|see ya)\!?$/.test(t);
}
function quickReplyIfAny(msg) {
  if (isGreeting(msg)) return "สวัสดีค่ะ/ครับ";
  if (isThanks(msg))   return "ยินดีค่ะ/ครับ";
  if (isBye(msg))      return "สวัสดีค่ะ/ครับ";
  return null;
}

// ---------- Load KB ----------
let KB = { chunks: [], tfidf: new natural.TfIdf() };

function readTextFile(fp) {
  const raw = fs.readFileSync(fp, "utf-8");
  return normalizeTH(fp.toLowerCase().endsWith(".md") ? removeMd(raw) : raw);
}

function buildIndex() {
  const kbPath = path.join(__dirname, KB_DIR);
  if (!fs.existsSync(kbPath)) fs.mkdirSync(kbPath, { recursive: true });

  const pattern = path.join(kbPath, "**", "*.{txt,md}").replace(/\\/g, "/");
  const files = glob.sync(pattern, { nodir: true });

  const tfidf = new natural.TfIdf();
  const chunks = [];
  let counter = 0;

  for (const fp of files) {
    const text = readTextFile(fp).replace(/\s+/g, " ").trim();
    if (!text) continue;
    const rel = path.relative(__dirname, fp);
    chunks.push({ id: `doc_${counter++}`, text, source: rel, idx: 0 });
    tfidf.addDocument(text);
  }

  KB = { chunks, tfidf };
  console.log(`📚 Indexed ${files.length} files from ${KB_DIR}/`);
}

// ---------- Retrieval ----------
function scoreDocs(query) {
  if (!KB.chunks.length) return [];
  const q = normalizeTH(query);
  const qTokens = tokenizeTH(q);

  const tfidfScores = KB.chunks.map(() => 0);
  KB.tfidf.tfidfs(q, (i, v) => (tfidfScores[i] = v || 0));

  function overlapCount(txt) {
    const t = normalizeTH(txt);
    let c = 0; for (const w of qTokens) if (t.includes(w)) c++;
    return c;
  }

  return KB.chunks.map((c, i) => {
    const overlap = overlapCount(c.text);
    const score = overlap * 5 + tfidfScores[i];
    return { ...c, overlap, tfidf: tfidfScores[i], score };
  });
}

function retrieveTopK(query, k = TOP_K) {
  const scored = scoreDocs(query);
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(1, k));
}

function buildContext(docs) {
  return docs.map((d, i) => `[${i + 1}] Source: ${d.source}\n${d.text}`).join("\n\n");
}

// ---------- Routes ----------
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    model: LM_MODEL,
    baseUrl: LM_BASE_URL,
    kb_files: KB.chunks.length,
    top_k: TOP_K,
    strict_threshold: STRICT_THRESHOLD,
  });
});

app.post("/api/reindex", (_req, res) => {
  buildIndex();
  res.json({ ok: true, chunks: KB.chunks.length });
});

app.get("/api/debug_context", (req, res) => {
  const q = req.query.q || "";
  const hits = retrieveTopK(q, TOP_K);
  res.json({
    query: q,
    hits: hits.map((h, i) => ({
      rank: i + 1, source: `${h.source}#${h.idx}`,
      overlap: h.overlap, tfidf: Number(h.tfidf.toFixed(4)),
      score: Number(h.score.toFixed(4)),
      preview: h.text.slice(0, 160) + (h.text.length > 160 ? "..." : "")
    }))
  });
});

// ---------- Chat ----------
app.post("/api/chat", async (req, res) => {
  const msg = req.body?.message?.trim();
  if (!msg) return res.status(400).json({ error: "message required" });

  // 0) เคสตอบสั้นๆ (ไม่เรียกโมเดล)
  const quick = quickReplyIfAny(msg);
  if (quick) return res.json({ reply: quick });

  // 1) ลองค้น KB
  const hits = retrieveTopK(msg, TOP_K);
  const maxOverlap = hits.length ? Math.max(...hits.map(h => h.overlap)) : 0;
  const strictKB = maxOverlap >= STRICT_THRESHOLD;

  console.log("🧩 Context sources:", hits.map(h => h.source), "| strictKB:", strictKB, "| maxOverlap:", maxOverlap);

  // 2) ถ้าเจอใน KB ชัดเจน → ตอบจากไฟล์ทันที
  if (strictKB && hits.length) {
    const best = hits[0];
    const snippet = extractSnippet(msg, best.text);
    const reply = `${snippet} [1]`;
    return res.json({
      reply,
      meta: {
        mode: "DETERMINISTIC_KB",
        used_context: hits.map((h, i) => ({
          idx: i + 1, source: `${h.source}#${h.idx}`,
          overlap: h.overlap, tfidf: Number(h.tfidf.toFixed(4)),
          score: Number(h.score.toFixed(4))
        }))
      }
    });
  }

  // 3) ไม่เจอ → ให้โมเดลตอบ "สั้น/ตรงคำถาม" (ห้ามพูดถึงเอกสาร/กติกา)
  const contextDocs = hits.length ? hits : KB.chunks;
  const context = buildContext(contextDocs);
  const messages = [
    {
      role: "system",
      content:
        "ตอบเป็นภาษาไทยแบบสั้น กระชับ และตรงคำถามเท่านั้น (ไม่เกิน 1 ประโยค). " +
        "ห้ามอธิบายกติกาการตอบ ห้ามกล่าวถึงเอกสารอ้างอิงหรือขั้นตอนใด ๆ."
    },
    {
      role: "user",
      content:
        `ข้อมูลเพิ่มเติม (สำหรับช่วยคิดเท่านั้น ห้ามกล่าวถึง):\n${context}\n\n` +
        `คำถาม: ${msg}\n` +
        `ตอบให้สั้นและตรงคำถามใน 1 ประโยค`
    }
  ];

  try {
    const r = await axios.post(
      `${LM_BASE_URL}/chat/completions`,
      { model: LM_MODEL, messages, temperature: 0.2, max_tokens: 80 },
      { timeout: 60_000 }
    );
    const reply = r.data?.choices?.[0]?.message?.content || "(no reply)";
    return res.json({
      reply,
      meta: {
        mode: "GENERAL_FALLBACK",
        used_context: contextDocs.map((h, i) => ({
          idx: i + 1, source: `${h.source}#${h.idx}`,
          overlap: h.overlap ?? 0, tfidf: Number(h.tfidf?.toFixed?.(4) ?? 0),
          score: Number(h.score?.toFixed?.(4) ?? 0)
        }))
      }
    });
  } catch (err) {
    console.error("❌ LM error:", err.response?.data || err.message);
    return res.status(500).json({ error: "LM request failed", detail: err.response?.data || err.message });
  }
});

// ---------- Start ----------
buildIndex();
app.listen(PORT, () => {
  console.log(`🚀 http://localhost:${PORT} | Model ${LM_MODEL}`);
});
