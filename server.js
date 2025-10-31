// server.js — RAG + Deterministic KB Answer + Short General Fallback + Self-Name Memory

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

// ============== Memory (learn "who am I") ==============
const MEM_PATH = path.join(__dirname, KB_DIR, "_memory.json");
const MEM_TXT  = path.join(__dirname, KB_DIR, "_memory_notes.txt");
let MEM = { self_name: null, company_alias: {}, facts: [] };

function ensureKbDir() {
  const kbPath = path.join(__dirname, KB_DIR);
  if (!fs.existsSync(kbPath)) fs.mkdirSync(kbPath, { recursive: true });
}
function loadMemory() {
  ensureKbDir();
  try {
    if (fs.existsSync(MEM_PATH)) MEM = JSON.parse(fs.readFileSync(MEM_PATH, "utf-8"));
  } catch {}
}
function saveMemory() {
  try {
    fs.writeFileSync(MEM_PATH, JSON.stringify(MEM, null, 2), "utf-8");
    const lines = [];
    if (MEM.self_name) lines.push(`SELF_NAME: ${MEM.self_name}`);
    for (const f of MEM.facts) lines.push(`FACT: ${f}`);
    fs.writeFileSync(MEM_TXT, lines.join("\n"), "utf-8");
  } catch {}
}
function learnSelfName(name) {
  if (!name) return;
  if (MEM.self_name === name) return;
  MEM.self_name = name;
  MEM.facts = MEM.facts || [];
  if (!MEM.facts.includes(`ผู้ใช้คนนี้คือ ${name}`)) MEM.facts.push(`ผู้ใช้คนนี้คือ ${name}`);
  saveMemory();
  buildIndex(); // ให้ TF-IDF เห็น _memory_notes.txt
}
loadMemory();

// ---------- Utils ----------
function normalizeTH(s = "") { return s.replace(/[\u200B-\u200D\uFEFF]/g, "").normalize("NFC"); }
function tokenizeTH(s = "") {
  return (normalizeTH(s).match(/[\p{L}\p{N}_\-]+/gu) || []).map(t => t.trim()).filter(t => t.length >= 2);
}
function extractSnippet(query, text) {
  const toks = tokenizeTH(query);
  const hay = normalizeTH(text);
  const sentences = hay.split(/[\n\r]+|(?<=[\.\!\?])\s+|(?<=[\u0E2F\u0E46])\s*/g).filter(Boolean);
  let best = "", bestScore = -1;
  for (const s of sentences) {
    let score = 0; for (const w of toks) if (s.includes(w)) score++;
    if (score > bestScore) { bestScore = score; best = s.trim(); }
  }
  if (!best) best = hay.slice(0, 220);
  return best;
}

// helper: ตัดสินว่าเป็นไฟล์ _memory_notes.txt หรือไม่ (ข้ามได้ชัวร์ทุก OS)
function isMemoryNotes(src = "") {
  return path.basename(src).toLowerCase() === "_memory_notes.txt";
}

// ---------- Quick intents ----------
function isGreeting(msg=""){return /^(สวัสดี|ดีจ้า|ดีครับ|ดีค่ะ|หวัดดี|hello|hi|hey|yo)\!?$/i.test(msg.trim());}
function isThanks(msg=""){return /^(ขอบคุณ|ขอบคุณครับ|ขอบคุณค่ะ|ขอบใจ|thanks|thx|thank you)\!?$/i.test(msg.trim());}
function isBye(msg=""){return /^(บ๊ายบาย|ลาแล้ว|ลาละ|goodbye|bye|see ya)\!?$/i.test(msg.trim());}
function quickReplyIfAny(msg){ if(isGreeting(msg))return"สวัสดีค่ะ/ครับ"; if(isThanks(msg))return"ยินดีค่ะ/ครับ"; if(isBye(msg))return"สวัสดีค่ะ/ครับ"; return null; }

// ---------- Intent: who is / who am I ----------
const RE_WHO_IS=/^(?:ใครคือ|who\s*is)\s*([^\?\n\r]+)$/i;
const RE_WHO_AM_I=/^(?:ฉัน|ผม|หนู|ดิฉัน|เรา|i)\s*(?:คือ)?\s*ใคร\??$/i;
function parseWhoIs(msg=""){ const m=normalizeTH(msg).match(RE_WHO_IS); return m?m[1].trim():null; }
function isWhoAmI(msg=""){ return RE_WHO_AM_I.test(normalizeTH(msg)); }
function expandQueryIfWhoAmI(originalMsg,userHint){
  if(!isWhoAmI(originalMsg)) return originalMsg;
  const name = MEM.self_name || (userHint||"").trim();
  return name ? `ใครคือ ${name}` : originalMsg;
}

// ---------- KB ----------
let KB = { chunks: [], tfidf: new natural.TfIdf() };
function readTextFile(fp){
  const raw = fs.readFileSync(fp, "utf-8"); // แก้บั๊ก utf-8
  return normalizeTH(fp.toLowerCase().endsWith(".md") ? removeMd(raw) : raw);
}
function buildIndex(){
  ensureKbDir();
  const pattern = path.join(__dirname, KB_DIR, "**", "*.{txt,md}").replace(/\\/g,"/");
  const files = glob.sync(pattern,{nodir:true});
  const tfidf = new natural.TfIdf();
  const chunks = [];
  let counter=0;
  for (const fp of files){
    const text = readTextFile(fp).replace(/\s+/g," ").trim();
    if(!text) continue;
    const rel = path.relative(__dirname, fp);
    chunks.push({ id:`doc_${counter++}`, text, source: rel, idx: 0 });
    tfidf.addDocument(text);
  }
  KB={chunks, tfidf};
  console.log(`📚 Indexed ${files.length} files from ${KB_DIR}/`);
}

// ---------- Retrieval ----------
function scoreDocs(query){
  if(!KB.chunks.length) return [];
  const q = normalizeTH(query);
  const qTokens = tokenizeTH(q);
  const tfidfScores = KB.chunks.map(()=>0);
  KB.tfidf.tfidfs(q,(i,v)=> (tfidfScores[i]=v||0));
  function overlapCount(txt){ const t=normalizeTH(txt); let c=0; for(const w of qTokens) if(t.includes(w)) c++; return c; }
  return KB.chunks.map((c,i)=>{ const overlap=overlapCount(c.text); const score=overlap*5+tfidfScores[i]; return {...c,overlap,tfidf:tfidfScores[i],score}; });
}
function retrieveTopK(query,k=TOP_K){ const scored=scoreDocs(query); scored.sort((a,b)=>b.score-a.score); return scored.slice(0,Math.max(1,k)); }
function buildContext(docs){ return docs.map((d,i)=>`[${i+1}] Source: ${d.source}\n${d.text}`).join("\n\n"); }

// ---------- Routes ----------
app.get("/health",(req,res)=>{
  res.json({ ok:true, model:LM_MODEL, baseUrl:LM_BASE_URL, kb_files:KB.chunks.length, top_k:TOP_K,
    strict_threshold:STRICT_THRESHOLD, memory:{ self_name: MEM.self_name, facts_count: MEM.facts?.length||0 }});
});
app.post("/api/reindex",(_req,res)=>{ buildIndex(); res.json({ok:true, chunks:KB.chunks.length}); });
app.get("/api/debug_context",(req,res)=>{
  const q=req.query.q||""; const hits=retrieveTopK(q,TOP_K);
  res.json({ query:q, hits:hits.map((h,i)=>({rank:i+1, source:`${h.source}#${h.idx}`, overlap:h.overlap, tfidf:+h.tfidf.toFixed(4), score:+h.score.toFixed(4), preview:h.text.slice(0,160)+(h.text.length>160?"...":"") })) });
});
app.get("/api/memory",(_req,res)=>{ res.json({ ok:true, memory: MEM }); });
app.post("/api/memory/reset",(_req,res)=>{ MEM={ self_name:null, company_alias:{}, facts:[] }; saveMemory(); buildIndex(); res.json({ ok:true, reset:true }); });

// ---------- Chat ----------
app.post("/api/chat", async (req, res) => {
  const msg = req.body?.message?.trim();
  const userHint = (req.body?.user_hint || "").trim();
  if (!msg) return res.status(400).json({ error: "message required" });

  // 0) quick replies
  const quick = quickReplyIfAny(msg);
  if (quick) return res.json({ reply: quick });

  const normalized = normalizeTH(msg);

  // ✅ NEW: “บริษัท TRANSDEV.CO.TH คือบริษัทอะไร” → ดึงจาก KB (เช่น company_notes.md)
  const isAskCompany =
    /บริษัท\s*transdev\.co\.th.*(?:คือ(?:บริษัท)?อะไร|คืออะไร)?\??$/i.test(normalized) ||
    /^บริษัท\s*transdev\.co\.th$/i.test(normalized);

  const askedWho = parseWhoIs(msg);

  // ✅ "ฉันคือใคร" → ตอบสั้น 1 บรรทัด
  if (isWhoAmI(msg)) {
    const name = MEM.self_name || userHint || "";
    if (name) {
      learnSelfName(name);
      return res.json({ reply: `ผู้ใช้คนนี้คือ ${name}`, meta: { mode: "WHOAMI_DIRECT" } });
    }
  }

  // ขยาย query หากถามตัวเอง
  const expandedMsg = expandQueryIfWhoAmI(msg, userHint);

  // 1) Retrieval
  const hits = retrieveTopK(expandedMsg, TOP_K);
  const maxOverlap = hits.length ? Math.max(...hits.map(h => h.overlap)) : 0;
  const strictKB = maxOverlap >= STRICT_THRESHOLD;

  console.log("🧩 Context sources:", hits.map(h => h.source), "| strictKB:", strictKB, "| maxOverlap:", maxOverlap, "| expanded:", expandedMsg !== msg, "| askedWho:", askedWho || "-", "| askCompany:", !!isAskCompany);

  // ✅ ถามบริษัท TRANSDEV → คืน “ข้อความเต็มจากไฟล์ KB” (เช่น company_notes.md), ไม่ใช้ _memory_notes.txt
  if (isAskCompany) {
    let companyDoc = (hits || []).find(h => !isMemoryNotes(h.source) && /transdev\.co\.th/i.test(h.text));
    if (!companyDoc) {
      companyDoc = (KB.chunks || []).find(c => !isMemoryNotes(c.source) && /transdev\.co\.th/i.test(c.text));
    }
    if (companyDoc) {
      return res.json({
        reply: companyDoc.text.trim(),
        meta: { mode: "COMPANY_PROFILE", source: `${companyDoc.source}#${companyDoc.idx || 0}` }
      });
    }
    // ถ้าไม่พบในไฟล์ ปล่อยไปขั้นตอนต่อไป (KB deterministic / LLM)
  }

  // ✅ "ใครคือ <ชื่อ>" → คืน “โปรไฟล์เต็ม” จากไฟล์ KB (ไม่ใช้ _memory_notes.txt)
  if (askedWho) {
    let profileDoc = (hits || []).find(h => !isMemoryNotes(h.source) && h.text && h.text.includes(askedWho));
    if (!profileDoc) {
      profileDoc = (KB.chunks || []).find(c => !isMemoryNotes(c.source) && c.text && c.text.includes(askedWho));
    }
    if (profileDoc) {
      learnSelfName(askedWho);
      return res.json({
        reply: profileDoc.text.trim(),
        meta: { mode: "WHOIS_PROFILE", source: `${profileDoc.source}#${profileDoc.idx || 0}` }
      });
    }
  }

  // 2) ถ้าเจอ KB ชัดเจน → ตอบแบบสไนเป็ต
  if (strictKB && hits.length) {
    const best = hits[0];
    const snippet = extractSnippet(expandedMsg, best.text);

    if (askedWho) learnSelfName(askedWho);
    if (isWhoAmI(msg) && userHint) learnSelfName(userHint);

    return res.json({
      reply: `${snippet} [1]`,
      meta: {
        mode: "DETERMINISTIC_KB",
        expanded_from: expandedMsg !== msg ? msg : null,
        used_context: hits.map((h, i) => ({
          idx: i + 1, source: `${h.source}#${h.idx}`,
          overlap: h.overlap, tfidf: Number(h.tfidf.toFixed(4)),
          score: Number(h.score.toFixed(4))
        }))
      }
    });
  }

  // 3) Fallback ให้โมเดลสรุปสั้นๆ
  const contextDocs = hits.length ? hits : KB.chunks;
  const context = buildContext(contextDocs);
  const messages = [
    {
      role: "system",
      content: "ตอบเป็นภาษาไทยแบบสั้น กระชับ และตรงคำถามเท่านั้น (ไม่เกิน 1 ประโยค). ห้ามอธิบายกติกาการตอบ ห้ามกล่าวถึงเอกสารอ้างอิงหรือขั้นตอนใด ๆ."
    },
    {
      role: "user",
      content: `ข้อมูลเพิ่มเติม (สำหรับช่วยคิดเท่านั้น ห้ามกล่าวถึง):\n${context}\n\nคำถาม: ${expandedMsg}\nตอบให้สั้นและตรงคำถามใน 1 ประโยค`
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
        expanded_from: expandedMsg !== msg ? msg : null,
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
app.listen(PORT, ()=>{ console.log(`🚀 http://localhost:${PORT} | Model ${LM_MODEL}`); });
