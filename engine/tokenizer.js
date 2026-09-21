/** Byte-level BPE matching HuggingFace tokenizers (PetitGPT tokenizer.json). */

const PAT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteToUni = new Array(256);
  const uniToByte = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToUni[bs[i]] = ch;
    uniToByte.set(cs[i], bs[i]);
  }
  return { byteToUni, uniToByte };
}

const { byteToUni, uniToByte } = bytesToUnicode();

function encodeUtf8(str) {
  return new TextEncoder().encode(str);
}

function bpe(token, ranks) {
  if (token.length === 0) return [];
  let word = Array.from(token);
  if (word.length === 1) return word;
  while (word.length > 1) {
    let minRank = Infinity;
    let minI = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const key = word[i] + "\0" + word[i + 1];
      const r = ranks.get(key);
      if (r !== undefined && r < minRank) {
        minRank = r;
        minI = i;
      }
    }
    if (minI < 0) break;
    const merged = word[minI] + word[minI + 1];
    word = word.slice(0, minI).concat(merged, word.slice(minI + 2));
  }
  return word;
}

export function loadTokenizerFromJson(spec) {
  const vocab = spec.model.vocab;
  const unkId = vocab[spec.model.unk_token] ?? 1;
  const ranks = new Map();
  const merges = spec.model.merges || [];
  for (let i = 0; i < merges.length; i++) {
    const m = merges[i];
    const a = Array.isArray(m) ? m[0] : m.split(" ")[0];
    const b = Array.isArray(m) ? m[1] : m.split(" ").slice(1).join(" ");
    ranks.set(a + "\0" + b, i);
  }
  const special = new Map();
  for (const t of spec.added_tokens || []) {
    if (t.special) special.set(t.content, t.id);
  }

  const specialList = [...special.keys()].sort((a, b) => b.length - a.length);

  function encodeRaw(text) {
    const parts = text.match(PAT) || (text ? [text] : []);
    const ids = [];
    for (const part of parts) {
      const bytes = encodeUtf8(part);
      let mapped = "";
      for (let i = 0; i < bytes.length; i++) mapped += byteToUni[bytes[i]];
      const pieces = bpe(mapped, ranks);
      for (const p of pieces) {
        const id = vocab[p];
        ids.push(id === undefined ? unkId : id);
      }
    }
    return ids;
  }

  function encode(text) {
    if (!specialList.length || !text) return encodeRaw(text);
    const ids = [];
    let i = 0;
    while (i < text.length) {
      let hit = null;
      for (const s of specialList) {
        if (text.startsWith(s, i)) {
          hit = s;
          break;
        }
      }
      if (hit) {
        ids.push(special.get(hit));
        i += hit.length;
        continue;
      }
      let j = i + 1;
      while (j <= text.length) {
        let next = false;
        for (const s of specialList) {
          if (text.startsWith(s, j)) {
            next = true;
            break;
          }
        }
        if (next) break;
        j++;
      }
      ids.push(...encodeRaw(text.slice(i, j)));
      i = j;
    }
    return ids;
  }

  function decode(ids, skipSpecial = false) {
    let mapped = "";
    for (const id of ids) {
      let piece = null;
      if (skipSpecial) {
        let isSpecial = false;
        for (const [, sid] of special) if (sid === id) isSpecial = true;
        if (isSpecial) continue;
      }
      for (const [tok, vid] of Object.entries(vocab)) {
        if (vid === id) {
          piece = tok;
          break;
        }
      }
      if (piece == null) continue;
      mapped += piece;
    }
    const bytes = [];
    for (let i = 0; i < mapped.length; i++) {
      const b = uniToByte.get(mapped.charCodeAt(i));
      if (b !== undefined) bytes.push(b);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes));
  }

  // Reverse vocab for faster decode
  const idToTok = new Array(Object.keys(vocab).length);
  for (const [tok, id] of Object.entries(vocab)) idToTok[id] = tok;

  const specialIds = new Set([...special.values()]);

  function decodeFast(ids, skipSpecial = false) {
    let mapped = "";
    for (const id of ids) {
      if (skipSpecial && specialIds.has(id)) continue;
      const piece = idToTok[id];
      if (piece) mapped += piece;
    }
    const bytes = [];
    for (let i = 0; i < mapped.length; i++) {
      const b = uniToByte.get(mapped.charCodeAt(i));
      if (b !== undefined) bytes.push(b);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes));
  }

  return {
    encode,
    decode: decodeFast,
    unkId,
    special,
    vocabSize: Object.keys(vocab).length,
  };
}

export const SPECIAL = {
  PAD: 0,
  UNK: 1,
  BOS: 2,
  EOS: 3,
  SYSTEM: 4,
  USER: 5,
  ASSISTANT: 6,
};

export function encodeChat(tokenizer, messages, { defaultSystem = null } = {}) {
  return encodeMessages(tokenizer, messages, { template: "petitgpt", defaultSystem });
}

export function encodeMessages(tokenizer, messages, card = {}) {
  const msgs = messages.map((m) => ({
    role: String(m.role).trim().toLowerCase(),
    content: String(m.content),
  }));
  const tpl = card.template || "petitgpt";
  if (tpl === "completion") {
    const last = msgs[msgs.length - 1];
    return tokenizer.encode(last?.content || "");
  }
  if (tpl === "chatml") {
    const sys = card.defaultSystem;
    if (sys && (!msgs.length || msgs[0].role !== "system")) {
      msgs.unshift({ role: "system", content: sys });
    }
    const imStart =
      tokenizer.special.get("<|im_start|>") ??
      card.specials?.["<|im_start|>"] ??
      card.bosId ??
      1;
    const imEnd =
      tokenizer.special.get("<|im_end|>") ??
      card.specials?.["<|im_end|>"] ??
      card.eosId ??
      2;
    const ids = [];
    for (const m of msgs) {
      ids.push(imStart);
      ids.push(...tokenizer.encode(m.role));
      ids.push(...tokenizer.encode("\n"));
      ids.push(...tokenizer.encode(m.content));
      ids.push(imEnd);
      ids.push(...tokenizer.encode("\n"));
    }
    ids.push(imStart);
    ids.push(...tokenizer.encode("assistant"));
    ids.push(...tokenizer.encode("\n"));
    return ids;
  }
  const defaultSystem = card.defaultSystem || null;
  if (defaultSystem && (!msgs.length || msgs[0].role !== "system")) {
    msgs.unshift({ role: "system", content: defaultSystem });
  }
  const bos = card.bosId ?? SPECIAL.BOS;
  const eos = card.eosId ?? SPECIAL.EOS;
  const roleId = {
    system: card.specials?.["<|system|>"] ?? SPECIAL.SYSTEM,
    user: card.specials?.["<|user|>"] ?? SPECIAL.USER,
    assistant: card.specials?.["<|assistant|>"] ?? SPECIAL.ASSISTANT,
  };
  const ids = [bos];
  for (const m of msgs) {
    if (roleId[m.role] == null) throw new Error(`invalid role ${m.role}`);
    const contentIds = tokenizer.encode(m.content);
    if (!contentIds.length) throw new Error("empty message encoding");
    ids.push(roleId[m.role]);
    ids.push(...contentIds);
    if (m.role === "assistant") ids.push(eos);
  }
  if (msgs[msgs.length - 1]?.role !== "user") {
    throw new Error("prompt must end with a user turn");
  }
  ids.push(roleId.assistant);
  return ids;
}
