/** Objective JS-checkable tasks. Scores are pass/fail, not style judgments. */

export const SAMPLE_PROMPTS = [
  { title: "Greeting", text: "Say hello in one sentence." },
  { title: "Paris", text: "What is the capital of France?" },
  { title: "Arithmetic", text: "What is 2 + 2?" },
  { title: "Sky", text: "Why is the sky blue?" },
  { title: "Haiku", text: "Write a haiku about rain." },
  { title: "Python", text: "Write a Python function that adds two numbers." },
  {
    title: "Odd ingredient",
    text: "Which of these ingredients doesn't belong in a recipe for apple pie: apples, pie crust, rusty nails, granulated sugar, egg. Explain why it doesn't belong in the recipe.",
  },
  { title: "Yes/no", text: "Is a cat an animal? Answer yes or no." },
  { title: "Repeat", text: "Repeat the word banana exactly three times." },
  { title: "Alphabet", text: "The first letter of the English alphabet is" },
  { title: "Copy secret", text: "The secret code is ORANGE-42. What is the secret code? Reply with only the code." },
  { title: "Pick animal", text: "Pick the animal from this list: car, dog, house. Answer with one word." },
];

function contains(re) {
  return (text) => ({ pass: re.test(text), detail: re.toString() });
}

function exactTrim(expected) {
  return (text) => {
    const t = text.trim().replace(/[.!"']+$/, "");
    return { pass: t.toLowerCase() === expected.toLowerCase(), detail: `got ${JSON.stringify(t)}` };
  };
}

export const BUILTIN_SUITE = {
  id: "builtin-v1",
  name: "PetitGPT objective suite",
  description:
    "Tasks a ~125M model might complete. Checks are deterministic JS predicates, not human quality scores.",
  maxNewTokens: 48,
  tests: [
    {
      id: "hello",
      prompt: "Say hello in one sentence.",
      check: contains(/hello/i),
    },
    {
      id: "paris",
      prompt: "What is the capital of France?",
      check: contains(/\bparis\b/i),
    },
    {
      id: "alphabet",
      prompt: "The first letter of the English alphabet is",
      check: (text) => {
        const t = text.trim();
        return { pass: /^a\b/i.test(t) || t === "A" || /\bA\b/.test(t.slice(0, 8)), detail: t.slice(0, 40) };
      },
    },
    {
      id: "copy_code",
      prompt: "The secret code is ORANGE-42. What is the secret code? Reply with only the code.",
      check: contains(/ORANGE-42/),
    },
    {
      id: "pick_animal",
      prompt: "Pick the animal from this list: car, dog, house. Answer with one word.",
      check: contains(/\bdog\b/i),
    },
    {
      id: "yes_no_cat",
      prompt: "Is a cat an animal? Answer yes or no.",
      check: (text) => {
        const t = text.trim();
        const yes = /^yes\b/i.test(t);
        const no = /^no\b/i.test(t);
        return { pass: yes && !/^no\b/i.test(t), detail: `starts=${JSON.stringify(t.slice(0, 24))} yes=${yes} no=${no}` };
      },
    },
    {
      id: "banana",
      prompt: "Repeat the word banana exactly three times.",
      check: (text) => {
        const n = (text.toLowerCase().match(/banana/g) || []).length;
        return { pass: n >= 1, score: Math.min(n / 3, 1), detail: `count=${n}` };
      },
    },
    {
      id: "one_plus_one",
      prompt: "What is 1 + 1? Reply with only the number.",
      check: (text) => {
        const t = text.trim();
        return { pass: /\b2\b/.test(t) && !/\b3\b/.test(t), detail: t.slice(0, 60) };
      },
    },
    {
      id: "two_plus_two",
      prompt: "What is 2 + 2?",
      check: (text) => ({ pass: /(^|\D)4(\D|$)/.test(text) && !/2 \+ 2 is 2 \+ 2/.test(text), detail: text.slice(0, 80) }),
    },
    {
      id: "odd_one",
      prompt:
        "Which of these ingredients doesn't belong in a recipe for apple pie: apples, pie crust, rusty nails, granulated sugar, egg. Answer with the ingredient that does not belong.",
      check: contains(/rusty nails/i),
    },
    {
      id: "dog_legs",
      prompt: "How many legs does a typical dog have? Reply with only a digit.",
      check: contains(/\b4\b/),
    },
    {
      id: "stop_hello",
      prompt: "Say hello in one sentence.",
      check: (_text, info) => ({
        pass: info.stopReason === "eos",
        detail: `stop=${info.stopReason} n=${info.generatedIds?.length}`,
      }),
    },
    {
      id: "no_repeat_hello",
      prompt: "Say hello in one sentence.",
      check: (text) => {
        const ids = text.split(/\s+/);
        const grams = [];
        for (let i = 0; i < ids.length - 3; i++) grams.push(ids.slice(i, i + 4).join(" "));
        const frac = grams.length ? 1 - new Set(grams).size / grams.length : 0;
        return { pass: frac < 0.4, detail: `repeat4=${frac.toFixed(2)}` };
      },
    },
    {
      id: "color_word",
      prompt: "Name a common color. Reply with one word only.",
      check: (text) => {
        const t = text.trim().split(/\s+/)[0]?.replace(/[^a-z]/gi, "") || "";
        const colors = /^(red|blue|green|yellow|black|white|orange|purple|brown|pink|gray|grey)$/i;
        return { pass: colors.test(t), detail: t };
      },
    },
    {
      id: "water_freeze",
      prompt: "At what Celsius temperature does water freeze at standard pressure? Reply with a number.",
      check: contains(/\b0\b/),
    },
    {
      id: "earth_moon",
      prompt: "What natural satellite orbits the Earth? Answer in one word.",
      check: contains(/\bmoon\b/i),
    },
    {
      id: "copy_number",
      prompt: "Remember this number: 7391. What number should you remember? Reply with only the number.",
      check: contains(/7391/),
    },
    {
      id: "berlin",
      prompt: "What is the capital of Germany?",
      check: contains(/\bberlin\b/i),
    },
    {
      id: "html_tag",
      prompt: "Write the HTML tag used for the largest heading. Reply with the tag only.",
      check: contains(/h1/i),
    },
    {
      id: "even_number",
      prompt: "Is 8 an even number? Answer yes or no.",
      check: (text) => {
        const t = text.trim();
        return { pass: /^yes\b/i.test(t), detail: t.slice(0, 40) };
      },
    },
    {
      id: "sustained_speed",
      name: "Sustained Speed (256 tokens)",
      prompt:
        "Write a detailed essay explaining how small language models running on edge devices enable autonomous spacecraft and rovers to make real-time decisions without waiting for ground control communication. Discuss latency, reliability, local reasoning, and sensor triage across multiple paragraphs.",
      maxNewTokens: 256,
      opts: { ignoreEos: true },
      check: (text, info) => {
        const n = info?.generatedIds?.length || 0;
        const tps = info?.tokPerS || 0;
        return {
          pass: n >= 120,
          score: Math.min(n / 256, 1),
          detail: `sustained: ${tps.toFixed(1)} tok/s (${n} tok in ${Math.round(info?.totalMs || 0)}ms)`,
        };
      },
    },
  ],
};

export function estimateSuiteSeconds({ nModels, nTests, tokPerS, avgNew = 18, avgPrompt = 36 }) {
  const tps = Math.max(8, tokPerS || 80);
  const tokens = nModels * nTests * (avgNew + avgPrompt);
  return (tokens / tps) * 1.25;
}

export function fourGramRepeat(text) {
  const toks = text.split(/\s+/).filter(Boolean);
  const grams = [];
  for (let i = 0; i < toks.length - 3; i++) grams.push(toks.slice(i, i + 4).join(" "));
  if (!grams.length) return 0;
  return 1 - new Set(grams).size / grams.length;
}

export const EXAMPLE_CUSTOM = `{
  id: "copy-from-context",
  name: "Copy from context",
  description: "The answer is in the prompt — a fair test for a 125M model.",
  maxNewTokens: 24,
  tests: [
    {
      id: "planet",
      prompt: "The planet named in this sentence is Neptune. Which planet was named? Reply with one word.",
      check(text) {
        return { pass: /neptune/i.test(text), detail: text.slice(0, 80) };
      }
    },
    {
      id: "number",
      prompt: "Remember this number: 7391. What number should you remember? Reply with only the number.",
      check(text) {
        return { pass: /7391/.test(text), detail: text.slice(0, 80) };
      }
    }
  ]
}`;

export const EXAMPLE_CUSTOM_FN = `function benchmark() {
  const expected = ["Paris", "Berlin", "Tokyo"];
  const tests = [
    ["France", "Paris"],
    ["Germany", "Berlin"],
    ["Japan", "Tokyo"]
  ].map(([country, city]) => ({
    id: city.toLowerCase(),
    prompt: "What is the capital of " + country + "?",
    check(text) {
      const re = new RegExp("\\\\b" + city + "\\\\b", "i");
      return { pass: re.test(text), detail: text.slice(0, 60) };
    }
  }));
  return {
    id: "capitals-mini",
    name: "Capitals (mini)",
    maxNewTokens: 32,
    tests
  };
}`;

export const LLM_PROMPT = `You write PetitGPT custom benchmarks. Output ONLY JavaScript that evaluates to a benchmark object (or a function that returns one). Do not markdown-fence unless asked.

The host will eval() the code, then run each test.prompt through a 124.6M greedy model and call test.check(text, info).

Contract:
{
  id: string,
  name: string,
  description?: string,
  maxNewTokens?: number,          // default 48, max 256
  tests: [{
    id: string,
    prompt: string,               // user-turn only; the host wraps the chat template
    check: (text, info) => ({ pass: boolean, score?: number, detail?: string })
  }]
}

info = { generatedIds: number[], stopReason: "eos"|"max_new_tokens", backend: string, dtype: string }

Rules:
- Checks must be objective (regex, counts, exact tokens). No "does this sound good".
- A 125M model is weak at facts and arithmetic. Prefer answers copied from the prompt, format following, short classifications, or well-known facts (Paris, 1+1).
- Keep prompts short. ASCII is safest.
- check() must be a real function, not a string.
- Do not fetch(), import scripts, or touch cookies/localStorage.

Example:
({
  id: "ctx-copy",
  name: "Context copy",
  maxNewTokens: 16,
  tests: [{
    id: "code",
    prompt: "The code is ZEBRA-9. Reply with only the code.",
    check: (text) => ({ pass: /ZEBRA-9/.test(text), detail: text.slice(0, 40) })
  }]
})`;

export function validateSuite(s) {
  if (!s || typeof s !== "object") throw new Error("benchmark must be an object");
  if (typeof s.id !== "string" || !s.id) throw new Error("missing id");
  if (typeof s.name !== "string" || !s.name) throw new Error("missing name");
  if (!Array.isArray(s.tests) || !s.tests.length) throw new Error("tests[] required");
  if (s.tests.length > 64) throw new Error("max 64 tests");
  for (const t of s.tests) {
    if (!t || typeof t.prompt !== "string" || !t.prompt.trim()) throw new Error("each test needs prompt");
    if (typeof t.check !== "function") throw new Error(`test ${t.id || t.prompt} check() must be a function`);
    if (t.prompt.length > 4000) throw new Error("prompt too long");
  }
  const cap = Number(s.maxNewTokens || 48);
  if (!Number.isFinite(cap) || cap < 1 || cap > 256) throw new Error("maxNewTokens 1..256");
  return true;
}

export function compileCustom(source) {
  const src = String(source || "").trim();
  if (!src) throw new Error("empty program");
  // eslint-disable-next-line no-eval
  let value = eval(src);
  if (typeof value === "function") value = value();
  validateSuite(value);
  return value;
}

export const SUSTAINED_SUITE = {
  id: "sustained-v1",
  name: "Sustained Throughput Benchmark",
  description: "Continuous 256-token sustained generation test.",
  maxNewTokens: 256,
  tests: [
    {
      id: "sustained_speed",
      name: "Sustained Speed (256 tokens)",
      prompt:
        "Write a detailed essay explaining how small language models running on edge devices enable autonomous spacecraft and rovers to make real-time decisions without waiting for ground control communication. Discuss latency, reliability, local reasoning, and sensor triage across multiple paragraphs.",
      maxNewTokens: 256,
      opts: { ignoreEos: true },
      check: (text, info) => {
        const n = info?.generatedIds?.length || 0;
        const tps = info?.tokPerS || 0;
        return {
          pass: n >= 120,
          score: Math.min(n / 256, 1),
          detail: `sustained: ${tps.toFixed(1)} tok/s (${n} tok in ${Math.round(info?.totalMs || 0)}ms)`,
        };
      },
    },
  ],
};
