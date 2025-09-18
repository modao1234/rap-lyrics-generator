import os, re, json, argparse, time
from tqdm import tqdm
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from geminiapi import Gemini_LLM

# ==== schema ====
schema = Object(
    id="blk", many=False,
    attributes=[
        Text(id="tm",   description="Theme; emotion (connected with a semicolon, e.g., self-motivation; urgency)"),
        Text(id="img",  description="Imagery, separated by commas (e.g., stage, crowd, sweaty palms, clock)"),
        Text(id="Rhyme",description=("Rhyme schemes:rs=AAAA|AABB|ABAB|free ...; Rhyme types: (End rhyme, Multisyllabic rhyme, Slant rhyme, Internal rhyme, Mosaic/compound rhymes ...)")),
    ],
    examples=[
        ("His palms are sweaty, knees weak, arms are heavy\nThere's vomit on his sweater already, mom's spaghetti\nHe's nervous, but on the surface, he looks calm and ready\nTo drop bombs, but he keeps on forgetting\nWhat he wrote down, the whole crowd goes so loud\nHe opens his mouth, but the words won't come out\nHe's chokin', how? Everybody's jokin' now\nThe clock's run out, time's up, over, blaow",
         {"tm":"Stage fright, critical moments, self-doubt; Extreme anxiety, panic, tension, shame, frustration",
          "img":"Physical reactions (sweaty palms/weak knees), vomit, cheap food (mom's pasta), noisy crowds, countdown clocks",
          "Rhyme":"rs=AAAA BBBB; Multisyllabic slant rhyme, Internal rhyme, Mosaic/Compound Rhymes, Chain Rhymes"}),
        ("Fuck you mean?\nYoung Gunna Wunna, they workin' my nerves\nI'm about to pour up some syrup\nFucking this bitch like a perv'\nSmack from the back, grab her perm\nIce, the berg, uh, shittin' on all you lil' turds\nCan't take that dick, wait your turn\nIn my own lane, we can't merge",
         {"tm":"Hedonistic lifestyle, Status declaration; Calm contempt and impatient arrogance",
          "img":"Drug culture, Sexual domination, Exaggeration of wealth",
          "Rhyme":"rs=Monorhyme; Slant Rhyme"})
    ],
)

SECTION_RE       = re.compile(r"^\s*\[([^\]]+)\]\s*$")
ONLY_BRACKETS_RE = re.compile(r"^\s*[\[\]\(\)\-~|/\\]+\.?\s*$")
META_LINE_RE     = re.compile(r"^\s*(produced|written|recorded|mixed|mastered)\s+by\b", re.I)

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def split_sections(raw: str):
    lines = [ln.rstrip() for ln in raw.splitlines()]
    out, name, buf = [], "UNK", []
    for ln in lines:
        m = SECTION_RE.match(ln)
        if m:
            if buf:
                out.append((name, [x for x in buf if x.strip()]))
            name = m.group(1).strip()
            buf = []
        else:
            buf.append(ln)
    if buf:
        out.append((name, [x for x in buf if x.strip()]))
    return out

def is_intro_outro(name: str) -> bool:
    n = name.lower()
    return ("intro" in n) or ("outro" in n) or ("skit" in n)

def normalize_para(name: str):
    n = name.lower()
    if re.search(r"\b(pre[-\s]?chorus|prechorus)\b", n): return "Pre-chorus"
    if re.search(r"\b(chorus|hook|refrain)\b", n):       return "Chorus"
    if re.search(r"\bbridge\b", n):                      return "Bridge"
    if re.search(r"\b(verse|ver\.)\b", n):               return "Verse"
    return None

def section_belongs_to_author(name: str, author: str) -> bool:
    n, a = name.lower(), author.lower()
    cand = ""
    if ":" in n:
        cand = n.split(":", 1)[1]
    else:
        m = re.search(r"\(([^)]+)\)", n)
        if m: cand = m.group(1)
    if cand:
        return a in cand
    return True

def clean_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if SECTION_RE.match(s): continue
        if ONLY_BRACKETS_RE.match(s): continue
        if META_LINE_RE.match(s): continue
        if re.fullmatch(r"\(?x\d+\)?", s, flags=re.I): continue
        if re.fullmatch(r"\((repeat|refrain|ad-?lib|instrumental)\)", s, flags=re.I): continue
        out.append(s)
    return out

def chunk_by_8(lines):
    return [
        [x for x in lines[i:i+8] if x.strip()]
        for i in range(0, len(lines), 8)
        if any(t.strip() for t in lines[i:i+8])
    ]

def extract_author(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" in stem: au = stem.split("_", 1)[0]
    elif "-" in stem: au = stem.split("-", 1)[0]
    else: au = os.path.basename(os.path.dirname(path))
    return au.strip()

def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def build_instruction(tags: dict, author: str, para: str) -> str:
    tm  = (tags.get("tm")    or "").strip()
    img = (tags.get("img")   or "").strip()
    ctl = (tags.get("Rhyme") or "").strip()
    parts = [f"L=8", f"A={author}", f"P={para}"]
    if tm:  parts.append(f"TMI={tm}")
    if img: parts.append(f"IMG={img}")
    if ctl: parts.append(ctl)
    return "; ".join(parts)

RATE_EX_PAT = re.compile(r"retry_delay.*?seconds:\s*(\d+)", re.S|re.I)

def parse_retry_seconds(exc_text: str, default_sec=60) -> int:
    m = RATE_EX_PAT.search(exc_text)
    if m:
        return int(m.group(1)) + 2
    m2 = re.search(r"retry\s*after\s*(\d+)", exc_text, re.I)
    if m2:
        return int(m2.group(1)) + 2
    return default_sec

def safe_batch(chain, reqs, max_concurrency=6, window=12, max_retries=5):
    resps = []
    for i in range(0, len(reqs), window):
        batch = reqs[i:i+window]
        conc  = max_concurrency
        tries = 0
        while True:
            try:
                resps.extend(chain.batch(batch, config={"max_concurrency": conc}))
                break
            except Exception as e:
                txt = str(e)
                if "ResourceExhausted" in e.__class__.__name__ or "429" in txt or "rate" in txt.lower() or "quota" in txt.lower():
                    wait = parse_retry_seconds(txt, default_sec=65)
                    tries += 1
                    if tries > max_retries:
                        # 串行降级
                        for r in batch:
                            s_tries = 0
                            while True:
                                try:
                                    resps.append(chain.invoke(r))
                                    break
                                except Exception as ee:
                                    s_tries += 1
                                    if s_tries > 3:
                                        resps.append(None)
                                        break
                                    time.sleep(parse_retry_seconds(str(ee), 40))
                        break
                    time.sleep(wait)
                    conc = max(1, conc // 2)
                else:
                    tries += 1
                    if tries > max_retries:
                        resps += [None] * len(batch)
                        break
                    time.sleep(2)
    return resps

def process_file(txt_path: str, chain, f_out) -> int:
    author = extract_author(txt_path)
    raw = read_text(txt_path)
    sections = split_sections(raw)

    reqs, metas = [], []
    seen_chorus = set()

    for name, lines in sections:
        if is_intro_outro(name):
            continue
        if not section_belongs_to_author(name, author):
            continue

        para = normalize_para(name)
        if para is None:
            # 不是 Verse/Chorus/Pre-chorus/Bridge，跳过
            continue

        filt_lines = clean_lines(lines)
        if not filt_lines:
            continue

        blocks = chunk_by_8(filt_lines)
        for block in blocks:
            text = "\n".join(block)
            if para == "Chorus":
                key = norm_text(text)
                if key in seen_chorus:
                    continue
                seen_chorus.add(key)
            reqs.append({"text": f"[{para}]\n{text}"})
            metas.append((author, para, text))

    if not reqs:
        return 0

    wrote = 0
    resps = safe_batch(chain, reqs, max_concurrency=6, window=12, max_retries=5)
    for resp, (author, para, out_text) in zip(resps, metas):
        data = (resp or {}).get("validated_data") or (resp or {}).get("data") or {}
        blk  = data.get("blk")
        if isinstance(blk, list):
            blk = blk[0] if blk else {}
        tags = blk or {}
        instr = build_instruction(tags, author, para)
        rec = {"instruction": instr, "input": "", "output": out_text}
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush(); os.fsync(f_out.fileno())  # 实时落盘
        wrote += 1
    return wrote

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing *.txt lyrics")
    ap.add_argument("--out",    required=True, help="Output SFT jsonl path")
    args = ap.parse_args()

    llm = Gemini_LLM()
    chain = create_extraction_chain(llm, schema)

    files = [fn for fn in sorted(os.listdir(args.in_dir)) if fn.lower().endswith(".txt")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("writing to:", os.path.abspath(args.out))

    total = 0
    with open(args.out, "a", encoding="utf-8") as f_out:
        for fn in tqdm(files, desc="files"):
            total += process_file(os.path.join(args.in_dir, fn), chain, f_out)

    print(f"OK: {total} samples -> {args.out}")

if __name__ == "__main__":
    main()
