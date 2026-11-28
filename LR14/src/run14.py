# ЛР 14, полный прогон: python run14.py <контекст 0..4> [rev]
# Шесть вариантов контекста (6c+1…6c+6). Каждый перенос (файлы, конфигурация, seed) пишется в store14_c<c>.jsonl
# и при повторном запуске пропускается; store перечитывается перед каждым переносом (можно запускать два процесса — с начала и с конца).
import json, os, sys, time
os.environ.setdefault("LAB_THREADS", "1")
import lab14_lib as L

STORE = LOG = None


def key(cpath, spath, cfg):
    return json.dumps([cpath, spath, sorted((k, list(v) if isinstance(v, tuple) else v) for k, v in cfg.items())])


def load_store():
    st = {}
    if os.path.exists(STORE):
        for line in open(STORE, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); st[r["key"]] = r["m"]
    return st


def get(cpath, spath, cfg, tag):
    k = key(cpath, spath, cfg)
    st = load_store()
    if k not in st:
        m, hist = L.run(cpath, spath, cfg, f"out14/{tag}.png")
        m["loss_first"], m["loss_last"] = [round(v, 1) for v in hist[0]], [round(v, 1) for v in hist[-1]]
        open(STORE, "a", encoding="utf-8").write(json.dumps(dict(key=k, tag=tag, m=m), ensure_ascii=False) + "\n")
        open(LOG, "a", encoding="utf-8").write(f"{time.strftime('%H:%M:%S')} {tag} {m['seconds']} с\n")
        st[k] = m
    return st[k]


if __name__ == "__main__":
    c = int(sys.argv[1]); REV = "rev" in sys.argv
    STORE, LOG = f"store14_c{c}.jsonl", f"log14_c{c}.txt"
    os.makedirs("out14", exist_ok=True)
    jobs = []
    for j in range(6):
        n = c * 6 + j + 1
        cp, sp, a, b = L.variant_settings(n)
        jobs.append((cp, sp, a, f"c{c + 1}_A"))
        seeds = L.NOISE_SEEDS if b["init"] == "noise" else (b["seed"],)
        for s in seeds:
            bb = dict(b); bb["seed"] = s
            jobs.append((cp, sp, bb, f"v{n:02d}_B_s{s}"))
    for job in (jobs[::-1] if REV else jobs):
        get(*job)
    if not REV:
        open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S") + " конец\n")
