# Сводка ЛР 14: python summarize14.py  ->  summary14.txt
# Строка варианта: n|контекст|фактор|метрика:A/B/Δ … ; для фактора init_noise B — среднее по трём seed и выборочное s.
import json, statistics as S
import lab14_lib as L
from run14 import key

MET = ("content_dist", "style_dist", "ssim", "color_dist", "seconds")
st = {}
for c in range(5):
    try:
        for line in open(f"store14_c{c}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line); st[r["key"]] = r["m"]
    except FileNotFoundError:
        pass
out = []
for n in range(1, 31):
    cp, sp, a, b = L.variant_settings(n)
    ma = st.get(key(cp, sp, a))
    seeds = L.NOISE_SEEDS if b["init"] == "noise" else (b["seed"],)
    mbs = [st.get(key(cp, sp, dict(b, seed=s))) for s in seeds]
    if ma is None or None in mbs:
        out.append(f"{n}|нет данных"); continue
    parts = []
    for m in MET:
        vb = [x[m] for x in mbs]
        B = S.mean(vb)
        sb = f"±{S.stdev(vb):.4f}" if len(vb) > 1 else ""
        parts.append(f"{m}:{ma[m]}/{B:.4f}{sb}/{B - ma[m]:+.4f}")
    out.append(f"{n}|{(n - 1) // 6 + 1}|{L.FACTORS[(n - 1) % 6]}|" + " ".join(parts))
open("summary14.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
