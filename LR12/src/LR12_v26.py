# ЛР 12, вариант 26: режим 5 (малый набор, 3 шага D, lr_d=1e-3), приём 2 (instance_noise).
import time
import json
import numpy as np
import torch
from PIL import Image
import lab12_lib as L
import lab11_lib as G

N = 26
kind, a_cfg, b_cfg = L.variant_settings(N)
print("=" * 70, flush=True)
print(f"ЛР 12, вариант {N}: режим 5 + приём 2 (шум на входе D)", flush=True)
print("=" * 70, flush=True)
print(f"Набор: {kind}", flush=True)
print(f"Режим A: n_train={a_cfg['n_train']}, n_critic={a_cfg['n_critic']}, lr_d={a_cfg['lr_d']}, noise={a_cfg['noise']}, steps={a_cfg['steps']}", flush=True)
print(f"Режим B: n_train={b_cfg['n_train']}, n_critic={b_cfg['n_critic']}, lr_d={b_cfg['lr_d']}, noise={b_cfg['noise']}, steps={b_cfg['steps']}", flush=True)
print("", flush=True)

# 1. Запуск variant_summary (10 обучений)
t0 = time.time()
print("Запуск variant_summary(26): 5 seed × 2 режима = 10 обучений...", flush=True)
print("Ожидаемое время: 15–25 минут.", flush=True)
res = L.variant_summary(N)
print(f"\nВсе обучения завершены за {time.time() - t0:.1f} с", flush=True)

print("\n=== Таблица метрик (вариант 26) ===", flush=True)
print(f"{'метрика':12s} {'A':>8s} {'s':>7s} {'B':>8s} {'d':>8s} {'|d|/s':>7s} {'2s':>4s}", flush=True)
for name, a, s, b, d, q, sig in res["rows"]:
    nd = 4 if name in ("diversity", "seconds") else (3 if name == "coverage" else 2)
    print(f"{name:12s} {a:8.{nd}f} {s:7.{nd}f} {b:8.{nd}f} {d:+8.{nd}f} {q:7.2f} {'да' if sig else 'нет':>4s}", flush=True)

with open("res12_v26.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

# 2. Обучение A и B на seed 101 со снимками и историей оценок D
print("\nЗапуск обучения режимов A и B на seed 101 со снимками (шаги 100, 300, 700, 1000)...", flush=True)
Xtr = G.L.dataset(kind)[0]
SEED = 101
SNAPS = (100, 300, 700, 1000)

out_hist = {}
for name, cfg in (("A", a_cfg), ("B", b_cfg)):
    snaps = {}
    def cb(step, gen, name=name, snaps=snaps, cfg=cfg):
        snaps[step] = L.metrics(gen, kind, SEED)
        x = L.sample(gen, 16, 7).permute(0, 2, 3, 1).numpy()
        row = np.concatenate([np.concatenate(list(x[r * 8:(r + 1) * 8]), axis=1) for r in range(2)], axis=0)
        Image.fromarray((row * 255).astype("uint8")).resize((512, 128), Image.NEAREST).save(f"LR12_v26_{name}_{step}.png")

    print(f"Обучение режима {name}...", flush=True)
    # Передаем n_train из конфига
    Xtr_sub = Xtr[:cfg["n_train"]]
    gen, hist = L.train_gan(Xtr_sub, cfg, SEED, snapshots=SNAPS, on_snapshot=cb)
    out_hist[name] = dict(cfg=cfg, snaps={str(k): v for k, v in snaps.items()}, hist=hist)
    print(f"Режим {name} готов.", flush=True)

with open("hist12_v26.json", "w", encoding="utf-8") as f:
    json.dump(out_hist, f, ensure_ascii=False, indent=1)

# Вывод динамики по шагам
print("\n=== Динамика обучения (seed 101) ===", flush=True)
print(f"{'Режим':6s} {'Шаг':>5s} {'nn_psnr':>8s} {'coverage':>8s} {'diversity':>10s} {'D_real':>7s} {'D_fake':>7s} {'потеря G':>9s} {'потеря D':>9s}", flush=True)
for name, data in out_hist.items():
    h, s = data["hist"], data["snaps"]
    for st in SNAPS:
        i1, i0 = st // 10, max(0, st // 10 - 10)
        w = lambda k: sum(h[k][i0:i1]) / max(1, len(h[k][i0:i1]))
        m = s[st]
        print(f"{name:6s} {st:5d} {m['nn_psnr']:8.2f} {m['coverage']:8.3f} {m['diversity']:10.3f} {w('d_real'):7.3f} {w('d_fake'):7.3f} {w('lg'):9.3f} {w('ld'):9.3f}", flush=True)

print("\n" + "=" * 70, flush=True)
print("ГОТОВО. Скачай из папки LR12:", flush=True)
print("- res12_v26.json (таблица метрик)", flush=True)
print("- hist12_v26.json (история потерь и оценок D)", flush=True)
print("- LR12_v26_A_*.png, LR12_v26_B_*.png (снимки)", flush=True)
print("=" * 70, flush=True)