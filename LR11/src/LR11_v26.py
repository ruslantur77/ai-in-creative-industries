# ЛР 11, вариант 26: набор 5 (two_discs), фактор 2 (gwidth_1_to_0.5).
import time
import json
import numpy as np
import torch
import lab11_lib as L

N = 26
kind, a_cfg, b_cfg = L.variant_settings(N)
print("=" * 70, flush=True)
print(f"ЛР 11, вариант {N}: набор {kind} (два диска), фактор {L.FACTORS[(N-1)%6]}", flush=True)
print("=" * 70, flush=True)
print(f"Режим A: z={a_cfg['z']}, gw={a_cfg['gw']}, lr={a_cfg['lr_g']}, beta1={a_cfg['beta1']}, bn={a_cfg['bn']}, loss={a_cfg['loss']}", flush=True)
print(f"Режим B: z={b_cfg['z']}, gw={b_cfg['gw']}, lr={b_cfg['lr_g']}, beta1={b_cfg['beta1']}, bn={b_cfg['bn']}, loss={b_cfg['loss']}", flush=True)
print(f"Шагов: {a_cfg['steps']}, пакет: {a_cfg['batch']}", flush=True)
print("", flush=True)

# 1. Запуск variant_summary (10 обучений)
t0 = time.time()
print("Запуск variant_summary(26): 5 seed × 2 режима = 10 обучений...", flush=True)
print("Ожидаемое время: 10–15 минут.", flush=True)
res = L.variant_summary(N)
print(f"\nВсе обучения завершены за {time.time() - t0:.1f} с", flush=True)

print("\n=== Таблица метрик (вариант 26) ===", flush=True)
print(f"{'метрика':12s} {'A':>8s} {'s':>7s} {'B':>8s} {'d':>8s} {'|d|/s':>7s} {'2s':>4s}", flush=True)
for name, a, s, b, d, q, sig in res["rows"]:
    nd = 4 if name in ("diversity", "seconds") else (3 if name == "coverage" else 2)
    print(f"{name:12s} {a:8.{nd}f} {s:7.{nd}f} {b:8.{nd}f} {d:+8.{nd}f} {q:7.2f} {'да' if sig else 'нет':>4s}", flush=True)

# Сохраняем результаты
with open("res11_v26.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

# 2. Обучение режима A на seed 101 со снимками
print("\nЗапуск обучения режима A на seed 101 со снимками (шаги 100, 300, 700, 1500)...", flush=True)
from PIL import Image
Xtr = L.L.dataset(kind)[0]
SEED = 101
SNAPS = (100, 300, 700, 1500)

snaps_A = {}
def cb_A(step, gen):
    snaps_A[step] = L.metrics(gen, kind, SEED)
    x = L.sample(gen, 16, 7).permute(0, 2, 3, 1).numpy()
    row = np.concatenate([np.concatenate(list(x[r * 8:(r + 1) * 8]), axis=1) for r in range(2)], axis=0)
    Image.fromarray((row * 255).astype("uint8")).resize((512, 128), Image.NEAREST).save(f"LR11_v26_A_{step}.png")

gen_A, hist_A = L.train_gan(Xtr, a_cfg, SEED, snapshots=SNAPS, on_snapshot=cb_A)
print("Режим A готов.", flush=True)

# 3. Обучение режима B на seed 101 со снимками
print("\nЗапуск обучения режима B на seed 101 со снимками...", flush=True)
snaps_B = {}
def cb_B(step, gen):
    snaps_B[step] = L.metrics(gen, kind, SEED)
    x = L.sample(gen, 16, 7).permute(0, 2, 3, 1).numpy()
    row = np.concatenate([np.concatenate(list(x[r * 8:(r + 1) * 8]), axis=1) for r in range(2)], axis=0)
    Image.fromarray((row * 255).astype("uint8")).resize((512, 128), Image.NEAREST).save(f"LR11_v26_B_{step}.png")

gen_B, hist_B = L.train_gan(Xtr, b_cfg, SEED, snapshots=SNAPS, on_snapshot=cb_B)
print("Режим B готов.", flush=True)

# 4. Сохраняем историю потерь и снимки метрик
out_hist = {}
for name, h, s in (("A", hist_A, snaps_A), ("B", hist_B, snaps_B)):
    out_hist[name] = dict(hist=h, snaps={str(k): v for k, v in s.items()})
with open("hist11_v26.json", "w", encoding="utf-8") as f:
    json.dump(out_hist, f, ensure_ascii=False, indent=1)

# Вывод динамики по шагам
print("\n=== Динамика обучения (seed 101) ===", flush=True)
print(f"{'Режим':6s} {'Шаг':>5s} {'nn_psnr':>8s} {'coverage':>8s} {'diversity':>10s} {'потеря G':>9s} {'потеря D':>9s}", flush=True)
for name, h, s in (("A", hist_A, snaps_A), ("B", hist_B, snaps_B)):
    for st in SNAPS:
        i1, i0 = st // 10, max(0, st // 10 - 10)
        w = lambda k: sum(h[k][i0:i1]) / max(1, len(h[k][i0:i1]))
        m = s[st]
        print(f"{name:6s} {st:5d} {m['nn_psnr']:8.2f} {m['coverage']:8.3f} {m['diversity']:10.3f} {w('lg'):9.3f} {w('ld'):9.3f}", flush=True)

print("\n" + "=" * 70, flush=True)
print("ГОТОВО. Скачай из папки LR11:", flush=True)
print("- res11_v26.json (таблица метрик)", flush=True)
print("- hist11_v26.json (история потерь)", flush=True)
print("- LR11_v26_A_100.png ... A_1500.png (снимки режима A)", flush=True)
print("- LR11_v26_B_100.png ... B_1500.png (снимки режима B)", flush=True)
print("=" * 70, flush=True)