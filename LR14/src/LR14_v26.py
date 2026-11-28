# ЛР 14, вариант 26: пара 5 (Сёра), фактор 2 (слои стиля: только relu1_1 и relu2_1).
import json
import os
import time
import lab14_lib as L

N = 26
cp, sp, A, B = L.variant_settings(N)
print("=" * 70, flush=True)
print(f"ЛР 14, вариант {N}", flush=True)
print(f"Контент: {cp} | Стиль: {sp}", flush=True)
print(f"Режим A (база): слои {A['style_layers']}", flush=True)
print(f"Режим B (фактор): слои {B['style_layers']}", flush=True)
print("=" * 70, flush=True)

os.makedirs("out14", exist_ok=True)

# 1. Режим A
print("\n[1/3] Запуск режима A (база)...", flush=True)
t0 = time.time()
mA, hA = L.run(cp, sp, A, "out14/v26_A.png")
mA["loss_first"] = [round(v, 1) for v in hA[0]]
mA["loss_last"] = [round(v, 1) for v in hA[-1]]
print(f"Готово за {time.time() - t0:.1f} с. SHA-256: {mA['sha256']}", flush=True)

# 2. Режим B
print("\n[2/3] Запуск режима B (фактор)...", flush=True)
t0 = time.time()
mB, hB = L.run(cp, sp, B, "out14/v26_B.png")
mB["loss_first"] = [round(v, 1) for v in hB[0]]
mB["loss_last"] = [round(v, 1) for v in hB[-1]]
print(f"Готово за {time.time() - t0:.1f} с. SHA-256: {mB['sha256']}", flush=True)

# 3. Повтор режима A (для SHA-256)
print("\n[3/3] Повтор режима A (проверка воспроизводимости)...", flush=True)
t0 = time.time()
mA2, _ = L.run(cp, sp, A, "out14/v26_A_repeat.png")
print(f"Готово за {time.time() - t0:.1f} с. SHA-256: {mA2['sha256']}", flush=True)

# Сводка
print("\n" + "=" * 70, flush=True)
print("=== РЕЗУЛЬТАТЫ (вариант 26) ===", flush=True)
print(f"{'метрика':14s} {'A (база)':>10s} {'B (фактор)':>10s} {'Δ (B-A)':>10s} {'допуск':>8s} {'значимо':>8s}", flush=True)
D = {"content_dist": 0.03, "style_dist": 0.03, "ssim": 0.03, "color_dist": 0.005}
for m in ("content_dist", "style_dist", "ssim", "color_dist", "seconds"):
    a, b = mA[m], mB[m]
    d = b - a
    if m in D:
        tol = D[m]
        sig = "да" if abs(d) > tol else "нет"
        print(f"{m:14s} {a:10.4f} {b:10.4f} {d:+10.4f} {tol:8.3f} {sig:>8s}", flush=True)
    else:
        print(f"{m:14s} {a:10.1f} {b:10.1f} {d:+10.1f} {'—':>8s} {'—':>8s}", flush=True)

print(f"\nВоспроизводимость: SHA-256 A = {mA['sha256']}, повтор = {mA2['sha256']} -> {'СОВПАЛО' if mA['sha256'] == mA2['sha256'] else 'НЕ СОВПАЛО'}", flush=True)

print(f"\nПотери (содержание; стиль×10000):", flush=True)
print(f"A: от {mA['loss_first']} до {mA['loss_last']}", flush=True)
print(f"B: от {mB['loss_first']} до {mB['loss_last']}", flush=True)

res = dict(variant=N, ctx=5, factor="style_low_only", A=mA, B=mB, repeat_sha256=mA2["sha256"])
with open("res14_v26.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\nСохранено в res14_v26.json", flush=True)