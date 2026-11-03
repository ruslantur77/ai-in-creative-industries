# ЛР 9, вариант 26: контекст 5 (electronic music cover art), элемент 2 (neon rim light).
import time
import numpy as np
import torch
import lab09_lib as L

print("=" * 70, flush=True)
print("ЛР 9, часть 1, вариант 26: контекст 5 × элемент 2 (neon rim light)", flush=True)
print("=" * 70, flush=True)

N = 26
c, k = (N - 1) // 6, (N - 1) % 6
print(f"Контекст: {c + 1} ({L.SHORT[c]})", flush=True)
print(f"Элемент: {k + 1} ({L.FACTORS[k]})", flush=True)
print(f"Запрос A: {L.prompt_a(c)}", flush=True)
print(f"Запрос B: {L.prompt_b(c, k)}", flush=True)
print(f"Текст-цель: {L.ELEMENTS[c][k][1]}", flush=True)
print(f"Серия seed: {list(L.SEEDS)}", flush=True)
print("", flush=True)

print("Загрузка модели и CLIP...", flush=True)
t0 = time.time()
pipe = L.load_pipe()
clip = L.load_clip()
print(f"Загружено за {time.time() - t0:.0f} с", flush=True)

print("", flush=True)
print("Запуск variant_summary(26): 2 запроса × 5 seed = 10 изображений", flush=True)
print("Ожидаемое время: 4-5 минут", flush=True)

t0 = time.time()
res = L.variant_summary(pipe, clip, N)
print(f"\nВсе изображения готовы за {time.time() - t0:.0f} с", flush=True)
print("=" * 70, flush=True)

print("\n=== Таблица метрик ===", flush=True)
print(f"{'метрика':12s} {'A':>8s} {'s':>7s} {'B':>8s} {'d':>8s} {'|d|/s':>7s} {'2s':>4s}", flush=True)
for name, a, s, b, d, q, sig in res["rows"]:
    print(f"{name:12s} {a:8.3f} {s:7.3f} {b:8.3f} {d:+8.3f} {q:7.2f} {'да' if sig else 'нет':>4s}", flush=True)

# Сохранить изображения A и B для seed 101
print("\nГенерация изображений A и B (seed 101)...", flush=True)
imgA = L.generate(pipe, L.prompt_a(c), 101)
imgB = L.generate(pipe, L.prompt_b(c, k), 101)
imgA.save("A_seed101.png")
imgB.save("B_seed101.png")
print("A_seed101.png и B_seed101.png сохранены", flush=True)

print("\n" + "=" * 70, flush=True)
print("ГОТОВО. Скачай A_seed101.png и B_seed101.png.", flush=True)
print("=" * 70, flush=True)