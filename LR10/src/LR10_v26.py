# ЛР 10, вариант 26: бриф 5 (электронная музыка), пара k=2 (SD Turbo и bk-sdm-tiny).
import time
import json
import lab10_lib as L

N = 26
c, k = (N - 1) // 6, (N - 1) % 6
print("=" * 70, flush=True)
print(f"ЛР 10, вариант {N}: бриф {c + 1} × пара {k + 1} ({L.PAIRS[k][0]} и {L.PAIRS[k][1]})", flush=True)
print("=" * 70, flush=True)
print(f"Бриф: {L.BRIEFS[c]}", flush=True)
print(f"Модель A: {L.MODELS[L.PAIRS[k][0]]['repo']} ({L.MODELS[L.PAIRS[k][0]]['steps']} шагов, guidance {L.MODELS[L.PAIRS[k][0]]['guidance']})", flush=True)
print(f"Модель B: {L.MODELS[L.PAIRS[k][1]]['repo']} ({L.MODELS[L.PAIRS[k][1]]['steps']} шагов, guidance {L.MODELS[L.PAIRS[k][1]]['guidance']})", flush=True)
print("", flush=True)

t0 = time.time()
print("Запуск variant_summary(26, save_dir='v26')...", flush=True)
print("Внимание: модели загружаются по очереди (лимит памяти 16 ГБ). Ожидаемое время: 6-8 минут.", flush=True)
print("", flush=True)

# Запускаем генерацию и метрики
res = L.variant_summary(N, save_dir="v26")

print("\nГенерация завершена за", round(time.time() - t0, 1), "с", flush=True)
print("=" * 70, flush=True)

# Вывод таблицы
print("\n=== Таблица метрик (вариант 26) ===", flush=True)
print(f"{'метрика':12s} {'A':>8s} {'s':>7s} {'B':>8s} {'d':>8s} {'|d|/s':>7s} {'2s':>4s}", flush=True)
for name, a, s, b, d, q, sig in res["rows"]:
    nd = 4 if name in ("edge", "contrast") else 2
    print(f"{name:12s} {a:8.{nd}f} {s:7.{nd}f} {b:8.{nd}f} {d:+8.{nd}f} {q:7.2f} {'да' if sig else 'нет':>4s}", flush=True)

# Генерация слепых листов
print("\nГенерация слепых листов...", flush=True)
key = L.blind_sheets(N, "v26")
print("Слепые листы (blind_101.png ... blind_105.png) и ключ (blind_key.json) сохранены в папке v26.", flush=True)

# Сохраняем результаты в JSON (чтобы потом сделать summary)
with open("res26.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

print("\n" + "=" * 70, flush=True)
print("ГОТОВО. Скачай из папки v26 картинки и листы blind_*.png.", flush=True)
print("=" * 70, flush=True)