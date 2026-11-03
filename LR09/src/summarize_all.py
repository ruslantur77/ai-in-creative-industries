# Сводка K1-K4 по всем девяти файлам ответов (контекст 5, три приёма × три повтора).
import json
import lab09_llm as M

CTX = 5
ANSWERS_DIR = "."

approaches = ["zero", "few", "step"]
results = {}

for ap in approaches:
    paths = [f"{ANSWERS_DIR}/ctx{CTX}_{ap}_r{r}.txt" for r in (1, 2, 3)]
    print(f"\n=== Приём: {ap} ===")
    k1s, k2s, k3s = [], [], []
    for p in paths:
        try:
            s = M.score_file(p, CTX)
            print(f"  {p}: parsed={s['parsed']}, n={s['n']}, K1={s['k1']:.3f}, K2={s['k2']:.3f}, K3={s['k3']:.3f}")
            if s['errors']:
                print(f"    errors: {s['errors'][:3]}...")
            k1s.append(s['k1'])
            k2s.append(s['k2'])
            k3s.append(s['k3'])
        except FileNotFoundError:
            print(f"  {p}: НЕ НАЙДЕН")
    k4 = M.stability(paths)
    print(f"  K4 (стабильность) = {k4:.3f}")
    results[ap] = dict(
        K1=sum(k1s) / len(k1s) if k1s else 0,
        K2=sum(k2s) / len(k2s) if k2s else 0,
        K3=sum(k3s) / len(k3s) if k3s else 0,
        K4=k4,
    )

print("\n" + "=" * 60)
print("СВОДКА K1-K4 (среднее по трём повторам)")
print("=" * 60)
print(f"{'приём':10s} {'K1':>8s} {'K2':>8s} {'K3':>8s} {'K4':>8s}")
for ap, r in results.items():
    print(f"{ap:10s} {r['K1']:8.3f} {r['K2']:8.3f} {r['K3']:8.3f} {r['K4']:8.3f}")