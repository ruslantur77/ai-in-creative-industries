import numpy as np
import matplotlib.pyplot as plt
import lab13_lib as L

kind = "two_discs"
f = L.load_feats(kind)

# Берем реальные (train, bilinear) и сгенерированные base_3000 (bilinear)
real = f["real_train_bilinear"].astype(np.float64)
fake = f["base_3000_bilinear"].astype(np.float64)

Ns = [125, 250, 500, 1000, 2000]
means, stds = [], []

rng = np.random.default_rng(101)
for n in Ns:
    vals = []
    for _ in range(5):
        idx_r = rng.choice(len(real), n, replace=False)
        idx_f = rng.choice(len(fake), n, replace=False)
        vals.append(L.frechet(real[idx_r], fake[idx_f]))
    means.append(np.mean(vals))
    stds.append(np.std(vals, ddof=1))
    print(f"N={n}: fid={means[-1]:.2f} (s={stds[-1]:.2f})")

plt.figure(figsize=(8, 5))
plt.errorbar(Ns, means, yerr=stds, fmt='o-', capsize=5, color='blue', label='base_3000')
plt.xscale('log')
plt.xlabel('Число изображений N (лог. шкала)')
plt.ylabel('FID')
plt.title('Зависимость FID от числа изображений (набор «два диска»)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("fid_vs_n.png", dpi=150)
print("График сохранен как fid_vs_n.png")