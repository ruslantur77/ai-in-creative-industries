# Демонстрационный вариант ЛР 14: python demo14.py  -> demo14.json, demo14_sheet.png
# Вариант 1: пара 1, фактор 1 (α/β 1e-2 -> 1e-3); повтор режима A (SHA-256); намеренная ошибка — признаки без нормировки ImageNet.
import json, os, time
import numpy as np
import torch
from PIL import Image, ImageDraw
import lab14_lib as L

os.makedirs("out14", exist_ok=True)
cp, sp, A, B = L.variant_settings(1)
res = {}
for tag, cfg in (("A", A), ("B", B), ("A_repeat", A), ("A_nonorm", dict(A, norm=False))):
    m, h = L.run(cp, sp, cfg, f"out14/demo_{tag}.png")
    m["loss_first"], m["loss_last"] = [round(v, 1) for v in h[0]], [round(v, 1) for v in h[-1]]
    res[tag] = m
    print(tag, m, flush=True)
res["repeat_equal"] = res["A"]["sha256"] == res["A_repeat"]["sha256"]
tiles = [L.load(cp, 192), *(L.load(f"out14/demo_{t}.png", 192) for t in ("A", "B", "A_nonorm")), L.load(sp, 192)]
names = ["content", "A: ratio 1e-2", "B: ratio 1e-3", "error: no ImageNet norm", "style"]   # латиница: в шрифте PIL по умолчанию нет кириллицы
sheet = Image.new("RGB", (192 * 5, 214), "white")
d = ImageDraw.Draw(sheet)
for i, (t, n) in enumerate(zip(tiles, names)):
    sheet.paste(Image.fromarray((t[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)), (192 * i, 22))
    d.text((192 * i + 4, 4), n, fill="black")
sheet.save("demo14_sheet.png")
res["_env"] = dict(torch=torch.__version__, threads=torch.get_num_threads(), date=time.strftime("%Y-%m-%d %H:%M"))
json.dump(res, open("demo14.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
