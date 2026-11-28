# Демонстрации лекции 8: python lec08_demo.py <часть>   (части: ratio, size, clip, order; результат — lec08_<часть>.json)
# Требуются lab14_lib.py и файлы content_c*.png, style_*.jpg (prep14.py). Для части clip — transformers с CLIP ViT-L/14 (как в ЛР 9).
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import lab14_lib as L

part = sys.argv[1]
res = {}


def unsharp(x, amount=1.0, sigma=1.0):
    """Нерезкое маскирование: x + amount·(x − размытие Гаусса), как в ЛР 4."""
    r = int(3 * sigma) * 2 + 1
    g = torch.exp(-((torch.arange(r) - r // 2) ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] @ g[None, :]).view(1, 1, r, r).repeat(3, 1, 1, 1)
    blur = F.conv2d(F.pad(x, (r // 2,) * 4, mode="reflect"), k, groups=3)
    return (x + amount * (x - blur)).clamp(0, 1)


def edge(x):
    """Средняя длина градиента яркости (оператор Собеля, нормировка 1/8) — «резкость»."""
    y = L.rgb_to_yiq(x)[:, :1]
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(y, kx), F.conv2d(y, kx.transpose(2, 3))
    return float(torch.sqrt(gx ** 2 + gy ** 2).mean() / 8)


if part == "ratio":                     # компромисс α/β на контексте 1 (три значения), 192 px, 40 итераций
    cp, sp = L.CONTEXTS[0][0], L.CONTEXTS[0][1]
    for r in (1e-1, 1e-2, 1e-3):
        cfg = dict(L.BASE, ratio=r)
        m, h = L.run(cp, sp, cfg, f"lec08_ratio_{r:g}.png")
        res[f"{r:g}"] = m
        print(r, m, flush=True)
elif part == "size":                    # время переноса от размера изображения (база, контекст 2)
    cp, sp = L.CONTEXTS[1][0], L.CONTEXTS[1][1]
    for s in (128, 192, 256):
        m, h = L.run(cp, sp, dict(L.BASE, size=s))
        res[str(s)] = dict(seconds=m["seconds"], evals=m["evals"])
        print(s, res[str(s)], flush=True)
elif part == "clip":                    # CLIP-сходство с брифом: исходное изображение, стилизация α/β = 1e-2 и 1e-3
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    from prep14 import BRIEFS
    cid, rev = "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"
    model = CLIPModel.from_pretrained(cid, revision=rev, use_safetensors=True).eval()
    proc = CLIPProcessor.from_pretrained(cid, revision=rev)

    @torch.no_grad()
    def cos(img, text):
        o = model(**proc(text=[text], images=img, return_tensors="pt", padding=True, truncation=True))
        a, b = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True), o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        return float((a @ b.T)[0, 0])

    def pil(x):
        return Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

    for i, (cp, sp, _) in enumerate(L.CONTEXTS):
        c, s = L.load(cp, L.BASE["size"]), L.load(sp, L.BASE["size"])
        row = {"content": cos(pil(c), BRIEFS[i]), "style_image": cos(pil(s), BRIEFS[i])}
        for r in (1e-2, 1e-3):
            out, _ = L.stylize(c, s, dict(L.BASE, ratio=r))
            row[f"stylized_{r:g}"] = cos(pil(out), BRIEFS[i])
        res[f"c{i + 1}"] = {k: round(v, 4) for k, v in row.items()}
        print(i + 1, res[f"c{i + 1}"], flush=True)
elif part == "order":                   # нерезкое маскирование до и после стилизации (контексты 1–5)
    for i, (cp, sp, _) in enumerate(L.CONTEXTS):
        c, s = L.load(cp, L.BASE["size"]), L.load(sp, L.BASE["size"])
        a, _ = L.stylize(unsharp(c), s, L.BASE)              # резкость, затем стилизация
        b0, _ = L.stylize(c, s, L.BASE)
        b = unsharp(b0)                                       # стилизация, затем резкость
        res[f"c{i + 1}"] = {"sharpen_then_style": dict(L.metrics(a, c, s), edge=round(edge(a), 4)),
                            "style_then_sharpen": dict(L.metrics(b, c, s), edge=round(edge(b), 4)),
                            "style_only": dict(L.metrics(b0, c, s), edge=round(edge(b0), 4)),
                            "content_edge": round(edge(c), 4)}
        print(i + 1, res[f"c{i + 1}"], flush=True)
res["_env"] = dict(torch=torch.__version__, threads=torch.get_num_threads(), date=time.strftime("%Y-%m-%d %H:%M"))
json.dump(res, open(f"lec08_{part}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
