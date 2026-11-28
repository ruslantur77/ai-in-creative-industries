# ЛР 14. Перенос художественного стиля (метод Gatys и др., arXiv:1508.06576). Библиотека учебных функций.
# Признаки — VGG19 из torchvision (веса IMAGENET1K_V1 загружаются с download.pytorch.org при первом вызове, ≈ 550 МБ).
# Оптимизируются пиксели изображения, сеть не обучается. Внешние данные: изображения содержания (из ЛР 5) и стиля (Wikimedia Commons, общественное достояние).
import hashlib, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights

torch.set_num_threads(int(os.environ.get("LAB_THREADS", "4")))
DEVICE = "cpu"
STYLE_SCALE = float(os.environ.get("STYLE_SCALE", "1e4"))

# Номера слоёв в vgg19().features: выходы ReLU после соответствующих свёрток.
LAYERS = {"relu1_1": 1, "relu2_1": 6, "relu2_2": 8, "relu3_1": 11, "relu4_1": 20, "relu4_2": 22, "relu5_1": 29}
STYLE_ALL = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Базовая конфигурация (режим A). Ключи описаны в stylize.
BASE = dict(size=192, content_layer="relu4_2", style_layers=STYLE_ALL, ratio=1e-2, init="content",
            steps=40, color="full", seed=101)

_VGG = None


def vgg():
    """Фиксированная сеть признаков: слои VGG19 до relu5_1, режим оценки, градиенты по весам отключены."""
    global _VGG
    if _VGG is None:
        net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _VGG = net
    return _VGG


def load(path, size):
    """Изображение -> тензор [1, 3, H, W] в [0, 1]; меньшая сторона приводится к size, затем центральный квадрат size × size."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()


def save(x, path):
    a = (x.detach().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(a).save(path)
    return hashlib.sha256(a.tobytes()).hexdigest()


def features(x, names, norm=True):
    """Карты признаков нужных слоёв за один проход. norm=False — без нормировки ImageNet (учебная ошибка демонстрации)."""
    want = {LAYERS[n]: n for n in names}
    out, h = {}, ((x - MEAN) / STD if norm else x)
    for i, layer in enumerate(vgg()):
        h = layer(h)
        if i in want:
            out[want[i]] = h
        if i >= max(want):
            break
    return out


def gram(f):
    """Матрица Грама, делённая на N·M (N — каналы, M — позиции)."""
    _, n, h, w = f.shape
    m = f.reshape(n, h * w)
    return m @ m.t() / (n * h * w)


def rgb_to_yiq(x):
    k = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    return torch.einsum("ij,bjhw->bihw", k, x)


def yiq_to_rgb(x):
    k = torch.linalg.inv(torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]))
    return torch.einsum("ij,bjhw->bihw", k, x)


def match_luminance(style, content):
    """Сдвиг и масштаб яркости стиля под среднее и разброс яркости содержания (для режима color = 'luminance')."""
    ys, yc = rgb_to_yiq(style)[:, :1], rgb_to_yiq(content)[:, :1]
    y = (ys - ys.mean()) / (ys.std() + 1e-8) * yc.std() + yc.mean()
    return y.repeat(1, 3, 1, 1).clamp(0, 1)


def stylize(content, style, cfg, log_every=0):
    """Перенос стиля. content, style — тензоры [1,3,S,S]. cfg: size (уже учтён при загрузке), content_layer, style_layers,
    ratio = α/β, init ('content' | 'noise'), steps (итерации L-BFGS), color ('full' | 'luminance'), seed.
    Возвращает (изображение, журнал потерь)."""
    torch.manual_seed(cfg["seed"])
    if cfg["color"] == "luminance":                       # перенос только яркости: стиль и содержание в оттенках серого
        c_in, s_in = rgb_to_yiq(content)[:, :1].repeat(1, 3, 1, 1), match_luminance(style, content)
    else:
        c_in, s_in = content, style
    names = tuple(cfg["style_layers"]) + (cfg["content_layer"],)
    norm = cfg.get("norm", True)
    with torch.no_grad():
        P = features(c_in, (cfg["content_layer"],), norm)[cfg["content_layer"]]
        A = {n: gram(f) for n, f in features(s_in, cfg["style_layers"], norm).items()}
    x = (c_in.clone() if cfg["init"] == "content" else torch.rand_like(c_in)).requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=cfg["steps"], history_size=50, line_search_fn="strong_wolfe")
    alpha, beta = cfg["ratio"], 1.0
    w = 1.0 / len(cfg["style_layers"])
    hist = []

    def closure():
        opt.zero_grad()
        fs = features(x, names, norm)
        lc = 0.5 * ((fs[cfg["content_layer"]] - P) ** 2).sum()
        ls = sum(w * 0.25 * ((gram(fs[n]) - A[n]) ** 2).sum() for n in cfg["style_layers"])
        loss = alpha * lc + beta * ls * STYLE_SCALE
        loss.backward()
        hist.append((lc.item(), ls.item() * STYLE_SCALE))
        return loss

    opt.step(closure)
    out = x.detach().clamp(0, 1)
    if cfg["color"] == "luminance":                       # яркость — из результата, цветность — из содержания
        yiq = rgb_to_yiq(content)
        yiq[:, :1] = rgb_to_yiq(out)[:, :1]
        out = yiq_to_rgb(yiq).clamp(0, 1)
    return out, hist


def _ssim_gray(a, b):
    """SSIM яркостей (окно Гаусса 11, σ = 1,5), значения в [0, 1]."""
    ya, yb = rgb_to_yiq(a)[:, :1], rgb_to_yiq(b)[:, :1]
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).view(1, 1, 11, 11)
    f = lambda t: F.conv2d(t, k)
    ma, mb = f(ya), f(yb)
    va, vb, cov = f(ya * ya) - ma ** 2, f(yb * yb) - mb ** 2, f(ya * yb) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def metrics(out, content, style):
    """Четыре показателя результата (все считаются одинаково для режимов A и B, независимо от их настроек):
    content_dist — относительное расстояние признаков relu4_2 до содержания (ниже — содержание сохранено лучше);
    style_dist — среднее по пяти слоям относительное расстояние матриц Грама до стиля (ниже — ближе к стилю);
    ssim — структурное сходство яркости с содержанием (выше — структура сохранена);
    color_dist — расстояние средних цветов в пространстве YIQ (каналы I, Q) до стиля (ниже — палитра ближе к стилю)."""
    with torch.no_grad():
        fo = features(out, STYLE_ALL + ("relu4_2",))
        fc = features(content, ("relu4_2",))["relu4_2"]
        fs = features(style, STYLE_ALL)
        cd = float((fo["relu4_2"] - fc).norm() / fc.norm())
        sd = float(np.mean([float((gram(fo[n]) - gram(fs[n])).norm() / gram(fs[n]).norm()) for n in STYLE_ALL]))
        iq_o, iq_s = rgb_to_yiq(out)[:, 1:].mean((2, 3)), rgb_to_yiq(style)[:, 1:].mean((2, 3))
        col = float((iq_o - iq_s).norm())
    return dict(content_dist=round(cd, 4), style_dist=round(sd, 4), ssim=round(_ssim_gray(out, content), 4), color_dist=round(col, 4))


def run(content_path, style_path, cfg, out_path=None):
    """Один перенос: загрузка, стилизация, метрики, время, SHA-256 результата."""
    c, s = load(content_path, cfg["size"]), load(style_path, cfg["size"])
    t0 = time.time()
    out, hist = stylize(c, s, cfg)
    sec = time.time() - t0
    m = metrics(out, c, s)
    m["seconds"] = round(sec, 1)
    m["evals"] = len(hist)
    if out_path:
        m["sha256"] = save(out, out_path)[:16]
    return m, hist


# Пять контекстов: изображение содержания (сгенерировано SD Turbo по брифу ЛР 5) и картина-стиль (Wikimedia Commons, общественное достояние).
CONTEXTS = [
    ("content_c1.png", "style_hokusai.jpg", "афиша фестиваля робототехники × Хокусай «Большая волна в Канагаве» (ок. 1830–1832)"),
    ("content_c2.png", "style_kandinsky.jpg", "обложка подкаста о городской среде × Кандинский «Композиция VII» (1913)"),
    ("content_c3.png", "style_vangogh.jpg", "постер научной выставки × Ван Гог «Звёздная ночь» (1889)"),
    ("content_c4.png", "style_monet.jpg", "визуал цикла лекций по экологии × Моне «Впечатление. Восходящее солнце» (1872)"),
    ("content_c5.png", "style_seurat.jpg", "обложка электронного релиза × Сёра «Воскресный день на острове Гранд-Жатт» (1884–1886)"),
]
# Шесть факторов: режим B = базовый режим A с одним изменением.
FACTORS = ["ratio_1e-2_to_1e-3", "style_low_only", "content_relu2_2", "init_noise", "steps_40_to_80", "color_luminance"]
NOISE_SEEDS = (101, 102, 103)


def factor_cfg(k):
    b = dict(BASE)
    f = FACTORS[k]
    if f == "ratio_1e-2_to_1e-3":
        b["ratio"] = 1e-3
    elif f == "style_low_only":
        b["style_layers"] = ("relu1_1", "relu2_1")
    elif f == "content_relu2_2":
        b["content_layer"] = "relu2_2"
    elif f == "init_noise":
        b["init"] = "noise"
    elif f == "steps_40_to_80":
        b["steps"] = 80
    elif f == "color_luminance":
        b["color"] = "luminance"
    return b


def variant_settings(n):
    """Вариант n = 1..30: контекст c = (n-1)//6, фактор k = (n-1)%6. Возвращает (файл содержания, файл стиля, A, B)."""
    c, k = (n - 1) // 6, (n - 1) % 6
    return CONTEXTS[c][0], CONTEXTS[c][1], dict(BASE), factor_cfg(k)
