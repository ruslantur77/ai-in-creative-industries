# ЛР 14, подготовка данных: python prep14.py
# 1) пять изображений содержания — SD Turbo по брифам ЛР 5 (seed 101, 1 шаг, 512×512), как в ЛР 5/9;
# 2) пять картин-стилей — уменьшенные копии (ширина 960) с Wikimedia Commons; файлы — общественное достояние.
import json, os, sys, urllib.parse, urllib.request
import torch

STYLES = {"style_hokusai.jpg": "Tsunami_by_hokusai_19th_century.jpg",
          "style_kandinsky.jpg": "Vassily_Kandinsky,_1913_-_Composition_7.jpg",
          "style_vangogh.jpg": "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
          "style_monet.jpg": "Monet_-_Impression,_Sunrise.jpg",
          "style_seurat.jpg": "Georges_Seurat_-_A_Sunday_on_La_Grande_Jatte_--_1884_-_Google_Art_Project.jpg"}
# Брифы — те же, что в ЛР 5 (lab05_lib.BRIEFS).
BRIEFS = [
    "editorial poster concept for a robotics festival, glowing mechanical arm silhouette, deep blue and amber palette, clean composition, no text, no logo, no people",
    "square podcast cover about the city environment, abstract layered skyline of geometric shapes, teal and orange palette, minimal flat style, no text, no people",
    "poster concept for a science exhibition, abstract constellation of connected nodes, dark background with white and cyan lines, balanced composition, no text, no logo",
    "visual for an ecology lecture series, stylized forest canopy seen from above, layered green shapes, soft morning light, no text, no people",
    "cover art for an electronic music release, brushed metal surface with iridescent reflections, macro photography style, strong contrast, no text, no logo, no people",
]
UA = {"User-Agent": "course-materials-lr14/1.0 (educational use)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


if __name__ == "__main__":
    for out, title in STYLES.items():
        if os.path.exists(out):
            continue
        q = "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url&iiurlwidth=960&titles=File:" + urllib.parse.quote(title)
        info = list(json.loads(get(q))["query"]["pages"].values())[0]["imageinfo"][0]
        open(out, "wb").write(get(info["thumburl"]))
        print(out, os.path.getsize(out), info["descriptionurl"], flush=True)
    from diffusers import AutoPipelineForText2Image
    torch.set_num_threads(int(os.environ.get("LAB_THREADS", "4")))
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", revision="b261bac6fd2cf515557d5d0707481eafa0485ec2", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    for i, prompt in enumerate(BRIEFS, 1):
        out = f"content_c{i}.png"
        if os.path.exists(out):
            continue
        g = torch.Generator("cpu").manual_seed(101)
        pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=512, width=512, generator=g).images[0].save(out)
        print(out, flush=True)
