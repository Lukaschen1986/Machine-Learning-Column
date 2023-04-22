# -*- coding: utf-8 -*-
"""
这个代码示例使用了OpenAI的CLIP模型，该模型可以同时处理文本和图像，并生成它们之间的相似度分数。
在这个示例中，我们使用CLIP模型来将输入的文本描述编码为特征向量，并将随机生成的图像编码为特征向量，
然后计算它们之间的相似度分数。最后，我们使用类别标签和 Unsplash API来获取与最相关的类别标签相对应的图像，
并将文本描述写入图像，最终保存图像。
"""
import os
import torch as th
from transformers import (CLIPProcessor, CLIPModel)
from PIL import (Image, ImageDraw, ImageFont)


# ----------------------------------------------------------------------------------------------------------------
# 加载CLIP模型和处理器
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
checkpoint = "openai/clip-vit-base-patch32"

pretrained = CLIPModel.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

processor = CLIPProcessor.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

pretrained.to(device)
pretrained.eval()

# ----------------------------------------------------------------------------------------------------------------
# 输入文本描述
text = "a cat sitting on a table"

# ----------------------------------------------------------------------------------------------------------------
# 用 CLIP 处理器编码文本描述
input_ids = processor(text, return_tensors="pt").input_ids.to(device)
text_features = pretrained.get_text_features(input_ids).to(device)
text_features.shape

# ----------------------------------------------------------------------------------------------------------------
# 生成图像
noise = th.randn(1, 3, 224, 224).to(device)
image_features = pretrained.get_image_features(noise).to(device)
image_features.shape

logits_per_image, logits_per_text = pretrained(image_features, text_features)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

# 加载类别标签
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------------------------------------------------------------------------------------------------------
# 找到最相关的图像标签
topk = 5
topk_indices = probs.argsort()[-topk:][::-1]
for i, idx in enumerate(topk_indices):
    print(f"{i + 1}: {classes[idx]} ({probs[idx]:.2f})")

# ----------------------------------------------------------------------------------------------------------------
# 加载图像
image_url = f"https://source.unsplash.com/224x224/?{classes[topk_indices[0]]}"
image = Image.open(requests.get(image_url, stream=True).raw)

# ----------------------------------------------------------------------------------------------------------------
# 显示并保存图像
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 16)
draw.text((0, 0), text, font=font)
image.show()
image.save("output.png")

