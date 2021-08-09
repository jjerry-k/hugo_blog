---
title: "쉽고 빠르게 결과를 확인해보자. Streamlit"
date: 2021-08-08T23:05:38+09:00
draft: false

#post thumb
image: 'https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg'

# meta description
description: "this is meta description"
math: true

# taxonomies
categories: 
  - "DeepLearning"
tags:
  - "Tools"

# post type
type: "post"
---

동일한 task에서 Deep learning model 들을 이래저래 학습을 하다보면 매우 귀찮은 상황이 발생하곤 합니다....  
N개의 모델에 대한 각각의 결과를 보기가 **매우 귀...찮...다...!!!!**(물론 wandb, neptune, 등의 log 관리를 하면 괜찮..)  
그래서 이번에는 [Streamlit](https://docs.streamlit.io/en/stable/)이라는 파이썬 라이브러리를 이용해서 Infernce 결과를 정말 쉽게 볼 수 있는 포스팅을 해보려합니다! 

뭐....제 포스팅의 성격 아시죠..? 
>자세한 설명..넘어갑니다. 바로 예제에요... 궁금하면 링크를 드릴테니 알아서 보세요.

## Example code
``` python
import streamlit as st

import json
from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import pretrainedmodels
from efficientnet_pytorch import EfficientNet



with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

available_models = [
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", 
    "efficientnet-b5", "efficientnet-b6", "efficientnet-b7", 
    "alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext101_32x4d", "resnext101_64x4d", 
    "squeezenet1_0", "squeezenet1_1", "nasnetamobile", "nasnetalarge", 
    "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", 
    "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
    "inceptionv4", "inceptionresnetv2", "xception", "fbresnet152", "bninception",
    "cafferesnet101", "pnasnet5large", "polynet"
]

def load_moel(model_name):
    if "efficientnet" in model_name:
        model = EfficientNet.from_pretrained(model_name)    
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000)
    return model

option = st.selectbox(
    'Select Model',
     available_models)
model = load_moel(option)
model.eval()

# load data
uploaded_file = st.file_uploader("Choose a Image")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    img_for_plot = np.array(image)
    
    img = transforms.ToTensor()(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    img = normalize(img).unsqueeze(dim=0)   
    result = model(img).squeeze(dim=0)
    predict_idx = result.argmax().item()
    prob = torch.softmax(result, dim=0)
    st.image(img_for_plot, use_column_width=True)
    st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")
```
## Result
{{< figure src="/images/post/streamlit/streamlit.gif" >}}

정말 간단...합니다.  
flask, django, fastApi 등 웹 프레임워크를 이용해서도 이런걸 만들 수 있길 합니다만...  
어쨌든 웹..이기 때문에 html, css, javascript 같은 언어 사용이 불가피한데요.  
이건...그냥 편하게 해주네요..!  
더 다양한 페이지를 만들어볼 수 있을 듯합니다!