import streamlit as st
import torch 
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

model = torchvision.models.vgg16(pretrained=False)
model.classifier=nn.Sequential(
    nn.Linear(25088,64),
    nn.ReLU(),
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,4)
)
model.load_state_dict(torch.load("model.pth",map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

st.title("Brain Tumor Classifier Based On PyTorch")
file = st.file_uploader("upload an image",type=['jpg','jpeg','png'])

classes = ['glioma','meningioma','no tumor','pituitary']

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image,caption='uploaded image',use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output,dim=1)
        pred = np.argmax(probs).item()
        pred_class = classes[pred]

    st.write(f"### predicted {pred_class}")
    st.write('confidence score:')
    for idx, score in enumerate(probs[0]):
        st.write(f"{classes[idx]}:{score.item():.4f}")