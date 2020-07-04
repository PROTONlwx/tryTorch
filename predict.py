
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

def load_checkpoint(filepath):
    checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    width, height = image.size
    ratio = width / height
    if width < height:
        width = 256
        height = int(256 / ratio)
    else:
        height = 256
        width = int(256 * ratio)
    image = image.resize((width, height))
    image = image.crop((width / 2 - 112, height / 2 - 112, width / 2 + 112, height / 2 + 112))
    np_image = np.array(image)
    np_image = np_image.astype(float)
    np_image = np_image / 255
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return np.transpose(np_image, (2, 0, 1))


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5):
    model.eval()
    with torch.no_grad():
        img = Image.open(image_path)
        img = process_image(img)
        img = torch.FloatTensor(img)
        logps = model(img.unsqueeze_(0))
        ps = torch.exp(logps)
        return ps.topk(topk, dim=1)


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

def idx_to_name(idx):
    class_to_idx = testmodel.class_to_idx
    flower_class = [get_key(class_to_idx, x) for x in idx.numpy()[0]]
    cat_to_name = json.load(open('./cat_to_name.json', 'r'))
    flower_names = [cat_to_name.get(x) for x in flower_class]
    return flower_names

testmodel = load_checkpoint("./checkpointtest.pth")
img_path = "./flowers/test/28/image_05230.jpg"
prob, idx = predict(img_path, testmodel)
names = idx_to_name(idx)
prob = prob.numpy()[0]
result = dict(zip(names, prob))
print(result)