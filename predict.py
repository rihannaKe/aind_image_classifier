import argparse
import json
import numpy as np
import torch
from torchvision import transforms
import PIL
from PIL import Image
from train import model_setup


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', action="store", required=True)
    parser.add_argument('--checkpoint', action="store", default="checkpoint.pth")
    parser.add_argument('--device', action="store", default="gpu", dest="device")
    parser.add_argument('--top_k', action="store", default=5)

    args = parser.parse_args()
    return {
        'image': args.image,
        'checkpoint': args.checkpoint,
        'device': args.device,
    }


def loading_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    img_pil = PIL.Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image = img_transforms(img_pil)

    return image


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()

    with torch.no_grad():
        img = process_image(image_path)
        img = img.unsqueeze(0)
        img = img.type(torch.FloatTensor)
        img = img.to(device)

        out = model(img)
        ps = torch.exp(out)

        top_probs, top_labels = torch.topk(ps, topk)

        idx_to_class = {val: key for key, val in
                        model.class_to_idx.items()}

        top_probs = [float(pb) for pb in top_probs[0]]
        labels = [int(lb) for lb in top_labels[0]]
        top_labels = [idx_to_class[lb] for lb in labels]
        top_flowers = [cat_to_name[str(lab)] for lab in labels]

    return top_probs, top_labels, top_flowers

if __name__ == "__main__":
    global predict_args
    predict_args = arg_parser()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if torch.cuda.is_available() and predict_args['device'] == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    trained_model,_,_ = model_setup(device)
    model = loading_checkpoint(trained_model, predict_args['checkpoint'])

    image_tensor = process_image(predict_args['image'])

    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, predict_args.top_k)

    print(f'Probabilities: {top_probs}')
    print(f'Top labels: {top_labels}')
    print(f'Top flowers: {top_flowers}')