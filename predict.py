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
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int, help='Choose top K matches as int.', default=3)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = parser.parse_args()
    return {
        'image': args.image,
        'checkpoint': args.checkpoint,
        'top_k': args.top_k,
        'category_names': args.category_names,
        'powered': args.gpu
    }


def loading_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = PIL.Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image = img_transforms(img_pil)

    return image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    model.to("cpu")

    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)

    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers

if __name__ == "__main__":
    global predict_args
    predict_args = arg_parser()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    trained_model,_,_ = model_setup()
    model = loading_checkpoint(trained_model, predict_args['checkpoint'])

    image_tensor = process_image(predict_args['image'])

    top_probs, top_labels, top_flowers = predict(image_tensor, model)

    print(f'Probabilities: {top_probs}')
    print(f'Top labels: {top_labels}')
    print(f'Top flowers: {top_flowers}')