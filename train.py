#!/bin/sh
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict


def arg_parser():
    parser = argparse.ArgumentParser(description="Parser for train.py")
    parser.add_argument('--data_dir', action="store", default="./flowers")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--device', action="store", default="gpu")
    parser.add_argument('--model_struct', action="store", default="vgg16", dest="model_struct")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=256)
    parser.add_argument('--epochs', action="store", default=3, type=int)

    args = parser.parse_args()
    return {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'device': args.device,
        'model_struct': args.model_struct,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'epochs': args.epochs
    }


def load_dataset():
    '''
    Load the dataset with ImageFolder and applies data trasnforms
    and returns the data loaders
    '''

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, defines the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

    return image_datasets, dataloaders

def model_setup(device, model_struct, lr, hidden_layer):
    '''
    Setup for building the model
    '''
    if model_struct == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 2048)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(2048, hidden_layer)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_layer, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer

def model_validation(dataset, model, criterion, device):
    model.eval()
    accuracy = 0.0
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model(inputs)
            batch_loss = criterion(log_ps, labels)
            valid_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss / len(dataset), accuracy / len(dataset)

def train_model(dataloaders, model, criterion, device, epochs):
    print_steps = 5

    print(f'--- model training started ---')
    for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass + compute loss + backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute total loss and print statistics
            running_loss += loss.item()
            if i % print_steps == 0:
                valid_loss, accuracy = model_validation(dataloaders['valid'], model, criterion, device)
                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Loss: {running_loss / print_steps:.3f}.. "
                      f"Validation Loss: {valid_loss:.3f}.. "
                      f"Accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()
    print(f'--- model training ended ---')

def save_checkpoint(model, img_dataset, name):
    model.class_to_idx = img_dataset.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'learning_rate': 0.001,
                'classifier': model.classifier,
                'epochs': training_args['epochs'],
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, name)
    print(f'Saved the checkopoint')


if __name__ == "__main__":
    global training_args
    training_args = arg_parser()

    if torch.cuda.is_available() and training_args['device'] == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_dir = training_args['data_dir']
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    image_datasets, dataloaders = load_dataset()

    model_struct = training_args['model_struct']
    lr = training_args['learning_rate']
    hidden_layer = training_args['hidden_units']
    epochs = training_args['epochs']
    saving_path = training_args['save_dir']

    model, criterion, optimizer = model_setup(device,  model_struct, lr, hidden_layer)

    train_model(dataloaders, model, criterion, device, epochs)

    model_validation(dataloaders, model, criterion, device)

    save_checkpoint(model, image_datasets['train'], saving_path)


