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
    parser.add_argument('data_dir', action="store", default="./flowers")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.2)
    parser.add_argument('--gpu', action="store", default="gpu")
    args = parser.parse_args()
    return {
        'data_dir': args.data_dir,
        'path': args.save_dir,
        'learning_rate': args.learning_rate,
        'struct': args.arch,
        'hidden_units': args.hidden_units,
        'powered': args.gpu,
        'epochs': args.epochs,
        'dropout': args.dropout
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

def model_setup(lr=0.001):
    '''
    Setup for building the model
    '''
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 2048)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(2048, 256)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer

def build_and_train_model(dataloaders, model, criterion, optimizer):
    '''
    Builds and tranins the model
    '''
    epochs = training_args['epochs']
    print_every = 5
    steps = 0

    print("--Training starting--")
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model.forward(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to running_loss
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Loss: {running_loss / print_every:.3f}.. "
                      f"Validation Loss: {valid_loss / len(dataloaders['valid']):.3f}.. "
                      f"Accuracy: {accuracy / len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
    print("--Training ended--")

def validate_model(dataloaders, model, criterion):
    '''
    Validates the model
    '''
    test_loss = 0
    accuracy = 0

    print("--Starting model validation --")
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")
    print("--Model validation ended--")

def save_checkpoint(model, img_dataset, name):
    model.class_to_idx = img_dataset.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'structure': 'vgg16',
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

    if torch.cuda.is_available() and training_args['powered'] == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_dir = training_args['data_dir']
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    image_datasets, dataloaders = load_dataset()
    model, criterion, optimizer = model_setup(training_args['learning_rate'])

    build_and_train_model(dataloaders, model, criterion, optimizer)
    validate_model(dataloaders, model, criterion)

    save_checkpoint(model, image_datasets['train'], 'checkpoint.pth')


