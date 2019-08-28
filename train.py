import numpy as np
import training_arguement
import torch
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import sys
import os
import json

    
def main():
    parser = training_arguement.invoke_arguement()
    #parser.add_argument('--version', action='version', version='%(prog)s ' + __version__ + ' by ' + __author__)
    cl_arg = parser.parse_args()
    if not os.path.isdir(cl_arg.data_directory):
        print(f'Data directory {cl_arg.data_directory} was not found.')
        exit(1)

   
    if not os.path.isdir(cl_arg.save_dir):
        print(f'Directory {cl_arg.save_dir} does not exist. Creating...')
        os.makedirs(cl_arg.save_dir)
        
    with open(cl_arg.categories_json, 'r') as f:
         cat_to_name = json.load(f)

    output_size = len(cat_to_name)
    print(f"Images are labeled with {output_size} categories.")
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
        
    training_datasets = datasets.ImageFolder(cl_arg.data_directory, transform=training_transforms)

    training_dataloaders = torch.utils.data.DataLoader(training_datasets, batch_size=32, shuffle=True)
        
    if not cl_arg.arch.startswith("vgg") and not cl_arg.arch.startswith("densenet"):
        print("Only vgg and densenet is supported")
        exit(1)

    print(f"Using a pretrained {cl_arg.arch} network.")
    nn_classfiyer = models.__dict__[cl_arg.arch](pretrained=True)

    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    input_length= 0

    if cl_arg.arch.startswith("vgg"):
        input_length = nn_classfiyer.classifier[0].in_features

    if cl_arg.arch.startswith("densenet"):
        input_length = densenet_input[cl_arg.arch]

    for param in nn_classfiyer.parameters():
        param.requires_grad = False
    nns = OrderedDict()
    hidden_layers = cl_arg.hidden_layers
    hidden_layers.insert(0,  input_length)

    print(f"Building a {len(cl_arg.hidden_layers)} hidden layer classifier with inputs {cl_arg.hidden_layers}")

    for i in range(len(hidden_layers) - 1):
        nns['fc' + str(i + 1)] = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
        nns['relu' + str(i + 1)] = nn.ReLU()
        nns['dropout' + str(i + 1)] = nn.Dropout(p=0.5)

    nns['output'] = nn.Linear(hidden_layers[i + 1], output_size)
    nns['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(nns)

    nn_classfiyer.classifier = classifier

    # Setting the gradients of all parameters to zero.
    nn_classfiyer.zero_grad()

    criterion = nn.NLLLoss()

    print(f"Setting optimizer learning rate to {cl_arg.learning_rate}.")
    optimizer = optim.Adam(nn_classfiyer.classifier.parameters(), lr=cl_arg.learning_rate)

    device = torch.device("cpu")

    if cl_arg.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Using CPU.")

    print(f"Sending model to device {device}.")
    nn_classfiyer = nn_classfiyer.to(device)

    datasets_len = len(training_dataloaders.batch_sampler)

    print_every = 50

    for e in range(cl_arg.epochs):
        total = 0
        correct = 0
        e_loss = 0
        preview_check = 0
        print(f'\nEpoch {e+1} of {cl_arg.epochs}\n----------------------------')
        for ii, (images, labels) in enumerate(training_dataloaders):
            
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs =  nn_classfiyer.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

     
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

           
            iteration = (ii + 1)
            if iteration % print_every == 0:
                average_loss = f'average. loss: {e_loss/iteration:.4f}'
                accuracy = f'accuracy: {(correct/total) * 100:.2f}%'
                print(f'  Batches {preview_check:03} to {iteration:03}: {average_loss}, {accuracy}.')
                preview_check = (ii + 1)

    print('Completed...')

    nn_classfiyer.class_to_idx = training_datasets.class_to_idx
    checkpoint = {
        'epoch': cl_arg.epochs,
        'state_dict': nn_classfiyer.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': nn_classfiyer.classifier,
        'class_to_idx': nn_classfiyer.class_to_idx,
        'arch': cl_arg.arch
    }

    location_saved = f'{cl_arg.save_dir}/{cl_arg.save_name}.pth'
    print(f"Checkpoint is saved in {location_saved}")

    torch.save(checkpoint, location_saved)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)