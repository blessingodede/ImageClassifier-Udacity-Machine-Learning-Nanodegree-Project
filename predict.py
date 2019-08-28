import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import warnings
import predicting_arguement
import json

def main():
    parser = predicting_arguement.invoke_arguement()
   
  
    cl_args = parser.parse_args()
    device = torch.device("cpu")
    
    if cl_args.use_gpu:
        device = torch.device("cuda:0")
    with open(cl_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)
    
    checkpoint_model = loading_checkpoint(device, cl_args.checkpoint_file)

    top_probabilities, top_classes = predict(cl_args.path_to_image, checkpoint_model, cl_args.top_k)

    label = top_classes[0]
    probabilities = top_probabilities[0]
    
    print(f'\nPredictions Information\n-----------------------')
    print(f'Flower Name      : {cat_to_name[label]}')
    print(f'Labels       : {label}')
    print(f'Probabilities : {probabilities*100:.2f}%')

    print(f'\nTop Predictions\n---------------------------------')
    probabilities = top_probabilities[0]
    
    for i in range(len(top_probabilities)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_probabilities[i]*100:.2f}%")
def predict(image_path, model, topk=5):
    model.eval()
    model.cpu()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
         output = model.forward(image)
         top_probabilities, top_labels = torch.topk(output, topk)
         top_probabilities = top_probabilities.exp()
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
   
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_probabilities.numpy()[0], mapped_classes

def loading_checkpoint(device, file_path='my_checkpoint.pth'):
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model = model.to(device)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    

    return model

def process_image(image):
    pil_images = Image.open(image).convert("RGB")
    image_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    pil_images = image_transforms(pil_images)
    return pil_images

if __name__ == '__main__':
  
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
        
    