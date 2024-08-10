import os, sys
sys.path.append("..")
os.chdir(sys.path[0])
import clip
import json
from tqdm import tqdm
import random
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from model import TEG
from utils.utils import MyDataset, read_split_data, TextTensorDataset
from clip import partial_model
import torch.nn.functional as F

def extract_text_features(templates, text_encoder, lab2cname):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'prompts': {},
        'lab2cname': lab2cname,
    }
    templates = templates
    text_encoder.feature_extractor.eval()
    with torch.no_grad():
        for label, cname in lab2cname.items():
            str_prompts = [template.format(cname.replace("_", " ")) for template in templates]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([label for _ in templates]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
            features_dict['prompts'][label] = str_prompts
    return features_dict

def get_text_dataset_per_class(text_dataset):
    print("Building text dataset per class...")
    text_dataset_per_class = {}
    for text, text_label, eot_indices in tqdm(text_dataset):
        text_label = int(text_label)
        if text_label not in text_dataset_per_class:
            text_dataset_per_class[text_label] = []
        text_dataset_per_class[text_label].append([text, eot_indices])
    num_of_templates = len(text_dataset_per_class[text_label])
    for text_label in text_dataset_per_class:
        assert len(text_dataset_per_class[text_label]) == num_of_templates
    return text_dataset_per_class, num_of_templates

def get_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
    with torch.no_grad():
        text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in range(num_classes):
            texts = None
            eot_indices = None
            for i in range(len(text_dataset_per_class[label])):
                text, eot_indice = text_dataset_per_class[label][i]
                text = text.unsqueeze(0).to(device)
                eot_indice = eot_indice.unsqueeze(0).to(device)
                if texts is None:
                    texts = text
                    eot_indices = eot_indice
                else:
                    texts = torch.cat([texts, text], dim=0)
                    eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
            prompt_features = text_encoder(texts, eot_indices)
            prompt_features = prompt_features.mean(dim=0)
            weights[label] = prompt_features
        # normalize the weights
        weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights

def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs, _  = model(images.to(device), text_features.to(device))
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum() 
    return 100 * correct / float(total)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size of the data loader")

    parser.add_argument("--data_path", type=str, default="/home/wangjikai/DATA/Theme25", help="The address of the data file")
    parser.add_argument("--weight", type=float, default=3, help="The weight to add with loss")
    parser.add_argument("--load_pre_path", type=str, default="./checkpoint/weight/best_model.pth", help="The path of pretrained model")
    parser.add_argument("--pretrained_model", type=str, default="ViT-B/32", help="The name of pretrained CLIP model")
    parser.add_argument("--shot", type=int, default=1, help="sample number of shot every category, -1 means all")
    parser.add_argument("--seed", type=int, default=0, help="the random seed number")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    with open('./utils/theme_cls_indices.json','r',encoding='utf-8') as f :
        data = json.loads(f.read())
        lab2cname = {int(k): v for k, v in data.items()}
        id_list = [id for id in data.keys()]
        labels_list = [data[id] for id in id_list]

    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels_list])
    num_classes = len(labels_list)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    templates = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}."
    ]

    clip_model, preprocess = clip.load(args.pretrained_model, jit=False)
    clip_model.float()
    text_encoder = partial_model.get_text_encoder(0, clip_model)
    
    prompt_features = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
        'prompts': [],
        'classnames': [],
    }
    prompt_features = extract_text_features(templates, text_encoder, lab2cname)
    text_dataset = TextTensorDataset(
        prompt_features['features'], prompt_features['labels'], prompt_features['eot_indices'])
    
    # Load the model
    linear_head = nn.Linear(512, num_classes, bias=False)
    linear_head.weight.data = get_zero_shot_weights(
        text_dataset, num_classes, 512, text_encoder.partial_model.cuda())
    model = TEG(num_classes=num_classes, pretrained_model=args.pretrained_model, linear_head=linear_head).train().to(device)
    checkpoint = torch.load(args.load_pre_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load the dataset
    args.data_path = os.path.join(args.data_path, "images")
    _, _, val_images_path, val_images_label = read_split_data(args.data_path, convert2num = True)

    test_dataset = MyDataset(images = val_images_path,
                            labels = val_images_label,
                            transform = preprocess)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=args.num_workers)

    text_features = extract_text_features(["a photo of a {}."], text_encoder, lab2cname)
    text_features = text_features['features']
    text_features = F.normalize(text_features, dim=1)
    
    acc = test(model)
    print('Accuracy on test set: {:.2f} %'.format(acc))