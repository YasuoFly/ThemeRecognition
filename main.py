import os, sys
sys.path.append("..")
os.chdir(sys.path[0])
import clip
import json
from tqdm import tqdm
import random
import torch
import argparse
import logging
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from model import TEG
from utils.utils import MyDataset, read_split_data, FewShotDataset, TextTensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from clip import partial_model
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0, help="Initial momentum")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size of the data loader")
    parser.add_argument("--num_epochs", type=int, default=100, help="The number of epoch to train")

    parser.add_argument("--data_path", type=str, default="/home/wangjikai/DATA/Theme25", help="The address of the data file")
    parser.add_argument("--log_dir", type=str, default="./log", help="The path to save log file")
    parser.add_argument("--save_model", type=str, default="./checkpoint/weight", help="The path to save best model")
    parser.add_argument("--loss_plot", type=str, default="'./Plot/loss_weight.svg'", help="The path to save loss figure")
    parser.add_argument("--weight", type=float, default=3, help="The weight to add with loss")
    parser.add_argument("--load_pre_path", type=str, default=None, help="The path of pretrained model")
    parser.add_argument("--lr_decay", type=bool, default=False, help="Choose to keep lr decay on or off")
    parser.add_argument("--pretrained_model", type=str, default="ViT-B/32", help="The name of pretrained CLIP model")
    parser.add_argument("--shot", type=int, default=1, help="sample number of shot every category, -1 means all")
    parser.add_argument("--seed", type=int, default=0, help="the random seed number")
    parser.add_argument("--img_aug", type=bool, default=False, help="choose whether the data is need image augment")

    args = parser.parse_args()

    return args

# Train
def train(epoch, model):

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        outputs1, outputs2 = model(images.to(device), text_features.to(device))
        aux_loss = criterion(outputs1, labels.to(device))
        cls_loss = criterion(outputs2, labels.to(device))
        loss =  weight * aux_loss + cls_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.lr_decay == True:
            scheduler.step()
        
        if args.shot != -1:
            if (batch_idx+1) % total_step == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))
        else:
            if (batch_idx+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))                  
    return loss

# Test
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


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    args.log_path = (os.path.join(args.log_dir, "imgaug{}_weight{}_shot{}_seed{}.log")).format(args.img_aug,args.weight,args.shot,args.seed)
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename=args.log_path,
                        filemode='a')

    logging.info("lr={},momentum={},batch_size={},num_epochs={},dataset={},lr_decay={},shot={}"
                .format(args.lr,args.momentum,args.batch_size,args.num_epochs,os.path.basename(args.data_path),
                args.lr_decay,args.pretrained_model,args.shot))

    with open('./utils/theme_cls_indices.json','r',encoding='utf-8') as f :
        data = json.loads(f.read())
        lab2cname = {int(k): v for k, v in data.items()}
        id_list = [id for id in data.keys()]
        labels_list = [data[id] for id in id_list]

    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels_list])
    num_classes = len(labels_list)

    # Hyper-parameters 
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight = args.weight
    
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

    aug_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),\
    ])

    # Load the dataset
    args.data_path = os.path.join(args.data_path, "images")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path, convert2num = True)

    if args.img_aug == False:
        train_dataset = FewShotDataset(images = train_images_path,
                                labels = train_images_label,
                                transform = preprocess,
                                shot = args.shot)
    else:
        train_dataset = FewShotDataset(images = train_images_path,
                                labels = train_images_label,
                                transform = transforms.Compose([aug_transforms, preprocess]),
                                shot = args.shot)

    test_dataset = MyDataset(images = val_images_path,
                            labels = val_images_label,
                            transform = preprocess)

    # Dataloader
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=args.num_workers)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=args.num_workers)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    best_acc = 0.0
    loss_history = []
    total_step = len(train_loader)

    text_features = extract_text_features(["a photo of a {}."], text_encoder, lab2cname)
    text_features = text_features['features']
    text_features = F.normalize(text_features, dim=1)


    os.makedirs(args.save_model,exist_ok=True)
    if args.load_pre_path is None:
        initepoch = 0
    else:
        if os.path.isfile(args.load_pre_path):
            print("Resume from checkpoint...")
            checkpoint = torch.load(args.load_pre_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
            logging.info("====>loaded checkpoint {} (epoch{})"
                        .format(args.load_pre_path, checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            initepoch = 0   # 如果没进行训练过，初始训练epoch值为1

    for epoch in range(initepoch, num_epochs):
        loss = train(epoch, model)
        # torch.Tensor.cpu(loss)
        loss_history.append(loss.cpu().detach().numpy())
        acc = test(model)
        logging.info('Accuracy on test set: {:.2f} %'.format(acc))
        print('Accuracy on test set: {:.2f} %'.format(acc))
        if args.save_model is None:
            print("argument save_model is None!")
        else:
            checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc}
            latest_model = "latest_model.pth"
            per_3epoch_model = "per_3epoch_model.pth"
            latest_model_path = os.path.join(args.save_model, latest_model)
            # 保存最新模型
            print("save model to the path {}".format(args.save_model))
            torch.save(checkpoint, latest_model_path)
            if acc > best_acc:
                best_acc = acc
                best_model = "best_model.pth"
                torch.save(checkpoint, os.path.join(args.save_model, best_model))
    
    logging.info('best_acc: {:.2f} %'.format(best_acc))