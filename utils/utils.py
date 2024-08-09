from cProfile import label
import os, sys
import shutil
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import random
import json
from shutil import copy, rmtree
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath),exist_ok=True)
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = os.path.join(output, "log.txt")

    if os.path.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(fpath)

# 读取日志文件
def read_logs(log_path):
    fo = open(log_path,"rb")
    fo.seek(0,2)
    for line in fo.readlines():
        output = str(line.decode())
    fo.close()
    return output

def divide_file(num, fpath, new_fpath):
    '''
    一个文件夹中有多个文件，把所有文件分成 num 份，新建文件夹放入
    num:等分文件份数
    fpath:原文件所存放的文件夹路径
    new_fpath:新文件所存放的文件夹路径
    '''
        
    #每个文件夹存放的个数
    num = num
    file_path = fpath
    new_file_path = new_fpath

    list_ = os.listdir(file_path)
    list_.sort()
    if num > len(list_):
        print('num长度需小于:', len(list_))
        exit()
    if int(len(list_) % num) == 0:
        num_file = int(len(list_) / num)
    else:
        num_file = int(len(list_) / num) + 1
    cnt = 0
    for n in range(1, num_file + 1):  # 创建文件夹
        new_file = os.path.join(new_file_path + str(n))
        if os.path.exists(new_file + str(cnt)):
            print('该路径已存在，请解决冲突', new_file)
            exit()
        print('创建文件夹：', new_file)
        os.mkdir(new_file)
        list_n = list_[num * cnt:num * (cnt + 1)]
        for m in list_n:
            old_path = os.path.join(file_path, m)
            new_path = os.path.join(new_file, m)
            shutil.copy(old_path, new_path)
        cnt += 1

class NLTK():
# 获取单词的词性
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    def split2word(self, sentence):
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)     # 获取单词词性
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
        return lemmas_sent
    def combine_word(self,lemmas_sent):
        flag = False
        phrase = ""
        caption = []
        for tag in lemmas_sent:
            if tag[1] == "NN" or tag[1]  == "JJ":
                flag = True
                phrase = phrase + tag[0] # 合词
            else:
                if len(phrase) != 0:
                    caption.append(phrase)
                flag = False
                phrase = ""
                caption.append(tag[0])
        if len(phrase) != 0 and flag:
            caption.append(phrase)
        return caption



"""
training steps:
    1.Prepare dataset
        tools: Dataset & Dataloader
    2.Design model using Class
        inherit from nn.Module
    3.Construct loss and optimizer
        using Pytorch API
    4.Training cycle
        forward, backward, update
"""
def CreateDataset(filepath):
    data_list, label_list, dir_list = [], [], []
    class_flag = -1
    for path,dirs,files in os.walk(filepath):
        for dir in dirs:
            dir_list.append(dir)
        for file in files:
            data = os.path.join(path, file)
            data_list.append(data)
            label_list.append(dir_list[class_flag])
        class_flag += 1
        
    # if not os.path.exists("./dataset.txt"):
    with open("./dataset.txt", "w", encoding="UTF-8") as f:
        for data, label in zip(data_list, label_list):
            f.write(str(data + "\t" + label + "\n"))

def load_data(filepath):
    data_list, label_list, dir_list = [], [], []
    class_flag = -1
    for path,dirs,files in os.walk(filepath):
        for dir in dirs:
            dir_list.append(dir)
        for file in files:
            data = os.path.join(path, file)
            data_list.append(data)
            label_list.append(dir_list[class_flag])
        class_flag += 1
        
    return data_list, label_list

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)
    
def read_split_data(root:str, split_rate:float = 0.20, convert2num:bool = False, plot_image:bool = False):
    """分割数据集为训练集和验证集

    Args:
        root (str): 数据集根目录的路径
        split_rate (float, optional): 分割验证集的比例, 默认为0.2
        convert2num (bool, optional): 是否将类别对应转化为数字, 默认为False(否).
        plot_image (bool, optional): 是否展示每个类别的样本数, 默认为False(否).
    """
    # 保证随机可复现
    data_rng = random.Random(0)

    # 指向文件夹
    assert os.path.exists(root), "dataset root: '{}' does not exist.".format(root)
    # 筛选出文件夹并将其名称作为类别
    classes = [cla for cla in os.listdir(root)
                    if os.path.isdir(os.path.join(root, cla)) and len(os.listdir(os.path.join(root, cla))) > 0]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称及对应数字索引
    if convert2num:
        class_indices = dict((k, v) for v, k in enumerate(classes))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
    
    train_images_path = [] # 存储训练集的所有图片路径
    train_images_label = [] # 存储训练集的所有图片对应图片索引信息
    val_images_path = [] # 存储验证集的所有图片路径
    val_images_label = [] # 存储验证集的所有图片对应索引信息
    every_class_num = [] # 存储每个类别的样本总数
    supported = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'] # 支持的文件后缀格式   
    
    # 遍历每个文件夹下的文件
    for cla in classes:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径, splitext用于获取文件后缀名
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序以保证图片路径按序读入
        images.sort()
        # 获取该类别对应的索引
        if convert2num:
            image_class = class_indices[cla]
        # 记录该类被的样本数量
        every_class_num.append(len(images))
        # 随机采样验证集的索引
        val_path = data_rng.sample(images, k=int(len(images) * split_rate))
        # 顺序采样验证集索引
        # val_path = images[-int(len(images) * split_rate):]
        
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                if convert2num:
                    val_images_label.append(image_class)
                else:
                    val_images_label.append(cla)
            else:
                train_images_path.append(img_path)
                if convert2num:
                    train_images_label.append(image_class)
                else:
                    train_images_label.append(cla)
    
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    
    if plot_image:
        plt.figure(figsize=(20, 8))
        plt.rcParams['font.family'] = ['serif']
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.bar(range(len(classes)), every_class_num, align="center",label="Number")
        plt.xticks(range(len(classes)), ["Non-motorized\nvehicle" if cls == "Non-motorized vehicle" else cls for cls in classes])
        plt.xticks(rotation=90)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=20, loc="best")
        # for i, v in enumerate(every_class_num):
        #     plt.text(x=i, y=v + 4, s=str(v), ha='center')
        # plt.xlabel('image class')
        # plt.ylabel('number of iamges')
        # plt.title('class distribution')
        plt.savefig("../data_distribution.svg", bbox_inches='tight')
    print("processing done!")
    # print(plt.rcParams['font.serif'])
    
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data_multi(root:str, split_rate:float = 0.2, convert2num:bool = False, plot_image:bool = False):
    """分割数据集为训练集和验证集

    Args:
        root (str): 数据集根目录的路径
        split_rate (float, optional): 分割验证集的比例, 默认为0.2
        convert2num (bool, optional): 是否将类别对应转化为数字, 默认为False(否).
        plot_image (bool, optional): 是否展示每个类别的样本数, 默认为False(否).
    """
    # 保证随机可复现
    # random.seed(0)

    # 指向文件夹
    assert os.path.exists(root), "dataset root: '{}' does not exist.".format(root)
    # 筛选出文件夹并将其名称作为类别
    classes = [cla for cla in os.listdir(root)
                    if os.path.isdir(os.path.join(root, cla)) and len(os.listdir(os.path.join(root, cla))) > 0]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称及对应数字索引
    if convert2num:
        class_indices = dict((k, v) for v, k in enumerate(classes))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
    
    train_images_dic = {} # 存储训练集的所有图片路径和标签的字典
    val_images_dic = {} # 存储验证集的所有图片路径和标签的字典
    every_class_num = [] # 存储每个类别的样本总数
    supported = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'] # 支持的文件后缀格式   
    
    # 遍历每个文件夹下的文件
    for cla in classes:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径, splitext用于获取文件后缀名
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        if convert2num:
            class_idx = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 随机采样验证集的索引
        # val_path = random.sample(images, k=int(len(images) * split_rate))
        # 顺序采样验证集索引
        val_path = images[-int(len(images) * split_rate):]
        
        for img_path in images:
            if img_path in val_path:
                if img_path not in val_images_dic:
                    if convert2num:
                        val_images_dic[img_path] = class_idx
                    else:
                        val_images_dic[img_path] = cla
                else:
                    if convert2num:
                        val_images_dic[img_path].append(class_idx)
                    else:
                        val_images_dic[img_path].append(cla)
            else:
                if img_path not in val_images_dic:
                    if convert2num:
                        train_images_dic[img_path] = class_idx
                    else:
                        train_images_dic[img_path] = cla
                else:
                    if convert2num:
                        train_images_dic[img_path].append(class_idx)
                    else:
                        train_images_dic[img_path].append(cla)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    
    if plot_image:
        plt.bar(range(len(classes)), every_class_num, align="center")
        plt.xticks(range(len(classes)), classes)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of iamges')
        plt.title('class distribution')
        plt.show()
    print("processing done!")
    
    return train_images_dic, val_images_dic

@staticmethod
def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels
    
class MyDataset(Dataset):
    def __init__(self, images:list, labels:list, transform = None):
        """
        processing data:
            1:All data load to memory (Structured Data)
            2:Define a list to store the path and label of each sample
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        # if img.mode != "RGB":
        #     raise ValueError("Img is not RGB type!")
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.images)
        

def listdir(path, list_name):  # 传入存储的list
    '''
    递归得获取对应文件夹下的所有文件名的全路径
    存在list_name 中
    :param path: input the dir which want to get the file list
    :param list_name:  the file list got from path
	no return
    '''
    list_dirs = os.listdir(path)
    list_dirs.sort()
    for file in list_dirs:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    list_name.sort()
    # print(list_name)

class Few_shot_ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256),shot=-1):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        :param shot:  how many samples will be used for training?   shot=-1 : all.  shot=5: 5 imgs will be used for training
        """
        super(Few_shot_ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()

        shot = min(shot, len(self.frame))
        self.shot = shot
        if shot != -1:
            self.frame = self.frame[:shot]

        self.transform = transform
        self.im_size = im_size
    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            # image_path = os.path.join(self.root, img_names[i])
            image_path = img_names[i]
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg' or image_path[-5:] == '.JPEG':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # new_idx = idx% len(self.frame)
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img

class FewShotDataset(Dataset):
    def __init__(self, images:list, labels:list, transform=None, shot=-1, aug_emb = None):
        """
        processing data:
            1:All data load to memory (Structured Data)
            2:Define a list to store the path and label of each sample
        """
        shot = min(len(images), shot)
        self.shot = shot
        self.aug_emb = aug_emb

        if shot != -1:
            image_dict, label_dict = {}, {}
            for path in images:
                category = path.split('/')[-2]
                if category not in image_dict:
                    image_dict[category] = [path]
                else:
                    image_dict[category].append(path)

            for path, label in zip(images, labels):
                category = path.split('/')[-2]
                if category not in label_dict:
                    label_dict[category] = [label]
                else:
                    label_dict[category].append(label)
            few_images,few_labels = [], []
            for category, path_list in image_dict.items():
                # 取固定的随机 shot 作为样本数
                few_images.extend(random.sample(path_list, k=shot))
                # 顺序取前 shot 作为样本数
                # few_images.extend(path_list[:shot])
            for category, label_list in label_dict.items():
                few_labels.extend(random.sample(label_list, k=shot))
                # few_labels.extend(label_list[:shot])
            self.images = few_images
            self.labels = few_labels
        else:
            self.images = images
            self.labels = labels
        # print(few_images)
        # print(type(few_images))
        # print("few_labels",len(few_labels))
        self.transform = transform
        
    def __getitem__(self, index):
        # print("__getitem__, index:",index)
        # print(len(self.images))
        index = index % len(self.images)
        img = Image.open(self.images[index])
        # if img.mode != "RGB":
        #     raise ValueError("Img is not RGB type!")
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.aug_emb is not None:
            emb_list = self.aug_emb[label]
            txt_aug_emb = random.sample(emb_list, 1)[0].squeeze(0)
            return img, label, txt_aug_emb
        else:
            return img, label
    def __len__(self):
        return len(self.images)

class FewShotDataset_old(Dataset):
    def __init__(self, images:list, labels:list, transform=None, shot=-1):
        """
        processing data:
            1:All data load to memory (Structured Data)
            2:Define a list to store the path and label of each sample
        """
        shot = min(len(images), shot)
        self.shot = shot

        if shot != -1:
            image_dict, label_dict = {}, {}
            for path in images:
                category = path.split('/')[-2]
                if category not in image_dict:
                    image_dict[category] = [path]
                else:
                    image_dict[category].append(path)

            for path, label in zip(images, labels):
                category = path.split('/')[-2]
                if category not in label_dict:
                    label_dict[category] = [label]
                else:
                    label_dict[category].append(label)
            few_images,few_labels = [], []
            for category, path_list in image_dict.items():
                # 取固定的随机 shot 作为样本数
                few_images.extend(random.sample(path_list, k=shot))
                # 顺序取前 shot 作为样本数
                # few_images.extend(path_list[:shot])
            for category, label_list in label_dict.items():
                few_labels.extend(random.sample(label_list, k=shot))
                # few_labels.extend(label_list[:shot])
            self.images = few_images
            self.labels = few_labels
        else:
            self.images = images
            self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        index = index % len(self.images)
        img = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.images)
    
class TextTensorDataset(Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)
    
class AugTextTensorDataset(Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)
    
def count_num_param(model=None, params=None):
    r"""Count number of parameters in a model.

    Args:
        model (nn.Module): network model.
        params: network model`s params.
    Examples::
        >>> model_size = count_num_param(model)
    """

    if model is not None:
        return sum(p.numel() for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].numel()
            else:
                s += p.numel()
        return s

    raise ValueError("model and params must provide at least one.")