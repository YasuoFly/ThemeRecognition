import clip
import torch
import torch.nn.functional as F
from torch import nn

class TEG(nn.Module):
    """
    返回两个batch_size*num_classes大小的张量
    """
    def __init__(self, num_classes, pretrained_model, linear_head, logit_scale=4.60517):
        super(TEG, self).__init__()

        self.clip_model, self.preprocess = clip.load(pretrained_model)
        self.head = linear_head
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        self.fusion_linear = nn.Linear(2, 1)
        
    def forward(self, image, text_features):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image) # batch_size*512

        image_features = F.normalize(image_features, dim=1).float()
        head_features = self.head(image_features.float()) # batch_size*num_classes
        # calculate the cosine similarity
        logit = head_features * self.logit_scale.exp() # batch_size*num_classes
        with torch.no_grad():
            similarity = image_features @ text_features.T * self.logit_scale.exp() # batch_size*num_classes

        combine_features = torch.stack([logit,similarity],dim=1).transpose(1,2) # batch_size*num_classes*2
        final_features = (self.fusion_linear(combine_features).squeeze(dim=-1)) # batch_size*num_classes

        return logit, final_features
