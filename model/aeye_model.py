import torch.nn as nn
from model.utils import conv_3x3_bn, MV2Block
from model.modified_mobilevit import ModifiedMobileViT
from data.radial_tokenizer import RadialTokenizer

class AEyeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config['dims']
        embed_dim = config['embed_dim']
        self.tokenizer = RadialTokenizer()
        self.stage1 = conv_3x3_bn(3, dims[0], stride=2)
        self.stage2 = MV2Block(dims[0], dims[1], stride=2)
        self.stage3 = ModifiedMobileViT(dims[1], embed_dim)
        self.stage4 = MV2Block(dims[1], dims[2], stride=2)
        self.stage5 = ModifiedMobileViT(dims[2], embed_dim)
        self.stage6 = MV2Block(dims[2], dims[3], stride=2)
        self.stage7 = ModifiedMobileViT(dims[3], embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[3], 1)
    def forward(self, x_img):
        tokens = self.tokenizer(x_img)
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x, tokens)
        x = self.stage4(x)
        x = self.stage5(x, tokens)
        x = self.stage6(x)
        x = self.stage7(x, tokens)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)