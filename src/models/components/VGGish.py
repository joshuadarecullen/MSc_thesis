import torch
from torch import nn

'''
adapted from this tuturial https://github.com/harritaylor/torchvggish/tree/master/docs
'''

VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"

def make_layers():
    layers = []
    in_channels = 1
    for v in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGish(nn.Module):
    """
    VGGish with pre-loaded weights
    """
    def __init__(self,
                 pretrain: bool = True,
                 frozen: bool = True):
        super().__init__()
        self.features = make_layers()
        self.embeddings = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=128),
            nn.ReLU(True)
        )
        if pretrain:
            print("Loading pre-trained weights for VGGish")
            state_dict = torch.hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
            self.load_state_dict(state_dict)
            if frozen:
                print("Freezing VGGish parameters")
                for param in self.parameters():
                   param.requires_grad_(False)
        else:
            print("Using VGGish architecture without pre-trained weights")

    def flatten(self,
                x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.embeddings(x)
        return x
