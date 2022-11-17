import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from max_vit import MaxViT,MaxVit_GAN_layer,MaxViT_layer,CrissCrossAttention,FeedForward,MBConv
from einops import rearrange
from einops.layers.torch import Rearrange
import math

class MyData(Dataset):
    ## Edit below here
    def __init__(self, video_data, label_data, bpm_data):
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.video_data = np.resize(video_data,(video_data.shape[0],64,256,3))
        # self.video_data = video_data.repeat(2,axis=2).repeat(2,axis=3)
        self.video_data = video_data
        # smaller_img.repeat(2, axis=0).repeat(2, axis=1)
        # self.video_data = video_data[:,:,:,:,:]*2-1
        self.label = label_data
        # self.bpm = bpm_data


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # video_data = torch.tensor(self.video_data[index],dtype=torch.float32)
        video_data = torch.tensor(np.transpose(self.video_data[index], (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(self.label[index], dtype=torch.float32)
        # bpm_data = torch.tensor(self.bpm[index],dtype=torch.float32)


        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')
            # bpm_data = bpm_data.to('cuda')

        return video_data, label_data  # , bpm_data

    def __len__(self):
        return len(self.label)


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        length = 32
        height,width = (128,128)

        self.main_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b l) c h w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b l) c h w -> b c l (h w)', l=length)
        )
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,32),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        )

        self.ptt_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b h) c l w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b h) c l w -> b c h (l w)', h=height)
        )
        self.ptt_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))

        self.bvp_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b w) c l h'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b w) c l h -> b c w (l h)', w=width)
        )
        self.bvp_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        self.adaptation = nn.AdaptiveAvgPool2d((4,2))

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32,16))
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        main_1 = self.main_seq_stem(x)
        main_2 = self.main_seq_max_1(main_1)
        #ver1
        main_3 = self.sa_main(main_2)
        #ver2
        # main_att = self.sa_main(main)
        # main = main_att*main + main
        main_4 = self.adaptive(main_3)
        main_5 = rearrange(main_4,'b c l (w h) -> b c l w h',w = 4, h = 4)
        # main = self.main_seq_max_2(main)

        bvp_1 = self.bvp_seq_stem(x)
        bvp_2 = self.bvp_seq_max_1(bvp_1)
        #ver1
        bvp_3 = self.sa_bvp(bvp_2)
        #ver2
        # bvp_att = self.sa_bvp(bvp)
        # bvp = bvp_att*bvp + bvp
        bvp_4 = rearrange(bvp_3, 'b c w (l h) -> b c l w h', l=4, h=4)


        ptt_1 = self.ptt_seq_stem(x)
        ptt_2 = self.ptt_seq_max_1(ptt_1)
        #ver1
        ptt_3 = self.sa_bvp(ptt_2)
        #ver2
        # ptt_att = self.sa_bvp(ptt)
        # ptt = ptt_att*ptt + ptt
        ptt_4 = rearrange(ptt_3, 'b c h (l w) -> b c l w h', l=4, w=4)

        # att = ptt_4@bvp_4
        att = F.interpolate(ptt_4,scale_factor=(1,1,1/16))
        main_6 = main_5 * F.interpolate(att,scale_factor=(8,1,1)) + main_5

        main_7 = rearrange(main_6,'b c l w h -> b c l (w h)')
        out_1 = self.max_vit(main_7)

        out_2 = torch.squeeze(out_1)
        out_3 = torch.mean(out_2,dim = -1)

        out_att = self.be_conv1d(out_3)
        out_4 = (1 + self.sigmoid(out_att)) * out_3
        out_5 = self.out_conv1d(out_4)
        out = torch.squeeze(out_5)
        # out = self.linear(out)
        return out
            # ,[main_1,main_2,main_3,main_4,main_5,main_6,main_7],\
            #    [bvp_1,bvp_2,bvp_3,bvp_4],[ptt_1,ptt_2,ptt_3,ptt_4],[att,out_att],\
            #    [out_1,out_2,out_3,out_4,out_5]

class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)

# def plot_predictions(data, pred):
#     plt.scatter(
#         data[:, 0], data[:, 1], c=pred,
#         cmap='bwr')
#     plt.scatter(
#         data[:, 0], data[:, 1], c=torch.round(pred),
#         cmap='bwr', marker='+')
#     plt.show()


# if __name__ == "__main__":
#     alice_data, alice_targets, bob_data, bob_targets = XORDataset(
#         100).split_by_label()
#     plt.scatter(alice_data[:, 0], alice_data[:, 1], label='Alice')
#     plt.scatter(bob_data[:, 0], bob_data[:, 1], label='Bob')
#     plt.legend()
#     plt.show()
