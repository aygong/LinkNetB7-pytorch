"""
LinkNetB7.py - Define the neural network for LinkNetB7.
"""
import torch
from torch import nn
from torchvision import models


class decoder_block(nn.Module):
    # Instantiate all the modules
    def __init__(self, in_channels, out_channels, stride=2, output_padding=1):
        super(decoder_block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, 
                               stride=stride, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    # Define the block structure
    def forward(self, x):
        """
        decoder_block's forward function.
        Args:
            x (tensor): input tensor
        Returns:
            x (tensor): the output of this block after processing
        """
        x = self.block(x)
        return x
    

class LinkNetB7(nn.Module):
    # Instantiate all the modules
    def __init__(self):
        super(LinkNetB7, self).__init__()
        # Construct a EfficientNetB7 architecture from https://arxiv.org/abs/1905.11946
        # Return a model pre-trained on ImageNet
        efficientnet = models.efficientnet_b7(pretrained=True)
        
        # Input Block
        self.input_block = nn.Sequential(*(list(efficientnet.children())[0][0]))
        
        # Encoder Blocks
        self.encoder1 = nn.Sequential(*(list(efficientnet.children())[0][1]))
        self.encoder2 = nn.Sequential(*(list(efficientnet.children())[0][2]))
        self.encoder3 = nn.Sequential(*(list(efficientnet.children())[0][3]))
        self.encoder4 = nn.Sequential(*(list(efficientnet.children())[0][4]))
        self.encoder5 = nn.Sequential(*(list(efficientnet.children())[0][5]))
        self.encoder6 = nn.Sequential(*(list(efficientnet.children())[0][6]))
        self.encoder7 = nn.Sequential(*(list(efficientnet.children())[0][7]))
        
        # Decoder Blocks
        self.decoder7 = decoder_block(640, 384, stride=1, output_padding=0)
        self.decoder6 = decoder_block(384, 224, stride=2, output_padding=1)
        self.decoder5 = decoder_block(224, 160, stride=1, output_padding=0)
        self.decoder4 = decoder_block(160, 80, stride=2, output_padding=1)
        self.decoder3 = decoder_block(80, 48, stride=2, output_padding=1)
        self.decoder2 = decoder_block(48, 32, stride=2, output_padding=1)
        self.decoder1 = decoder_block(32, 64, stride=2, output_padding=1)
        
        # Output Block
        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=2, padding=1),
            nn.Sigmoid()
        )
    
    # Define the network structure
    def forward(self, x):
        """
        LinkNetB7's forward function.
        Args:
            x (tensor): input tensor
        Returns:
            o1 (tensor): the output of this model after processing
        """
        # Input
        i1 = self.input_block(x)
        
        # Encoding
        e1 = self.encoder1(i1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        
        # Decoding
        d7 = self.decoder7(e7) + e6
        d6 = self.decoder6(d7) + e5
        d5 = self.decoder5(d6) + e4
        d4 = self.decoder4(d5) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        # Output
        o1 = self.output_block(d1)

        return o1