import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, inc, outc, mode='reflect'):
        super().__init__()
        layers = [
            nn.Conv2d(inc, outc, 3, 1, padding='same', padding_mode=mode),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(outc, outc, 3, 1, padding='same', padding_mode=mode),
            nn.BatchNorm2d(outc),
            nn.ReLU()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class Attention_block(nn.Module):
    
    def __init__(self, F_g, F_l, F_init):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_init, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        shape = x.shape[-2:]
        g1 = F.interpolate(self.w_g(g), shape, mode='bilinear')
        x1 = self.w_x(x)
        psi = self.psi(self.relu(g1 + x1))
        out = x * psi
        return out

class AttUNet(nn.Module):
    def __init__(self, layers=None, in_c=3, out_c=1, features=16, boundary_type = 'D'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features    , features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)

        self.uconv4 = DoubleConv(features *16, features * 8)
        self.uconv3 = DoubleConv(features * 8, features * 4)
        self.uconv2 = DoubleConv(features * 4, features * 2)
        self.uconv1 = DoubleConv(features * 2, features)

        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1), padding_mode='reflect')

        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (2, 2), (2, 2))

        self.ag1 = Attention_block(features *16, features * 8, features * 8)
        self.ag2 = Attention_block(features * 8, features * 4, features * 4)
        self.ag3 = Attention_block(features * 4, features * 2, features * 2)
        self.ag4 = Attention_block(features * 2, features    , features)


    def forward(self, x):
        x1 = self.dconv0(x)
        x2 = self.dconv1(self.maxpool(x1))
        x3 = self.dconv2(self.maxpool(x2)) 
        x4 = self.dconv3(self.maxpool(x3)) 
        x5 = self.dconv4(self.maxpool(x4)) 
        
        g4 = self.ag1(g = x5, x = x4)
        y4 = self.up4(x5)
        y4 = self.uconv4(torch.cat([g4, y4], 1))

        g3 = self.ag2(g = y4, x = x3)
        y3 = self.up3(y4)
        y3 = self.uconv3(torch.cat([g3, y3], 1))

        g2 = self.ag3(g = y3, x = x2)
        y2 = self.up2(y3)
        y2 = self.uconv2(torch.cat([g2, y2], 1))

        g1 = self.ag4(g = y2, x = x1)
        y1 = self.up1(y2)
        y1 = self.uconv1(torch.cat([g1, y1], 1))

        y = self.final(y1)
        return y