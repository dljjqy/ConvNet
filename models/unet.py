import torch
import torch.nn as nn

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

class UNet(nn.Module):
    def __init__(self, layers=None, in_c=3, out_c=1, features=16, boundary_type='D'):
        super().__init__()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features, features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)
        
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (2, 2), (2, 2))

        self.uconv3 = DoubleConv(features *16, features * 8)
        self.uconv2 = DoubleConv(features * 8, features * 4)
        self.uconv1 = DoubleConv(features * 4, features * 2)
        self.uconv0 = DoubleConv(features * 2, features * 1)

        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1), padding_mode='reflect')
            
    def forward(self , x):
        x0 = self.dconv0(x)
        x1 = self.dconv1(self.maxpool(x0))
        x2 = self.dconv2(self.maxpool(x1))
        x3 = self.dconv3(self.maxpool(x2))
        x4 = self.dconv4(self.maxpool(x3))

        y = self.uconv3(torch.cat([self.up4(x4),x3], 1))
        y = self.uconv2(torch.cat([self.up3(y), x2], 1))
        y = self.uconv1(torch.cat([self.up2(y), x1], 1))
        y = self.uconv0(torch.cat([self.up1(y), x0], 1))
        y = self.final(y)
        return y

class myUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=8, layers=[1, 2, 4, 8, 16, 32, 64], boundary_type='D'):
        super().__init__()
        self.features = features
        self.layers = layers
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))
        
        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1), padding_mode='reflect')

        self.dconvs = [DoubleConv(in_c, features)]
        self.ups = []
        self.uconvs = []
        for i in range(1, len(layers)):
            self.dconvs.append(DoubleConv(layers[i-1] * features, layers[i] * features))
            self.uconvs.append(DoubleConv(layers[i] * features, layers[i-1] * features))
            self.ups.append(nn.ConvTranspose2d(layers[i] * features, layers[i-1] * features, (2, 2), (2, 2)))
        self.ups[0] = nn.ConvTranspose2d(features * layers[1],  features * layers[0], (3, 3), (2, 2))
        self.uconvs = self.uconvs[::-1]
        self.ups = self.ups[::-1]

        self.dconvs = nn.ModuleList(self.dconvs)
        self.ups = nn.ModuleList(self.ups)
        self.uconvs = nn.ModuleList(self.uconvs)

    def forward(self, x):
        xs = [self.dconvs[0](x)]
        for i in range(1, len(self.layers)):
            xs.append(self.dconvs[i](self.maxpool(xs[i-1])))

        y = self.uconvs[0](torch.cat([self.ups[0](xs[-1]), xs[-2]], 1))
        for i in range(1, len(self.ups)):
            y = self.ups[i](y)
            y = self.uconvs[i](torch.cat([y, xs[-i-2]], 1))

        y = self.final(y)
        return y

if __name__ == '__main__':
    x = torch.rand(3, 3, 64, 64)
    net = UNet(boundary_type='N')
    y = net(x)
    print(y.shape)