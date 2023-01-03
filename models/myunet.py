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

class Encoder(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        layers = [
            nn.MaxPool2d((2, 2), (2, 2)),
            DoubleConv(inc, outc),
        ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, inc, outc, kernel_size=(2, 2), step_size=(2, 2)):
        super().__init__()
        self.decoder = nn.ConvTranspose2d(inc, outc, kernel_size, step_size)
        self.dconv = DoubleConv(inc, outc)
    
    def forward(self, x1, x2):
        x1 = self.decoder(x1)
        x = torch.cat([x1, x2], 1)
        return self.dconv(x)

class varyUNet(nn.Module):
    '''
    Params:
        in_c: input channels.
        out_c: output channels.
        features: The first layers' features.
            In the following layers,every Encoder will double the features as output channels.
        layers: How deep the UNet is.
        end_padding: The padding mode in the last one conv layer.
            For finite volume method with cell type mesh,end_padding should be 'same' no matter which boundary type.
            For finite difference method with point type mesh, end_padding should be 'valid' for Dirichlet type.
            And (0, 1) for Mixed type boundary.
    '''
    def __init__(self, in_c, out_c=1, features=8, layers=5, end_padding='same'):
        super().__init__()
        self.layers = layers
        self.first = DoubleConv(in_c, features)
        self.decoders = []
        self.encoders = []
        for i in range(layers):
            self.encoders.append(
                Encoder(2**i * features, 
                        2**(i+1) * features))
            self.decoders.append(
                Decoder(2**(layers-i) * features, 
                        2**(layers-i-1) * features))
        
        self.decoders.pop()
        self.decoders.append(Decoder(2 * features, features, (3, 3)))

        self.end = nn.Conv2d(features, out_c, 3, 1, padding=end_padding)

    def forward(self, x):
        Ys = [self.first(x)]
        for encoder in self.encoders:
            X = Ys.pop()
            Ys.append(X)
            Ys.append(encoder(X))

        for decoder in self.decoders:
            X1 = Ys.pop()
            X2 = Ys.pop()
            Y = decoder(X1, X2)
            Ys.append(Y)
        X = Ys.pop()
        return self.end(X)

if __name__ == '__main__':
    x = torch.rand(3, 3, 65, 65)
    net = varyUNet(3, 1, 8, 3)
    y = net(x)
    print(y.shape)

