import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(n_channels)
        self.activation = nn.PReLU(n_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out
    
def make_n_convs(n_channels, number):
    layers = []
    for i in range(number):
        layers.append(ConvBlock(n_channels))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels):
        super(InputTransition, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv3d(in_channels, 16, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(16)
        self.activation = nn.PReLU(16)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # print("x.shape: ", x.shape)
        # print("out.shape: ", out.shape)
        # copy x 16 times 
        x16 = torch.cat([x] * (int)(16 / self.in_channels), 1)
        # print("x16.shape: ", x16.shape)
        out = self.activation(torch.add(x16, out))
        # print("out.shape: ", out.shape)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, dropout=False):
        super(Encoder, self).__init__()
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation1 = nn.PReLU(out_channels)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout3d(p=0.5)
        self.activation2 = nn.PReLU(out_channels)
        self.n_convs = make_n_convs(out_channels, n_convs)

    def forward(self, x):
        down = self.down_conv(x)
        down = self.bn(down)
        down = self.activation1(down)
        
        if self.dropout:
            out = self.dropout(down)
        out = self.n_convs(down)
        out = torch.add(down, out)
        out = self.activation2(out)
        
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, dropout=False):
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels // 2)
        self.activation1 = nn.PReLU(out_channels // 2)
        self.dropout = None
        self.skip = nn.Dropout3d(p=0.5)
        if dropout:
            self.dropout = nn.Dropout3d(p=0.5)
        self.activation2 = nn.PReLU(out_channels)
        self.n_convs = make_n_convs(out_channels, n_convs)

    def forward(self, x, x_encoder):
        # e.g. input (1, 256, 16, 16, 16) -> output (1, 128, 32, 32, 32)
        if self.dropout:
            x = self.dropout(x)
        up = self.up_conv(x)
        up = self.bn(up)
        up = self.activation1(up)
        skip_x = self.skip(x_encoder)
        # (1, 128, 32, 32, 32) -> (1, 256, 32, 32, 32)?
        x_cat = torch.cat((up, skip_x), 1)
        
        out = self.n_convs(x_cat)
        out = torch.add(x_cat, out)
        out = self.activation2(out)
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 3, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(3)
        self.activation = nn.PReLU(out_channels)
        self.conv2 = nn.Conv3d(3, out_channels, kernel_size=1)
        
        # if nll:
        #     self.softmax = F.log_softmax
        # else:
        #     self.softmax = F.softmax

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.conv2(out)

        return out  

class VNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNet, self).__init__()
        self.input_tr = InputTransition(in_channels)
        self.encoder1 = Encoder(16, 32, 1)
        self.encoder2 = Encoder(32, 64, 2)
        self.encoder3 = Encoder(64, 128, 3, True)
        self.encoder4 = Encoder(128, 256, 2, True)

        self.decoder1 = Decoder(256, 256, 1, True)
        self.decoder2 = Decoder(256, 128, 2, True)
        self.decoder3 = Decoder(128, 64, 3)
        self.decoder4 = Decoder(64, 32, 2)

        self.output_tr = OutputTransition(32, out_channels)

    def forward(self, x):
        x1 = self.input_tr(x)
        print("x1.shape: ", x1.shape)
        x2 = self.encoder1(x1)
        print("x2.shape: ", x2.shape)
        x3 = self.encoder2(x2)
        print("x3.shape: ", x3.shape)
        x4 = self.encoder3(x3)
        print("x4.shape: ", x4.shape)
        x5 = self.encoder4(x4)
        print("x5.shape: ", x5.shape)

        print("up start")
        x = self.decoder1(x5, x4)
        print("out.shape: ", x.shape)
        x = self.decoder2(x, x3)
        print("out.shape: ", x.shape)
        x = self.decoder3(x, x2)
        print("out.shape: ", x.shape)
        x = self.decoder4(x, x1)
        print("out.shape: ", x.shape)
        x = self.output_tr(x)
        print("out.shape: ", x.shape)
        return x
print("hi")
x = torch.randn((1, 4, 32, 192, 192)).cuda()
model = VNet(4, 3).cuda()
model(x)

# x2 = torch.randn((1, 1, 32, 180, 180)).cuda()
# model2 = InputTransition2(16, None).cuda()
# model2(x2)

print("done")