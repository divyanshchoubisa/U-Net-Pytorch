import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_c, out_c, kernel_size=3),
                        nn.ReLU(inplace=True))
    return conv                    

def cropit(tensor, target):
    target_size = target.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]   



class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.uptrans_1 = nn.ConvTranspose2d(in_channels=1024, 
                                            out_channels= 512, kernel_size=2, stride=2)
        self.upconv_1 = double_conv(1024, 512)

        self.uptrans_2 = nn.ConvTranspose2d(in_channels=512 , 
                                            out_channels= 256, kernel_size=2, stride=2)
        self.upconv_2 = double_conv(512, 256)
        
        self.uptrans_3 = nn.ConvTranspose2d(in_channels=256, 
                                            out_channels= 128, kernel_size=2, stride=2)
        self.upconv_3 = double_conv(256, 128)
        
        self.uptrans_4 = nn.ConvTranspose2d(in_channels=128, 
                                            out_channels= 64, kernel_size=2, stride=2)
        self.upconv_4 = double_conv(128, 64)
                                    
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, image):
        #bs, c, h, w
        # Encoder
        x1 = self.down_conv_1(image) #
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool_2x2(x5)
        
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv_5(x8)
        
        #decoder
        x = self.uptrans_1(x9)
        y = cropit(x7, x)
        x = self.upconv_1(torch.cat([x, y], 1)) 

        x = self.uptrans_2(x)
        y = cropit(x5, x)
        x = self.upconv_2(torch.cat([x, y], 1))

        x = self.uptrans_3(x)
        y = cropit(x3, x)
        x = self.upconv_3(torch.cat([x, y], 1))
        
        x = self.uptrans_4(x)
        y = cropit(x1, x)
        x = self.upconv_4(torch.cat([x, y], 1))

        x = self.out(x)
        print(x.size())
        return x


if __name__ =="__main__":
    image = torch.rand((1, 1, 572, 572))
    model = Unet()
    print(model(image))

    






