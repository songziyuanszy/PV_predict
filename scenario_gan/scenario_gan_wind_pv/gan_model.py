import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, net_params):
        super(Generator, self).__init__()
        
        self.net_params = net_params
        input_channels = net_params['input']
        output_channels = net_params['output']
        kernel_size = net_params['kernel']
        stride = net_params['stride']
        padding = net_params['padding']
        
        self.deconv_0 = nn.Conv2d(input_channels[0], 
                                           output_channels[0],
                                           kernel_size=kernel_size[0], 
                                           stride=stride[0], 
                                           padding=padding[0])
        self.batchnorm_0 = nn.BatchNorm2d(output_channels[0])
        self.act_0 = nn.ReLU(inplace=True)
        
        self.deconv_1 = nn.Conv2d(input_channels[1], 
                                           output_channels[1],
                                           kernel_size=kernel_size[1], 
                                           stride=stride[1], 
                                           padding=padding[1])
        self.batchnorm_1 = nn.BatchNorm2d(output_channels[1])
        self.act_1 = nn.ReLU(inplace=True)
        
        self.deconv_2 = nn.Conv2d(input_channels[2], 
                                           output_channels[2],
                                           kernel_size=kernel_size[2], 
                                           stride=stride[2], 
                                           padding=padding[2])
        self.batchnorm_2 = nn.BatchNorm2d(output_channels[2])
        self.act_2 = nn.ReLU(inplace=True)

        self.deconv_3 = nn.Conv2d(input_channels[3], 
                                           output_channels[3],
                                           kernel_size=kernel_size[3], 
                                           stride=stride[3], 
                                           padding=padding[3])
        self.act_3 = nn.Sigmoid()
        
        self.layer_0 = nn.Sequential(self.deconv_0, 
                                     self.batchnorm_0,
                                     self.act_0)
        
        self.layer_1 = nn.Sequential(self.deconv_1, 
                                     self.batchnorm_1,
                                     self.act_1)
        
        self.layer_2 = nn.Sequential(self.deconv_2, 
                                     self.batchnorm_2,
                                     self.act_2)

        self.layer_3 = nn.Sequential(self.deconv_3, 
                                     self.act_3)
        
        self.model = nn.Sequential(self.layer_0,
                                   self.layer_1,
                                   self.layer_2,
                                   self.layer_3)
        
    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator(nn.Module):
   
    def __init__(self, net_params):
        super(Discriminator, self).__init__()
        
        self.net_params = net_params
        input_channels = net_params['input']
        output_channels = net_params['output']
        kernel_size = net_params['kernel']
        stride = net_params['stride']
        padding = net_params['padding']
        
        self.conv_0 = nn.Conv2d(input_channels[0], 
                                output_channels[0], 
                                kernel_size=kernel_size[0], 
                                stride=stride[0], 
                                padding=padding[0])
        self.batchnorm_0 = nn.BatchNorm2d(output_channels[0])
        self.act_0 = nn.LeakyReLU(0.2, inplace=True)

        
        self.conv_1 = nn.Conv2d(input_channels[1], 
                                output_channels[1], 
                                kernel_size=kernel_size[1], 
                                stride=stride[1], 
                                padding=padding[1])
        self.batchnorm_1 = nn.BatchNorm2d(output_channels[1])
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv_2 = nn.Conv2d(input_channels[2], 
                                output_channels[2], 
                                kernel_size=kernel_size[2], 
                                stride=stride[2], 
                                padding=padding[2])
        self.batchnorm_2 = nn.BatchNorm2d(output_channels[2])
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        
        self.conv_3 = nn.Conv2d(input_channels[3], 
                                output_channels[3], 
                                kernel_size=kernel_size[3], 
                                stride=stride[3], 
                                padding=padding[3])
                
        self.layer_0 = nn.Sequential(self.conv_0, 
                                     #self.batchnorm_0,
                                     self.act_0)
        
        self.layer_1 = nn.Sequential(self.conv_1, 
                                     #self.batchnorm_1,
                                     self.act_1)
        
        self.layer_2 = nn.Sequential(self.conv_2, 
                                     #self.batchnorm_2,
                                     self.act_2)
        
        self.layer_3 = nn.Sequential(self.conv_3)
        
        self.model = nn.Sequential(self.layer_0,
                                   self.layer_1,
                                   self.layer_2,
                                   self.layer_3
                                   )

        self.fc=nn.Sequential(  nn.Linear(73728, 1)  )

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        output=self.fc(x)
        return output
    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

