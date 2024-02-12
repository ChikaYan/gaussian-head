import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.deform_utils import get_embedder


# from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class Pix2PixDecoder(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, ngf=256, n_upsampling=0, n_blocks=6, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixDecoder, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample      
        mult = 2 # needed when n_upsampling=0
        for i in range(n_upsampling):
            mult = 2**(-i)
            model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        # import pdb; pdb.set_trace()
        model += [nn.ReflectionPad2d(3), nn.Conv2d(int(ngf * mult / 2), output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        # self.models = model
            
    def forward(self, input):
        # for i,m in enumerate(self.models):
        #     print(i)
        #     input = m.to(input.device)(input)
        return self.model(input)  
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class RefineModel:
    def __init__(self, feature_dim=256, t_multires=10, n_blocks=2):
        self.t_multires = t_multires
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.optimizer = None
        self.spatial_lr_scale = 5

        self.network = Pix2PixDecoder(input_nc=feature_dim + time_input_ch, n_blocks=n_blocks).cuda()

    def step(self, x, t):
        # fetch time embedding
        t_emb = self.embed_time_fn(t)
        _, h, w = x.shape
        x = torch.concat([x, t_emb.T.unsqueeze(1).repeat([1, h, w])], axis=0)
        
        return self.network(x.unsqueeze(0))

    def train_setting(self, training_args):
        l = [
            {'params': list(self.network.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "refine"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "refine/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(out_weights_path, 'refine.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "refine"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "refine/iteration_{}/refine.pth".format(loaded_iter))
        self.network.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "refine":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
