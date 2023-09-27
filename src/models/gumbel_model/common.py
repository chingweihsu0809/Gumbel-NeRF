import torch
import torch.nn as nn
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth, skips = [], latent_dim=0):
        super(MLP, self).__init__()
        self.skips = skips
        self.fc = [nn.Sequential(nn.Linear(input_size+latent_dim, hidden_size), nn.ReLU())]
        for d in range(1, depth-1):
            inp_size = hidden_size+latent_dim if d in skips else hidden_size
            layer = nn.Sequential(nn.Linear(inp_size, hidden_size), nn.ReLU())
            self.fc += [layer]
        self.fc += [nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU())]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x, shape_latent = None):
        h = x
        if shape_latent is not None:
            h = torch.cat([h, shape_latent], -1)
        for i, m in enumerate(self.fc):
            if i in self.skips:
                h = h + x
                if shape_latent is not None:
                    h = torch.cat([h, shape_latent], -1)
            h = m(h)
        
        return h
    
    
class RGBHead(nn.Module):
    def __init__(self, W, d_viewdir, latent_dim):
        super(RGBHead, self).__init__()
        self.encoding_shape = nn.Linear(W,W)
        rgb_inp = W + d_viewdir + latent_dim
        self.rgb = nn.Sequential(nn.Linear(rgb_inp, W//2), nn.ReLU(),
                                 nn.Linear(W//2, 3), nn.Sigmoid())
        
    def forward(self, shape_out, viewdir, texture_latent = None):

        y = self.encoding_shape(shape_out)

        ## rgb block
        y = torch.cat([y, viewdir], -1)
        if texture_latent is not None:
            y = torch.cat([y, texture_latent], -1)
        rgbs = self.rgb(y)
        
        return rgbs
    
import math
class LatentNet(nn.Module):
    def __init__(self, num_expert, num_instance, dim, uni_head = False, init_method = "random", latent_map_type = "same_dim"):
        super(LatentNet, self).__init__()
        self.num_expert = num_expert
        self.num_codes = num_instance
        if latent_map_type == "same_dim":
            self.obj_code_len = dim
            self.part_code_len = dim
        elif latent_map_type == "half_dim":
            self.obj_code_len = dim
            self.part_code_len = dim // 2
        
        self.uni_head = uni_head
        
        self.make_codes(num_instance, self.obj_code_len, init_method=init_method)
        self.shapefc = nn.Linear(self.obj_code_len, self.part_code_len*num_expert)
        if not uni_head:
            self.colorfc = nn.Linear(self.obj_code_len, self.part_code_len*num_expert)
        
    def forward(self):
        part_shape_codes = self.shapefc(self.shape_codes.weight)
        part_shape_codes = part_shape_codes.reshape(self.num_codes, self.num_expert, self.part_code_len)
        if not self.uni_head:
            part_color_codes = self.colorfc(self.texture_codes.weight)
            part_color_codes = part_color_codes.reshape(self.num_codes, self.num_expert, self.part_code_len)
        
        return part_shape_codes,  self.texture_codes.weight if self.uni_head else part_color_codes
     
    def make_codes(self, num_instance, embdim, init_method):
        self.shape_codes = nn.Embedding(num_instance, embdim)
        self.texture_codes = nn.Embedding(num_instance, embdim)
        if init_method == "random":
            self.shape_codes.weight = nn.Parameter(torch.randn(num_instance, embdim) / math.sqrt(embdim/2))
            self.texture_codes.weight = nn.Parameter(torch.randn(num_instance, embdim) / math.sqrt(embdim/2))
        elif init_method =="zero":
            self.shape_codes.weight = nn.Parameter(torch.zeros(num_instance, embdim))
            self.texture_codes.weight = nn.Parameter(torch.zeros(num_instance, embdim))
        elif init_method == "train_avg":    
            self.shape_codes.weight = None
            self.texture_codes.weight = None
        else:
            raise NotImplementedError
        
        
class LatentNet_nofc(nn.Module):
    def __init__(self, num_expert, num_instance, dim, uni_head = False, init_method = "random", latent_map_type = "same_dim"):
        super(LatentNet_nofc, self).__init__()
        self.num_expert = num_expert
        self.num_codes = num_instance
        if latent_map_type == "same_dim":
            self.obj_code_len = dim
            self.part_code_len = dim
        elif latent_map_type == "half_dim":
            self.obj_code_len = dim
            self.part_code_len = dim // 2
        
        self.make_codes(num_instance, self.part_code_len, self.obj_code_len, init_method=init_method)
        
        
    def forward(self):        
        return self.shape_codes.weight, self.texture_codes.weight
     
    def make_codes(self, num_instance, shape_embdim, texture_embdim, init_method):
        self.shape_codes = nn.Embedding(num_instance, self.num_expert, shape_embdim)
        self.texture_codes = nn.Embedding(num_instance, texture_embdim)
        if init_method == "random":
            self.shape_codes.weight = nn.Parameter(torch.randn(num_instance, self.num_expert, shape_embdim) / math.sqrt(shape_embdim/2))
            self.texture_codes.weight = nn.Parameter(torch.randn(num_instance, texture_embdim) / math.sqrt(texture_embdim/2))
        elif init_method =="zero":
            self.shape_codes.weight = nn.Parameter(torch.zeros(num_instance, self.num_expert,  shape_embdim))
            self.texture_codes.weight = nn.Parameter(torch.zeros(num_instance, texture_embdim))
        elif init_method == "train_avg":    
            self.shape_codes.weight = None
            self.texture_codes.weight = None
        else:
            raise NotImplementedError
        
class CodeNeRFMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth, skips = [], latent_dim=0):
        super(CodeNeRFMLP, self).__init__()
        W = hidden_size
        self.shape_blocks = depth
        for j in range(self.shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
            setattr(self, f"shape_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"shape_layer_{j+1}", layer)

    def forward(self, x, shape_latent = None):
        y = x
        assert shape_latent is not None
        for j in range(self.shape_blocks):
            z = getattr(self, f"shape_latent_layer_{j+1}")(shape_latent)
            y = y + z
            y = getattr(self, f"shape_layer_{j+1}")(y)
        
        return y    
        
class CodeNeRFRGBHead(nn.Module):
    def __init__(self, W, d_viewdir, latent_dim, texture_blocks = 1, moe_out_W = None):
        super(CodeNeRFRGBHead, self).__init__()
        if moe_out_W==None:
            moe_out_W = W
        self.encoding_shape = nn.Linear(moe_out_W,W)
        self.encoding_viewdir = nn.Sequential(nn.Linear(W+d_viewdir, W), nn.ReLU())
        self.texture_blocks = texture_blocks
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"texture_layer_{j+1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3), nn.Sigmoid())
        
    def forward(self, shape_out, viewdir, texture_latent = None):

        y = self.encoding_shape(shape_out)

        y = torch.cat([y, viewdir], -1)
        y = self.encoding_viewdir(y)
        for j in range(self.texture_blocks):
            z = getattr(self, f"texture_latent_layer_{j+1}")(texture_latent)
            y = y + z
            y = getattr(self, f"texture_layer_{j+1}")(y)
        rgbs = self.rgb(y)
        
        return rgbs