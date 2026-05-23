import torch
from torch_geometric import nn as torch_g_nn
from spherical_models import SDPAConv, SDPAConvTranspose, SphereSkipConnection, SpherePixelShuffle, SphereGDN
from utils import common_function as util_common
from utils import healpix as hp_utils
# pyright: reportGeneralTypeIssues=warning

class SLB_Downsample(torch.nn.Module):
    r"""Spherical Layer Block for Downsampling consists of:
     one or several convolutions (with desired aggregation of conv outputs) +
     optional non-linearity on the output of conv +
     optional attention module +
     optional down sampling

    Args:
        attention_pos (int): -1 if no attention, 0 if attention at beginning, 1 if attention after non-linearity 
        n_rbs (int): number of residual blocks if activation is 'RB'
    """
    def __init__(self,
                 conv_name:str,
                 in_channels:int,
                 out_channels:int,
                 bias:bool=True,
                 hop:int=1,
                 skip_conn_aggr=None,
                 activation=None,
                 activation_args:dict=dict(),
                 pool_func=None,
                 pool_size_sqrt:int=1,
                 attention_pos:int=-1,
                 n_rbs:int=3,
                 single_conv:bool=False,
                 mask:str='full',
                 conv1x1:bool=False):
        if (conv_name != 'SDPAConv') and (mask not in ['A', 'B', 'full'] or conv1x1):
            raise NotImplementedError("masked and 1x1 convolution only implemented for SDPAConv")
        super().__init__()
        # 1- Setting convolution
        self.node_dim = 1
        self.list_conv = torch.nn.ModuleList()
        num_conv = hop if conv_name in ["GraphConv", "SDPAConv"] else 1
        if single_conv: num_conv = 1
        if skip_conn_aggr=='cat':
            out_channels //= num_conv

        if conv_name == "ChebConv":
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, K=hop+1, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['TAGConv', 'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, K=hop, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, aggr='mean', bias=bias, node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels, aggr='mean', bias=bias, node_dim=self.node_dim) for _ in range(num_conv - 1)]))  # Maybe later not all of them has aggr as argument
        elif conv_name == "SDPAConv":
            conv = SDPAConv
            n_firstHopNeighbors = 8
            n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=1 if not single_conv else hop)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, kernel_size=n_neighbors+1 if not conv1x1 else 1, bias=bias, node_dim=self.node_dim, mask=mask))
            # mask only for first convolution
            self.list_conv.extend(torch.nn.ModuleList(
                [conv(in_channels=out_channels, out_channels=out_channels, kernel_size=n_neighbors+1 if not conv1x1 else 1, bias=bias, node_dim=self.node_dim) for _ in range(num_conv-1)]))
        else:
            raise ValueError('Convolution is not defined')

        self.in_channels = in_channels
        assert len(self.list_conv) == num_conv, "list conv must be equal to num_conv"
        # Setting aggregation of convolution results
        self.out_channels = out_channels
        if (num_conv > 1) and (skip_conn_aggr not in ["non", "none"]):
            self.skipconn = SphereSkipConnection(skip_conn_aggr)
            if skip_conn_aggr == "cat":
                self.out_channels *= num_conv
        else:
            self.register_parameter('skipconn', None)

        self.conv_out_channels = self.out_channels
        
        # 2- Setting nonlinearity
        self.activation_name = activation
        if activation is None:
            self.activation_name = ''
            self.register_parameter('activation', None)
        elif activation.upper() == "GDN":
            self.activation = SphereGDN(self.out_channels, **activation_args)
        elif activation.upper() == "RB":
            self.activation = torch.nn.ModuleList([ResBlock(conv_name, self.out_channels, bias) for _ in range(n_rbs)])
        else:
            self.activation = getattr(torch.nn, activation)(**activation_args)

        # 3- Setting Downsampling
        self.pool_size_sqrt = pool_size_sqrt
        self.pool_size = pool_size_sqrt*pool_size_sqrt
        assert ((self.pool_size==1) and (pool_func is None)) or ((self.pool_size > 1) and (pool_func is not None)), "pool_func and pool_size must match."
        if (pool_func is None) or (self.pool_size==1):
            self.register_parameter('pool', None)
        elif pool_func == 'max_pool':
            self.pool = getattr(torch.nn, "MaxPool3d")(kernel_size=(1, self.pool_size, 1))
        elif pool_func == "avg_pool":
            self.pool = getattr(torch.nn, "AvgPool3d")(kernel_size=(1, self.pool_size, 1))
        elif pool_func == "stride":
            self.pool = "stride"
        else:
            raise ValueError('Pooling is not defined')
        
        # 4- Setting attention module
        assert attention_pos in [-1, 0, 1], "attention_pos must be -1, 0 or 1 for Encoder"
        self.attention_pos = attention_pos
        if attention_pos in [0, 1]:
            self.attention_module = AttentionModule(conv_name, self.out_channels if attention_pos else in_channels, bias, activation='ReLU', n_rb_trunk=3, n_rb_mask=3)
        else:
            self.attention_module = None

    def forward(self, x, index=None, weight=None, valid_index=None, mapping=None, index_=None, weight_=None, valid_index_=None):  # x is a tensor of size [batch_size, num_nodes, num_features]
        'index_, weight_ and valid_index_ denote the respective tensors after convolution (if RB as nonlinearity or attention used)'
        device = x.device
        if index is not None: index = index.to(device)
        if weight is not None: weight = weight.to(device)
        if valid_index is not None: valid_index = valid_index.to(device)
        # index_, weight_ and valid_index_ for attention module after convolution
        if index_ is not None: index_ = index_.to(device)
        if weight_ is not None: weight_ = weight_.to(device)
        if valid_index_ is not None: valid_index_ = valid_index_.to(device)
        
        if self.attention_pos == 0:
            x = self.attention_module(x, index, weight, valid_index)
        
        xs = []
        for conv in self.list_conv:
            if conv.__class__.__name__ == "SDPAConv":
                x = conv(x, neighbors_indices=index, neighbors_weights=weight, valid_index=valid_index)
            else:
                x = conv(x, edge_index=index, edge_weight=weight)
            xs += [x] if self.pool!="stride" else [x.index_select(self.node_dim, torch.arange(0, x.size(self.node_dim), step=self.pool_size, device=x.device))]
        x = self.skipconn(xs) if self.skipconn is not None else xs[-1]
        
        if mapping is not None:
            mapping = mapping.to(device)
            x = x.index_select(self.node_dim, mapping)

        if (self.activation is not None) and (self.activation_name.upper() != "RB"):
            x = self.activation(x)

        if (self.pool is not None) and (not isinstance(self.pool, str)):
            x = torch.squeeze(self.pool(torch.unsqueeze(x, dim=0)), dim=0)
        
        if self.activation_name.upper() == "RB":
            for conv in self.activation:
                x = conv(x, index_, weight_, valid_index_)
        
        if self.attention_pos == 1:
            x = self.attention_module(x, index_, weight_, valid_index_)
        return x
    
    def get_conv_input_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct data for the "input of the conv".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        return 0

    def get_output_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct data for the "output of the module".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if self.pool is None:
            return 0
        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionDownsampled(0, self.pool_size_sqrt)

class SLB_Upsample(torch.nn.Module):
    r"""Spherical Layer Block for Upsampling sists of:
     one or several convolutions (with desired aggregation of conv outputs) +
     optional non-linearity on the output of conv +
     optional attention module +
     optional up-sampling

    Args:
        attention_pos (int): -1 if no attention, 0 if attention at beginning, 1 if attention after non-linearity, 2 if attention after conv
        n_rbs (int): number of residual blocks if activation is 'RB'
    """
    def __init__(self,
                 conv_name:str,
                 in_channels:int,
                 out_channels:int,
                 bias:bool=True,
                 hop:int=1,
                 skip_conn_aggr=None,
                 activation=None,
                 activation_args:dict=dict(),
                 unpool_func=None,
                 unpool_size_sqrt:int=1,
                 attention_pos:int=-1,
                 n_rbs:int=3,
                 single_conv:bool=False):
        super().__init__()
        self.node_dim = 1

        # 1- Setting up upsampling
        self.unpool_func = unpool_func
        self.unpool_size_sqrt = unpool_size_sqrt
        self.unpool_size = unpool_size_sqrt * unpool_size_sqrt
        assert ((self.unpool_size == 1) and (unpool_func is None)) or ((self.unpool_size > 1) and (unpool_func is not None)), "unpool_func and unpool_size must match."
        if (unpool_func in [None, 'SDPAConvTranspose']) or (self.unpool_size == 1):
            self.register_parameter('unpool', None)
        elif unpool_func in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            self.unpool = getattr(torch.nn, "Upsample")(scale_factor=(self.unpool_size, 1), mode=unpool_func)
        elif unpool_func == "pixel_shuffle":
            self.unpool = SpherePixelShuffle(self.unpool_size_sqrt, self.node_dim)
            out_channels *= self.unpool_size
        else:
            raise ValueError('Unpooling is not defined')

        # 2- Setting convolution
        self.list_conv = torch.nn.ModuleList()
        num_conv = hop if conv_name in ["GraphConv", "SDPAConv"] else 1
        if single_conv: num_conv = 1
        if skip_conn_aggr == 'cat':
            out_channels //= num_conv

        if conv_name == "ChebConv":
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, K=hop+1, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['TAGConv', 'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, K=hop, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, aggr='mean', bias=bias, node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels, aggr='mean', bias=bias, node_dim=self.node_dim) for _ in range(num_conv - 1)]))  # Maybe later not all of them has aggr as argument
        elif conv_name == "SDPAConv" and unpool_func != "SDPAConvTranspose":
            conv = SDPAConv
            n_firstHopNeighbors = 8
            n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=1 if not single_conv else hop)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, kernel_size=n_neighbors+1, bias=bias, node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels, kernel_size=n_neighbors+1, bias=bias, node_dim=self.node_dim) for _ in range(num_conv - 1)]))
        elif unpool_func == "SDPAConvTranspose":
            conv = SDPAConvTranspose
            n_firstHopNeighbors = 8
            n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=1 if not single_conv else hop)
            # only upscale for first conv
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, kernel_size=n_neighbors+1, upscale_factor=self.unpool_size, node_dim=self.node_dim, bias=bias))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels, kernel_size=n_neighbors+1, upscale_factor=1, node_dim=self.node_dim, bias=bias) for _ in range(num_conv - 1)]))
        else:
            raise ValueError('Convolution is not defined')

        self.in_channels = in_channels
        assert len(self.list_conv) == num_conv, "list conv must be equal to num_conv"
        # Setting aggregation of convolution results
        self.out_channels = out_channels
        if (num_conv > 1) and (skip_conn_aggr not in ["non", "none"]):
            self.skipconn = SphereSkipConnection(skip_conn_aggr)
            if skip_conn_aggr == "cat":
                self.out_channels *= num_conv
        else:
            self.register_parameter('skipconn', None)
        self.conv_out_channels = self.out_channels
        if unpool_func == "pixel_shuffle":
            self.out_channels //= self.unpool_size
        
        # 3- Setting nonlinearity
        self.activation_name = activation
        if activation is None:
            self.activation_name = ''
            self.register_parameter('activation', None)
        elif activation.upper() == "GDN":
            self.activation = SphereGDN(self.out_channels, **activation_args)
        elif activation.upper() == "RB":
            self.activation = torch.nn.ModuleList([ResBlock(conv_name, self.out_channels, bias) for _ in range(n_rbs)])
        else:
            self.activation = getattr(torch.nn, activation)(**activation_args)
        
        # 4- Setting attention module
        assert attention_pos in [-1, 0, 1, 2], "attention_pos must be -1, 0, 1 or 2 for Decoder"
        self.attention_pos = attention_pos
        if attention_pos in [0, 1, 2]:
            self.attention_module = AttentionModule(conv_name, self.out_channels if attention_pos else in_channels, bias, activation='ReLU', n_rb_trunk=3, n_rb_mask=3)
        else:
            self.attention_module = None

    def forward(self, x, index, weight, valid_index=None, mapping=None, index_=None, weight_=None, valid_index_=None):  # x is a tensor of size [batch_size, num_nodes, num_features]
        'index_, weight_ and valid_index_ denote the respective tensors after convolution (if RB as nonlinearity or attention used)'
        if mapping is not None: raise NotImplementedError("Not implemented")
        device = x.device

        # Note for unpooling:
        # if unpooling is Upsample the order is: Upsample then Convolution
        # if unpooling is SpherePixelShuffle the order is: Convolution then SpherePixelShuffle
        if (self.unpool is not None) and (self.unpool.__class__.__name__ == "Upsample"):
            x = torch.squeeze(self.unpool(torch.unsqueeze(x, dim=0)), dim=0)

        index = index.to(device)
        weight = weight.to(device)
        valid_index = valid_index.to(device) if valid_index is not None else None
        # index_, weight_ and valid_index_ for attention module after convolution
        if index_ is not None: index_ = index_.to(device)
        if weight_ is not None: weight_ = weight_.to(device)
        if valid_index_ is not None: valid_index_ = valid_index_.to(device)
        
        if self.attention_pos == 0:
            x = self.attention_module(x, index, weight, valid_index)
        
        xs = []
        for conv in self.list_conv:
            if conv.__class__.__name__ in "SDPAConv":
                x = conv(x, neighbors_indices=index, neighbors_weights=weight, valid_index=valid_index)
            elif self.unpool_func == "SDPAConvTranspose":
                x = conv(x, index_, weight_, valid_index_)
            else:
                x = conv(x, edge_index=index, edge_weight=weight)
            xs += [x]
        x = self.skipconn(xs) if self.skipconn is not None else xs[-1]

        if (self.unpool is not None) and (self.unpool.__class__.__name__ == "SpherePixelShuffle"):
            x = self.unpool(x)

        if self.attention_pos == 2:
            x = self.attention_module(x, index_, weight_, valid_index_)
        
        if self.activation is not None:
            if self.activation_name.upper() == "RB":
                for conv in self.activation:
                    x = conv(x, index_, weight_, valid_index_)
            else:
                x = self.activation(x)

        if self.attention_pos == 1:
            x = self.attention_module(x, index_, weight_, valid_index_)
        
        return x

    def get_conv_input_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct data for the "input of the conv".
        For example, if we use Upsampling, since first the upsampling is applied and then convolution, for unpool_size_sqrt=2
        it returns 1 because conv is applied on upsampled data.
        For pixel shuffling, since pixel shuffling is applied after convolution, the function return 0 no matter of unpool_size_sqrt

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if self.unpool is None:
            return 0
        # There is an unpooling
        if self.unpool.__class__.__name__ == "SpherePixelShuffle":
            return 0
        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionUpsampled(0, self.unpool_size_sqrt)

    def get_output_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct data for the "output of the module".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if (self.unpool is None) and (self.unpool_func != "SDPAConvTranspose"):
            return 0
        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionUpsampled(0, self.unpool_size_sqrt)


class AttentionModule(torch.nn.Module):
    r"""Attention Module introduced in Cheng et al. 2020: "Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules" (https://ieeexplore.ieee.org/document/9156817)
    
    Args:
        conv_name (str): convolution operator
        in_channels (int): number of input channels of the previous layer
        bias (bool): True if bias should be included. Default is True
        activation (str): activation function after every convolution except the last. Default is 'ReLU'
        n_rb_trunk (int): number of residual blocks in the trunk branch (minimum 1)
        n_rb_mask (int): number of residual blocks in the mask branch (minimum 1)
    """
    def __init__(self,
                 conv_name:str,
                 in_channels:int,
                 bias:bool=True,
                 activation:str='ReLU',
                 n_rb_trunk:int=3,
                 n_rb_mask:int=3):
        super().__init__()
        self.conv_name = conv_name
        self.in_channels = in_channels
        self.bias = bias
        self.activation = activation
        self.rb = ResBlock(conv_name, in_channels, bias, activation=activation)
        self.n_rb_trunk = n_rb_trunk
        self.n_rb_mask = n_rb_mask
        
        # 1x1 convolution
        self.node_dim = 1
        if conv_name == "ChebConv":
            conv = getattr(torch_g_nn, conv_name)
            self.conv1 = conv(in_channels=in_channels, out_channels=in_channels, K=1, bias=bias, node_dim=self.node_dim)
        elif conv_name in ['TAGConv', 'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_g_nn, conv_name)
            self.conv1 = conv(in_channels=in_channels, out_channels=in_channels, K=1, bias=bias, node_dim=self.node_dim)
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_g_nn, conv_name)
            self.conv1 = conv(in_channels=in_channels, out_channels=in_channels, aggr='mean', bias=bias, node_dim=self.node_dim)
        elif conv_name == "SDPAConv":
            conv = SDPAConv
            self.conv1 = conv(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=bias, node_dim=self.node_dim)
        else:
            raise ValueError('Convolution is not defined')
        # residual blocks
        self.rbs_mask = torch.nn.ModuleList([ResBlock(conv_name, in_channels, bias, activation) for _ in range(n_rb_mask)])
        self.rbs_trunk = torch.nn.ModuleList([ResBlock(conv_name, in_channels, bias, activation) for _ in range(n_rb_trunk)])
        
    def forward(self, x, index, weight, valid_index=None, mapping=None):
        device = x.device
        index = index.to(device)
        weight = weight.to(device)
        valid_index = valid_index.to(device) if valid_index is not None else None
        # additional inputs to convolution forward function
        conv_kwargs = {
            'neighbors_indices': index,
            'neighbors_weights': weight,
            'valid_index': valid_index
        } if self.conv_name=='SDPAConv' else {
            'edge_index': index,
            'edge_weight': weight
        }
        # mask branch
        for i, conv in enumerate(self.rbs_mask):
            x_mask = conv(x if i==0 else x_mask, index, weight, valid_index, mapping)
        
        x_mask = self.conv1(x, **conv_kwargs)
        x_mask = torch.sigmoid(x_mask)
        
        # trunk branch
        for i, conv in enumerate(self.rbs_trunk):
            x_trunk = conv(x if i==0 else x_trunk, index, weight, valid_index, mapping)
        
        x = x + torch.multiply(x_mask, x_trunk)
        
        if mapping is not None:
            mapping = mapping.to(device)
            x = x.index_select(self.node_dim, mapping)
        return x

class ResBlock(torch.nn.Module):
    r"""Residual Block consisting of:
    input: feature map with N channels
     convk1s1 (N/2) + ReLU (or other) +
     convk3s1 (N/2) + ReLU (or other) +
     convk1s1 (N) + 
     skip connection (addition)
    """
    def __init__(self,
                 conv_name:str,
                 in_channels:int,
                 bias:bool=True,
                 activation='ReLU',
                 activation_args:dict=dict()):
        super().__init__()
        self.conv_name = conv_name
        self.in_channels = in_channels
        self.bias = bias
        
        hop = 1
        self.node_dim = 1
        self.list_conv = torch.nn.ModuleList()
        if conv_name == "ChebConv":
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv = torch.nn.ModuleList([
                conv(in_channels, in_channels//2, K=1, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels//2, K=hop+1, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels, K=1, bias=bias, node_dim=self.node_dim),])
        elif conv_name in ['TAGConv', 'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv = torch.nn.ModuleList([
                conv(in_channels, in_channels//2, K=1, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels//2, K=hop, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels, K=1, bias=bias, node_dim=self.node_dim),])
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_g_nn, conv_name)
            self.list_conv = torch.nn.ModuleList([
                conv(in_channels, in_channels//2, aggr='mean', bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels//2, aggr='mean', bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels, aggr='mean', bias=bias, node_dim=self.node_dim),])
        elif conv_name == "SDPAConv":
            conv = SDPAConv
            self.list_conv = torch.nn.ModuleList([
                conv(in_channels, in_channels//2, kernel_size=1, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels//2, kernel_size=9, bias=bias, node_dim=self.node_dim),
                conv(in_channels//2, in_channels, kernel_size=1, bias=bias, node_dim=self.node_dim),])
        else:
            raise ValueError('Convolution is not defined')
        
        if activation is None:
            self.register_parameter('activation', None)
        elif activation in ["GDN"]:
            self.activation = SphereGDN(in_channels//2, **activation_args)
        else:
            self.activation = getattr(torch.nn, activation)(**activation_args)

    def forward(self, x, index, weight, valid_index=None, mapping=None):
        device = x.device
        index = index.to(device)
        weight = weight.to(device)
        valid_index = valid_index.to(device) if valid_index is not None else None
        # additional inputs to convolution forward function
        conv_kwargs = {
            'neighbors_indices': index,
            'neighbors_weights': weight,
            'valid_index': valid_index
        } if self.conv_name=='SDPAConv' else {
            'edge_index': index,
            'edge_weight': weight
        }
        for i, conv in enumerate(self.list_conv):
            x_out = conv(x if i==0 else x_out, **conv_kwargs)
            if (self.activation is not None) and (i < (len(self.list_conv)-1)):
                x_out = self.activation(x_out)
        
        x_out += x
        
        if mapping is not None:
            mapping = mapping.to(device)
            x_out = x_out.index_select(self.node_dim, mapping)
        return x_out



if __name__ == '__main__':
    import healpy as hp
    import healpix_graph_loader
    import healpix_sdpa_struct_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    resolution = 2
    patch_resolution = 2
    patch_id = 5
    nside = hp.order2nside(resolution)  # == 2 ** sampling_resolution
    nPix = hp.nside2npix(nside)
    use_geodesic = True

    folder = "../GraphData"
    cutGraphForPatchOutside = True
    weight_type = "gaussian"
    K = 1  # Number of hops

    conv_name = "SDPAConv"  # SDPAConv, 'ChebConv', 'TAGConv', 'SGConv', GraphConv
    unpool_func = "nearest" # 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', pixel_shuffle
    scale_factor = 2

    if conv_name=="SDPAConv":
        loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type=weight_type,
                                                                     use_geodesic=use_geodesic,
                                                                     use_4connectivity=False,
                                                                     normalization_method="sym",
                                                                     cutGraphForPatchOutside=cutGraphForPatchOutside,
                                                                     load_save_folder=folder)
        struct_data = loader.getStruct(resolution, K, patch_resolution, patch_id)
        # struct_sdpa = sdpa_loader.getStruct(resolution, K)
        index_downsample = struct_data[0]
        weight_downsample = struct_data[1]
        nodes = struct_data[3]
        if unpool_func=="pixel_shuffle":
            index_upsample = index_downsample
            weight_upsample = weight_downsample
        else:
            struct_data = loader.getStruct(hp_utils.healpix_getResolutionUpsampled(resolution, scale_factor), K,
                                           hp_utils.healpix_getResolutionUpsampled(patch_resolution, scale_factor), patch_id)
            # struct_graph = graph_loader.getGraph(sampling_res=resolution)
            index_upsample = struct_data[0]
            weight_upsample = struct_data[1]
    else:
        loader = healpix_graph_loader.HealpixGraphLoader(weight_type=weight_type,
                                                           use_geodesic=use_geodesic,
                                                           use_4connectivity=False,
                                                           load_save_folder=folder)

        n_hop_graph = 0 if cutGraphForPatchOutside else K
        struct_data = loader.getGraph(sampling_res=resolution, patch_res=patch_resolution, num_hops=n_hop_graph, patch_id=patch_id)
        # struct_graph = graph_loader.getGraph(sampling_res=resolution)
        index_downsample = struct_data[0]
        weight_downsample = struct_data[1]
        nodes = struct_data[2]
        if unpool_func=="pixel_shuffle":
            index_upsample = index_downsample
            weight_upsample = weight_downsample
        else:
            struct_data = loader.getGraph(sampling_res=hp_utils.healpix_getResolutionUpsampled(resolution, scale_factor),
                                          patch_res=hp_utils.healpix_getResolutionUpsampled(patch_resolution, scale_factor),
                                          num_hops=n_hop_graph, patch_id=patch_id)
            # struct_graph = graph_loader.getGraph(sampling_res=resolution)
            index_upsample = struct_data[0]
            weight_upsample = struct_data[1]


    B = 4  # batch size
    in_channels = 2
    out_channels = 10
    data_th = torch.randn(B, nPix, in_channels)
    data_th = data_th.index_select(dim=1, index=nodes)

    print("data_th.size()=", data_th.size())

    slb_down = SLB_Downsample(conv_name, in_channels, out_channels,
                              bias=True, hop=2,
                              skip_conn_aggr="sum",
                              activation="GDN",
                              pool_func="max_pool", pool_size_sqrt=scale_factor
                              )

    print(slb_down)
    out_down = slb_down(data_th, index_downsample, weight_downsample)
    print("out_down.size()=", out_down.size())

    # TODO: Check the same for SLB_Upsample
    slb_up = SLB_Upsample(conv_name, in_channels, out_channels,
                          bias=True, hop=2,
                          skip_conn_aggr="sum",
                          activation="GDN", activation_args={"inverse":True},
                          unpool_func=unpool_func, unpool_size_sqrt=scale_factor
                          )


    print(slb_up)
    out_up = slb_up(data_th, index_upsample, weight_upsample)
    print("out_up.size()=", out_up.size())

