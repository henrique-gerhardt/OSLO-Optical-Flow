import torch
import math
try:
    from utils import common_function as util_common
except ImportError:
    util_common = None
try:
    import torch_geometric  # noqa: F401
except ImportError:
    torch_geometric = None
# pyright: reportOptionalSubscript=warning

class SDPAConv (torch.nn.Module):
    r"""Class for implementing Sphere Directional and Position-Aware convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, node_dim=0, bias=True, mask:str='full'):
        if mask not in ['A', 'B', 'full']: raise NotImplementedError("masked convolution either 'A', 'B' or 'full'")
        super(SDPAConv, self).__init__()

        assert node_dim >= 0
        self.node_dim = node_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        self.mask = mask
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Took it from torch.nn.Conv2d()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.weight, gain=2.)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        # Took it from torch_geometric.nn.ChebConv
        # torch_geometric.nn.inits.glorot(self.weight)
        # torch_geometric.nn.inits.zeros(self.bias)

    def forward(self, x, neighbors_indices=None, neighbors_weights=None, valid_index=None):
        # in case of 1x1 convolution, the neighbors_indices and neighbors_weights are not needed
        assert (self.kernel_size==1) or ((self.kernel_size-1)<=neighbors_weights.size(1)), "size does not match"
        if self.kernel_size > 1:
            assert (neighbors_indices is not None) and (neighbors_weights is not None), "neighbors_indices and _weights must be provided for 2-hop convolution"
            neighbors_indices = neighbors_indices[:, :self.kernel_size-1]
            neighbors_weights = neighbors_weights[:, :self.kernel_size-1]
            if valid_index is not None: valid_index = valid_index[:, :self.kernel_size-1]
        
        if self.mask in ['A', 'B']: # set future neighbors to zero
            neighbors_weights = torch.mul(neighbors_weights, (neighbors_indices < torch.arange(x.size(1), device=neighbors_weights.device).view(-1, 1))) # type: ignore
        if self.mask != 'A': # current node included in convolution
            out = torch.matmul(x, self.weight[0])
        else:
            out = torch.zeros(x.size(0), x.size(1), self.out_channels, dtype=x.dtype, device=x.device)
        
        # test_out = torch.zeros(x.size(), dtype=x.dtype)
        # for k in range(neighbors_weights.size(1)):
        #     test_out += torch.mul(neighbors_weights.narrow(dim=1, start=k, length=1), x.index_select(self.node_dim, neighbors_indices[:, k]))
        # print("test_out finished")
        
        for k in range(1, self.kernel_size):
            col = k-1
            if valid_index is None:
                s = torch.mul(neighbors_weights.narrow(dim=1, start=col, length=1), x.index_select(self.node_dim, neighbors_indices[:, col]))   # or I could use neighbors_weights[:,col].view(-1, 1)
                out += torch.matmul(s, self.weight[k])
            else:
                valid_rows = valid_index[:, col]
                s = torch.mul(neighbors_weights[valid_rows, col].view(-1, 1), x.index_select(self.node_dim, neighbors_indices[valid_rows, col]))
                out[:, valid_rows, :] += torch.matmul(s, self.weight[k])

        if self.bias is not None:
            out += self.bias

        return out


class SDPAConvTranspose (SDPAConv):
    r"""Class for implementing Sphere Directional and Position-Aware transposed convolution
    """
    def __init__(self,  in_channels, out_channels, kernel_size, upscale_factor, node_dim=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, node_dim, bias)
        self.upscale_factor = upscale_factor
        self.weight = torch.nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        self.reset_parameters()

    def forward(self, x, neighbors_indices, neighbors_weights, valid_index=None):
        # in case of 1x1 convolution, the neighbors_indices and neighbors_weights are not needed
        assert (self.kernel_size-1)<=neighbors_weights.size(1) or self.kernel_size==1, "size does not match"
        assert x.size(self.node_dim)*self.upscale_factor==neighbors_weights.size(0), "neighbor structure should match output resolution"
        if self.kernel_size > 1:
            neighbors_indices = neighbors_indices[:, :self.kernel_size-1]
            neighbors_weights = neighbors_weights[:, :self.kernel_size-1]
            if valid_index is not None: valid_index = valid_index[:, :self.kernel_size-1]
        
        out = torch.zeros(x.size(0), x.size(self.node_dim)*self.upscale_factor, self.out_channels, dtype=x.dtype, device=x.device)
        out[:,::self.upscale_factor,:] += torch.matmul(x, self.weight[0])
        
        for k in range(1, self.kernel_size):
            col = k-1
            # neighbor structure corresponding to input resolution
            nb_indices_in = neighbors_indices[::self.upscale_factor,col] # output indices at input resolution pixels
            nb_weights_in = neighbors_weights[::self.upscale_factor,col]
            if valid_index is None:
                s = torch.mul(nb_weights_in[:,None], x)
                # invalid neighbors will be multiplied by 0 and added to 0th pixel
                out.index_add_(1, nb_indices_in, torch.matmul(s, self.weight[k]))
            else:
                valid_rows = valid_index[::self.upscale_factor,col]
                s = torch.mul(nb_weights_in[valid_rows,None], x.index_select(self.node_dim, torch.arange(x.size(self.node_dim))[valid_rows]))
                out.index_add_(1, nb_indices_in[valid_rows], torch.matmul(s, self.weight[k]))
        if self.bias is not None:
            out += self.bias

        return out


def sdpaconv(x, weights:list, biases=None, skip_conn_aggr:str='sum', mask:str='full', neighbors_indices=None, neighbors_weights=None, valid_index=None):
    r"""SDPA convolution

    Args:
        x (tensor): input tensor
        weights (list|tuple): weight tensors in list or tuple, len(weights) = n_hops
        biases (list|None): bias tensors, len(biases) = n_hops. Default None (no bias)
        skip_conn_aggr (str): skip connection aggregation method from ['', 'cat', 'max', 'sum'].
            Default 'sum'.
        mask (str): mask type from ['A', 'B', 'full']. Default 'full' (no masked convolution).
            'A': masked convolution with center excluded,
            'B': masked convolution with center included,
            'full': no masked convolution.
        neighbors_indices (tensor|None): neighbor indices. Not needed for 1x1 convolution.
            Default None.
        neighbors_weights (tensor|None): neighbor weights. Not needed for 1x1 convolution.
            Default None.
        valid_index (tensor|None): bool tensor of valid neighbors. Not needed for 1x1 convolution.
            Default None.
    """
    node_dim = 1
    kernel_size, num_conv = len(weights[0]), len(weights)
    # in case of 1x1 convolution, the neighbors_indices and neighbors_weights are not needed
    assert (kernel_size==1) or ((kernel_size-1)==neighbors_weights.size(1)), "size does not match"
    if biases is not None: assert len(biases)==len(weights)
    if mask in ['A', 'B']:
        assert (neighbors_indices is not None) and (neighbors_weights is not None), "neighbors_indices and _weights must be provided for masked convolution"
        neighbors_weights = torch.mul(neighbors_weights, (neighbors_indices < torch.arange(x.size(1), device=neighbors_weights.device).view(-1, 1)))
    if neighbors_indices is not None: neighbors_indices = neighbors_indices.to(x.device)
    if neighbors_weights is not None: neighbors_weights = neighbors_weights.to(x.device)
    if valid_index is not None: valid_index = valid_index.to(x.device)
    
    xs = []
    for i in range(num_conv):
        if mask != 'A':
            out = torch.matmul(x, weights[i][0])
        else:
            out = torch.zeros(x.size(0), x.size(1), weights[i].size(-1), dtype=x.dtype, device=x.device)

        for k in range(1, kernel_size):
            col = k-1
            if valid_index is None:
                s = torch.mul(neighbors_weights.narrow(dim=1, start=col, length=1), x.index_select(node_dim, neighbors_indices[:, col]))
                out += torch.matmul(s, weights[i][k])
            else:
                valid_rows = valid_index[:, col]
                s = torch.mul(neighbors_weights[valid_rows, col].view(-1, 1), x.index_select(node_dim, neighbors_indices[valid_rows, col]))
                out[:, valid_rows, :] += torch.matmul(s, weights[i][k])

        if biases is not None: out += biases[i]
        xs.append(out)
        
    out = _sphere_skip_connection(xs, skip_conn_aggr) if num_conv and skip_conn_aggr else xs[-1]
    return out

def sdpaconv_node_n(x, hops:int, weights:list, bias:list, n, neighbors_indices, neighbors_weights, single_conv:bool, skipconn=None, masked=False):
    'calculate SDPA convolution only at node n'
    if util_common is None:
        raise ImportError("utils.common_function is required for sdpaconv_node_n")
    if single_conv:
        n_conv = 1
    else:
        if hops not in [1, 2]: raise NotImplementedError("Number of hops must be 1 or 2 for sequential convolution")
        n_conv = hops
    assert len(weights)==n_conv, "number of weights must match number of convolutions"
    if bias is not None: assert len(bias)==n_conv, "number of biases must match number of convolutions"
    n_firstHopNeighbors = 8
    n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=hops if single_conv else 1)
    assert n_neighbors <= neighbors_indices.size(1), "neighborhood to small"
    neighbors_indices = neighbors_indices[:, :n_neighbors]
    neighbors_weights = neighbors_weights[:, :n_neighbors]
    
    if masked:
        neighbors_weights_in = torch.mul(neighbors_weights, (neighbors_indices < n))
    else:
        neighbors_weights_in = neighbors_weights
    # buffer for SDPA conv, should only be calculated at current node n
    if n_conv == 1:
        xs = [torch.zeros(x.size(0), 1, weights[0].size(-1), dtype=x.dtype, device=x.device)] # shape: [(batch, 1, out_channels)]
    else:
        xs = [torch.zeros(x.size(0), n_neighbors+1 if i==0 else 1, weights[i].size(-1), dtype=x.dtype, device=x.device) for i in range(len(weights))] # shape: [(batch, n_neighbors+1, out_channels), (batch, 1, out_channels)]
    
    calc_indices = [n]
    if n_conv==2: calc_indices += neighbors_indices[n].tolist()
    for k, k_ind in enumerate(calc_indices): # iteration over current node and its neighbors
        cur_neighbors = torch.mul(neighbors_weights_in[k_ind].view(1,-1,1), x[:, neighbors_indices[k_ind].tolist(), :])
        xs[0][:, k, :] = torch.matmul(cur_neighbors.flatten(1,2), weights[0].data[1:].flatten(0,1))
        if not (masked and (k_ind >= n)):
            xs[0][:, k, :] += torch.matmul(x[:, k_ind, :], weights[0].data[0])
    if bias[0] is not None:
        xs[0] += bias[0].data
    
    if n_conv == 2: # compute second convolution at current node
        xs[1][:, 0, :] = torch.matmul(xs[0][:, 0, :], weights[1].data[0])
        s = torch.mul(neighbors_weights[n].view(1,-1,1), xs[0][:, 1:, :])
        xs[1][:, 0, :] += torch.matmul(s.flatten(1,2), weights[1].data[1:].flatten(0,1))
        if bias[1] is not None:
            xs[1] += bias[1].data
        # only keep current node
        xs[0] = xs[0][:, 0:1, :]
    
    out = skipconn(xs) if (n_conv==2) and (skipconn is not None) else xs[-1]
    return out

def _sphere_skip_connection(xs, mode:str='sum'):
        r"""Aggregates representations across different layers.

        Args:
            xs (list or tuple): List containing layer-wise representations.
            mode (str): Aggregation mode. Can be one of the following: cat, max, sum
        """
        assert isinstance(xs, list) or isinstance(xs, tuple)
        assert mode in ['cat', 'max', 'sum']

        if mode == 'cat': return torch.cat(xs, dim=-1)
        elif mode == 'max': return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif mode == 'sum': return torch.stack(xs, dim=-1).sum(dim=-1)
