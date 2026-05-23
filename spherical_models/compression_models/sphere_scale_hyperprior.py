import warnings
import math
import torch
from spherical_models.compression_models import SphereCompressionModel
from spherical_models import SphereGaussianConditional
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from spherical_models import SLB_Downsample, SLB_Upsample
from spherical_models.sdpa_conv import sdpaconv_node_n
import numpy as np
# pyright: reportGeneralTypeIssues=warning
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SphereScaleHyperprior(SphereCompressionModel):
    r"""Scale Hyperprior model for spherical data.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    def __init__(self, N, M, conv_name, skip_conn_aggr, pool_func, unpool_func, attention:bool=False, activation:str='GDN', single_conv:bool=False, context_model:str='', arhops:int=1, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.attention = attention
        self.arhops = arhops
        a_pos = 1 if attention else -1
        self.activation = activation
        ####################### g_a #######################
        self.g_a = torch.nn.ModuleList()
        self.g_a.append(SLB_Downsample(conv_name, 3, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                        activation=activation,
                                        pool_func=pool_func, pool_size_sqrt=2))
        self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                       activation=activation,
                                       pool_func=pool_func, pool_size_sqrt=2, attention_pos=a_pos))
        self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                       activation=activation,
                                       pool_func=pool_func, pool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_a.append(SLB_Downsample(conv_name, N, M, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                       pool_func=pool_func, pool_size_sqrt=2, attention_pos=a_pos))
        ####################### g_s #######################
        self.g_s = torch.nn.ModuleList()
        a_pos = (0 if attention else -1)
        self.g_s.append(SLB_Upsample(conv_name, M, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation=activation, activation_args={"inverse":True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2, attention_pos=a_pos))
        a_pos = (2 if attention else -1)
        self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation=activation, activation_args={"inverse": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2, attention_pos=a_pos))
        self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation=activation, activation_args={"inverse": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_s.append(SLB_Upsample(conv_name, N, 3, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        ####################### h_a #######################
        self.h_a = torch.nn.ModuleList()
        # effective hop=1 => num_conv = 1, and there is no downsampling
        self.h_a.append(SLB_Downsample(conv_name, M, N, hop=1, single_conv=single_conv, skip_conn_aggr=None,
                                       activation="ReLU", activation_args={"inplace": True}))
        self.h_a.append(SLB_Downsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                       activation="ReLU", activation_args={"inplace": True},
                                       pool_func=pool_func, pool_size_sqrt=2))
        # For the last layer there is no ReLu anymore:
        self.h_a.append(SLB_Downsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                       pool_func=pool_func, pool_size_sqrt=2))
        ####################### h_s #######################
        self.h_s = torch.nn.ModuleList()
        self.h_s.append(SLB_Upsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        self.h_s.append(SLB_Upsample(conv_name, N, N, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # effective hop=1 => num_conv = 1, and there is no Upsampling
        self.h_s.append(SLB_Upsample(conv_name, N, M, hop=1, single_conv=single_conv, skip_conn_aggr=None,
                                     activation="ReLU", activation_args={"inplace": True}))
        ################## Context Model ##################
        self.context_model = context_model
        if context_model:
            mask = 'full' if context_model=='full' else 'A'
            self.autoregressive = SLB_Downsample(conv_name, M, M, hop=arhops, single_conv=True, skip_conn_aggr=skip_conn_aggr,
                                    activation=None, mask=mask)
            self.combine_ar_hp = torch.nn.Sequential(
                SLB_Downsample('SDPAConv', 2*M, M+256, hop=1, skip_conn_aggr=skip_conn_aggr,
                               activation="ReLU", activation_args={"inplace": True}, conv1x1=True),
                SLB_Downsample('SDPAConv', M+256, M+128, hop=1, skip_conn_aggr=skip_conn_aggr,
                               activation="ReLU", activation_args={"inplace": True}, conv1x1=True),
                SLB_Downsample('SDPAConv', M+128, M, hop=1, skip_conn_aggr=skip_conn_aggr, conv1x1=True)
            )
        ###################################################
        self.gaussian_conditional = SphereGaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self._computeResOffset()

    def _computeResOffset(self):
        # compute convolution resolution offset
        g_a_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_a]))
        self._g_a_offset = [self.g_a[0].get_conv_input_res_offset()]
        self._g_a_offset.extend([self.g_a[i].get_conv_input_res_offset() + g_a_output[i-1] for i in range(1, len(self.g_a))])

        h_a_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.h_a]))
        h_a_output = [res+g_a_output[-1] for res in h_a_output]
        self._h_a_offset = [self.h_a[0].get_conv_input_res_offset() + g_a_output[-1]]
        self._h_a_offset.extend([self.h_a[i].get_conv_input_res_offset() + h_a_output[i-1] for i in range(1, len(self.h_a))])

        h_s_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.h_s]))
        h_s_output = [res+h_a_output[-1] for res in h_s_output]
        self._h_s_offset = [self.h_s[0].get_conv_input_res_offset() + h_a_output[-1]]
        self._h_s_offset.extend([self.h_s[i].get_conv_input_res_offset() + h_s_output[i-1] for i in range(1, len(self.h_s))])

        assert h_s_output[-1] == g_a_output[-1], "resolutions do not match"

        g_s_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_s]))
        g_s_output = [res+g_a_output[-1] for res in g_s_output]
        self._g_s_offset = [self.g_s[0].get_conv_input_res_offset() + g_a_output[-1]]
        self._g_s_offset.extend([self.g_s[i].get_conv_input_res_offset() + g_s_output[i-1] for i in range(1, len(self.g_s))])

    def get_resOffset(self):
        return set(self._g_a_offset + self._h_a_offset + self._h_s_offset + self._g_s_offset)

    def forward(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):     # x is a tensor of size [batch_size, num_nodes, num_features]
        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            if (self.activation.upper()=='RB') or (self.attention and (self.g_a[i].attention_pos > 0)): # for attention/RB module after convolution or transposed conv
                conv_res_ = type(data_res)(np.add(data_res, self._g_a_offset[i+1] if i<(len(self.g_a)-1) else self._g_s_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        # print("applying g_a")
        # print("y.mean()=", y.mean(), "x.mean()=", x.mean())
        # print("y.max()=", y.max(), "y.min()=", y.min())
        # print("x.max()=", x.max(), "x.min()=", x.min())
        ########### apply h_a ###########
        z = torch.abs(y)
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        # print("applying h_a")
        # print("z.mean()=", z.mean(), "torch.abs(y).mean()=", torch.abs(y).mean())
        # print("z.max()=", z.max(), "z.min()=", z.min())
        # print("torch.abs(y).max()=", torch.abs(y).max(), "torch.abs(y).min()=", torch.abs(y).min())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._h_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        # print("applying h_s")
        # print("scales_hat.mean()=", scales_hat.mean(), "z_hat.mean()=", z_hat.mean())
        # print("scales_hat.max()=", scales_hat.max(), "scales_hat.min()=", scales_hat.min())
        # print("z_hat.max()=", z_hat.max(), "z_hat.min()=", z_hat.min())
        ###### apply context model ######
        skip_quant = bool(self.context_model)
        if skip_quant: # take quantization out of self.gaussian_conditional.forward()
            y_hat = self.gaussian_conditional.quantize(torch.abs(y), 'noise' if self.gaussian_conditional.training else 'dequantize', means=None)
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            phi_sp = self.autoregressive(y_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
            scales_hat = self.combine_ar_hp(torch.cat([scales_hat, phi_sp], dim=-1))
            # print("applying context model")
            # print("scales_hat.mean()=", scales_hat.mean(), "y_hat.mean()=", y_hat.mean())
            # print("scales_hat.max()=", scales_hat.max(), "scales_hat.min()=", scales_hat.min())
            # print("y_hat.max()=", y_hat.max(), "y_hat.min()=", y_hat.min())
        y_hat, y_likelihoods = self.gaussian_conditional(y if not skip_quant else y_hat, scales_hat, skip_quant=skip_quant)
        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            if (self.activation.upper()=='RB') or (self.attention and (self.g_s[i].attention_pos > 0)) or (self.g_s[i].unpool_func=="SDPAConvTranspose"):
                conv_res_ = type(data_res)(np.add(data_res, self._g_s_offset[i+1] if i<(len(self.g_s)-1) else self._g_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        # print("applying g_s")
        # print("x_hat.mean()=", x_hat.mean(), "y_hat.mean()=", y_hat.mean())
        # print("x_hat.max()=", x_hat.max(), "x_hat.min()=", x_hat.min())
        # print("y_hat.max()=", y_hat.max(), "y_hat.min()=", y_hat.min())
        # with torch.no_grad():
        #     print("input/out mean ratio=", x.mean()/x_hat.mean())

        return {
            'x_hat': x_hat,
            'likelihoods': {
                'y': y_likelihoods,
                'z': z_likelihoods
            },
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict)
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):
        if self.context_model:
            if next(self.parameters()).device != torch.device("cpu"):
                warnings.warn(
                    "Inference on GPU is not recommended for the autoregressive "
                    "models (the entropy coder is run sequentially on CPU).",
                    stacklevel=2)
        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            if (self.activation.upper()=='RB') or (self.attention and (self.g_a[i].attention_pos > 0)):
                conv_res_ = type(data_res)(np.add(data_res, self._g_a_offset[i+1] if i<(len(self.g_a)-1) else self._g_s_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        ########### apply h_a ###########
        z = torch.abs(y)
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[1])
        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._h_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        if not self.context_model:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes)
        else:
            y_strings = []
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            for i in range(y.size(0)):
                string = self._compress_ar(
                    y[i:i+1],
                    scales_hat[i:i+1],
                    dict_index[conv_res],
                    dict_weight[conv_res],
                    self.autoregressive.skipconn,
                )
                y_strings.append(string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[1]}
    
    def _compress_ar(self, y_hat, params, neighbors_indices, neighbors_weights, skipconn):
        assert len(self.autoregressive.list_conv) in [1,2], "only 1 or 2-hop convolution supported for autoregressive context model"
        neighbors_indices = neighbors_indices.to(y_hat.device)
        neighbors_weights = neighbors_weights.to(y_hat.device)
        
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.tolist()
        offsets = self.gaussian_conditional._offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        n_nodes = y_hat.size(1)
        weights = [conv.weight for conv in self.autoregressive.list_conv]
        bias = [conv.bias for conv in self.autoregressive.list_conv]
        # causal mask for convolution
        for n in range(n_nodes):
            ctx_p = sdpaconv_node_n(
                torch.abs(y_hat),
                self.arhops,
                weights,
                bias,
                n,
                neighbors_indices,
                neighbors_weights,
                single_conv=True,
                skipconn=skipconn,
                masked=True,
            )

            # 1x1 conv for the entropy parameters prediction network, so
            # we only keep the elements in the "center"
            p = params[:, n:n+1, :]
            scales_hat = self.combine_ar_hp(torch.cat((p, ctx_p), dim=-1))
            scales_hat = scales_hat.squeeze(1)

            indexes = self.gaussian_conditional.build_indexes(scales_hat)

            y_q = self.gaussian_conditional.quantize(y_hat[:, n, :], "symbols")
            y_hat[:, n, :] = y_q

            symbols_list.extend(y_q.squeeze().tolist())
            indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list,
            indexes_list,
            cdf,
            cdf_lengths,
            offsets,
        )
        string = encoder.flush()
        return string
    
    def decompress(self, strings, shape, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):
        assert isinstance(strings, list) and len(strings) == 2
        if self.context_model:
            if next(self.parameters()).device != torch.device("cpu"):
                warnings.warn(
                    "Inference on GPU is not recommended for the autoregressive "
                    "models (the entropy coder is run sequentially on CPU).",
                    stacklevel=2,
                )
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._h_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        if self.context_model:
            s = 4**2 # scaling factor between z and y
            y_hat = torch.zeros(z_hat.size(0), s*shape, self.M, dtype=z_hat.dtype, device=z_hat.device)
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            for i, y_string in enumerate(strings[0]):
                self._decompress_ar(
                    y_string,
                    y_hat[i:i+1],
                    scales_hat[i:i+1],
                    dict_index[conv_res],
                    dict_weight[conv_res],
                    self.autoregressive.skipconn,
                )
        else:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            if (self.activation.upper()=='RB') or (self.attention and (self.g_s[i].attention_pos > 0)) or (self.g_s[i].unpool_func=="SDPAConvTranspose"):
                conv_res_ = type(data_res)(np.add(data_res, self._g_s_offset[i+1] if i<(len(self.g_s)-1) else self._g_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        x_hat = x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def _decompress_ar(self, y_string, y_hat, params, neighbors_indices, neighbors_weights, skipconn):
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.tolist()
        offsets = self.gaussian_conditional._offset.tolist()

        neighbors_indices = neighbors_indices.to(y_hat.device)
        neighbors_weights = neighbors_weights.to(y_hat.device)
        
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        
        n_nodes = y_hat.size(1)
        weights = [conv.weight for conv in self.autoregressive.list_conv]
        bias = [conv.bias for conv in self.autoregressive.list_conv]
        # Warning: this is slow due to the auto-regressive nature of the decoding
        for n in range(n_nodes):
            ctx_p = sdpaconv_node_n(
                torch.abs(y_hat),
                self.arhops,
                weights,
                bias,
                n,
                neighbors_indices,
                neighbors_weights,
                single_conv=True,
                skipconn=skipconn,
                masked=False,
            )

            # 1x1 conv for the entropy parameters prediction network, so
            # we only keep the elements in the "center"
            p = params[:, n:n+1, :]
            scales_hat = self.combine_ar_hp(torch.cat((p, ctx_p), dim=-1))

            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            rv = decoder.decode_stream(
                indexes.squeeze().tolist(),
                cdf,
                cdf_lengths,
                offsets,
            )
            rv = torch.Tensor(rv).reshape(1, 1, -1)
            rv = self.gaussian_conditional.dequantize(rv)
            y_hat[:, n:n+1, :] = rv


if __name__ == '__main__':
    ssh = SphereScaleHyperprior(128, 192, "SDPAConv", "sum", "max_pool", "nearest")
    print(ssh.get_resOffset())
