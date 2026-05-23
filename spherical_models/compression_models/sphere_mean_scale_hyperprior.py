import warnings
import torch
from spherical_models.compression_models import SphereScaleHyperprior
from compressai.ans import BufferedRansEncoder, RansDecoder
from spherical_models import SLB_Downsample, SLB_Upsample
from spherical_models.sdpa_conv import sdpaconv_node_n
import numpy as np
# pyright: reportGeneralTypeIssues=warning


class SphereMeanScaleHyperprior(SphereScaleHyperprior):
    r"""Scale Hyperprior model with non zero-mean Gaussian conditionals for spherical data.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the encoder)
    """
    def __init__(self, N, M, conv_name, skip_conn_aggr, pool_func, unpool_func, attention:bool=False, activation:str='GDN', single_conv:bool=False, context_model:str='', arhops:int=2, **kwargs):
        super().__init__(N, M, conv_name, skip_conn_aggr, pool_func, unpool_func, attention, activation, single_conv, context_model, arhops, **kwargs)
        ####################### h_s #######################
        self.h_s = torch.nn.ModuleList()
        self.h_s.append(SLB_Upsample(conv_name, N, M, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        self.h_s.append(SLB_Upsample(conv_name, M, M*3//2, hop=2, single_conv=single_conv, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # effective hop=1 => num_conv = 1, and there is no Upsampling
        self.h_s.append(SLB_Upsample(conv_name, M*3//2, M*2, hop=1, single_conv=single_conv, skip_conn_aggr=None,
                                     activation=None))
        ################## Context Model ##################
        if context_model:
            mask = 'full' if context_model=='full' else 'A'
            self.autoregressive = SLB_Downsample(conv_name, M, M*2, hop=arhops, single_conv=True, skip_conn_aggr=skip_conn_aggr,
                                    activation=None, mask=mask)
            self.combine_ar_hp = torch.nn.Sequential(
                SLB_Downsample('SDPAConv', M*12//3, M*10//3, hop=1, skip_conn_aggr=skip_conn_aggr,
                               activation="ReLU", activation_args={"inplace": True}, conv1x1=True),
                SLB_Downsample('SDPAConv', M*10//3, M*8//3, hop=1, skip_conn_aggr=skip_conn_aggr,
                               activation="ReLU", activation_args={"inplace": True}, conv1x1=True),
                SLB_Downsample('SDPAConv', M*8//3, M*6//3, hop=1, skip_conn_aggr=skip_conn_aggr, conv1x1=True)
            )

    def forward(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):     # x is a tensor of size [batch_size, num_nodes, num_features]
        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            if (self.activation.upper()=='RB') or (self.attention and (self.g_a[i].attention_pos > 0)): # for attention after convolution or RBs as nonlinearity
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
        z = y
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        # print("applying h_a")
        # print("z.mean()=", z.mean(), "torch.abs(y).mean()=", torch.abs(y).mean())
        # print("z.max()=", z.max(), "z.min()=", z.min())
        # print("torch.abs(y).max()=", torch.abs(y).max(), "torch.abs(y).min()=", torch.abs(y).min())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        ########### apply h_s ###########
        params = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._g_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            params = self.h_s[i](params, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        # print("applying h_s")
        # print("params.mean()=", params.mean(), means_hat.mean()=", means_hat.mean(), "z_hat.mean()=", z_hat.mean())
        # print("params.max()=", params.max(), "params.min()=", params.min())
        # print("means_hat.max()=", means_hat.max(), "means_hat.min()=", means_hat.min())
        # print("z_hat.max()=", z_hat.max(), "z_hat.min()=", z_hat.min())
        ###### apply context model ######
        if self.context_model:
            y_hat = self.gaussian_conditional.quantize(y, 'noise' if self.gaussian_conditional.training else 'dequantize')
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            phi_sp = self.autoregressive(y_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
            gaussian_params = self.combine_ar_hp(torch.cat([params, phi_sp], dim=-1))
            scales_hat, means_hat = gaussian_params.chunk(2, -1)
            # print("applying context model")
            # print("scales_hat.mean()=", scales_hat.mean(), "y_hat.mean()=", y_hat.mean())
            # print("scales_hat.max()=", scales_hat.max(), "scales_hat.min()=", scales_hat.min())
            # print("y_hat.max()=", y_hat.max(), "y_hat.min()=", y_hat.min())
        else:
            scales_hat, means_hat = params.chunk(2, -1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
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
        z = y
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[1])
        ########### apply h_s ###########
        params = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._g_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            params = self.h_s[i](params, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        if not self.context_model:
            scales_hat, means_hat = params.chunk(2, -1)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)    
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        else:
            y_strings = []
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            for i in range(y.size(0)):
                string = self._compress_ar(
                    y[i:i+1],
                    params[i:i+1],
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
        for n in range(n_nodes):
            ctx_p = sdpaconv_node_n(
                y_hat,
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
            gaussian_params = self.combine_ar_hp(torch.cat((p, ctx_p), dim=-1))
            gaussian_params = gaussian_params.squeeze(1)
            scales_hat, means_hat = gaussian_params.chunk(2, -1)

            indexes = self.gaussian_conditional.build_indexes(scales_hat)

            y_q = self.gaussian_conditional.quantize(y_hat[:, n, :], "symbols", means_hat)
            y_hat[:, n, :] = y_q + means_hat

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
        params = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            if self.h_s[i].unpool_func == "SDPAConvTranspose":
                conv_res_ = type(data_res)(np.add(data_res, self._h_s_offset[i+1] if i<(len(self.h_s)-1) else self._h_a_offset[0]))
                index_, weight_, valid_index_ = dict_index[conv_res_], dict_weight[conv_res_], dict_valid_index[conv_res_] if dict_valid_index is not None else None
            else:
                index_, weight_, valid_index_ = None, None, None
            params = self.h_s[i](params, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None,
                index_=index_, weight_=weight_, valid_index_=valid_index_)
        if self.context_model:
            s = 4**2 # scaling factor between z and y # TODO calc from offsets
            y_hat = torch.zeros(z_hat.size(0), s*shape, self.M, dtype=z_hat.dtype, device=z_hat.device)
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[0]))
            for i, y_string in enumerate(strings[0]):
                self._decompress_ar(
                    y_string,
                    y_hat[i:i+1],
                    params[i:i+1],
                    dict_index[conv_res],
                    dict_weight[conv_res],
                    self.autoregressive.skipconn,
                )
        else:
            scales_hat, means_hat = params.chunk(2, -1)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)

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
                y_hat,
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
            gaussian_params = self.combine_ar_hp(torch.cat((p, ctx_p), dim=-1))
            scales_hat, means_hat = gaussian_params.chunk(2, -1)

            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            rv = decoder.decode_stream(
                indexes.squeeze().tolist(),
                cdf,
                cdf_lengths,
                offsets,
            )
            rv = torch.Tensor(rv).reshape(1, 1, -1)
            rv = self.gaussian_conditional.dequantize(rv, means_hat)
            y_hat[:, n:n+1, :] = rv