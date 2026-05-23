import os
import torch
import numpy as np
import healpy as hp
from torch_geometric import utils as torch_g_utils
from utils import healpix as hp_utils
# pyright: reportGeneralTypeIssues=warning

class HealpixGraphLoader:
    def __init__(self, weight_type, use_geodesic, use_4connectivity, load_save_folder=None):
        self.weight_type = weight_type
        self.use_geodesic = use_geodesic
        self.use_4connectivity = use_4connectivity
        self.isNest = True
        self.folder = load_save_folder
        if self.folder:
            os.makedirs(self.folder, exist_ok=True)

    def getGraph(self, sampling_res, patch_res=None, num_hops=None, patch_id=-1) -> tuple:
        if self.folder:
            filename = f"{self.weight_type}_{self.use_geodesic}_{self.use_4connectivity}_{sampling_res}"
            if patch_res:
                filename = filename + f"_{patch_res}_{num_hops}_{patch_id}"
            filename += ".pth"
            file_address = os.path.join(self.folder, filename)
            if os.path.isfile(file_address):
                # print(f"Loading file {file_address}")
                data_dict = torch.load(file_address)
                index = data_dict.get("index", None)
                weight = data_dict.get("weight", None)
                if patch_res is None:
                    return index, weight
                nodes = data_dict.get("nodes", None)
                mapping = data_dict.get("mapping", None)
                return index, weight, nodes, mapping


        if patch_res is None:
            nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution
            nPix = hp.nside2npix(nside)
            pixel_id = np.arange(0, nPix, dtype=int)
            graph_structure, _ = hp_utils.healpix_weightmatrix(resolution=sampling_res,
                                                               weight_type=self.weight_type,
                                                               use_geodesic=self.use_geodesic,
                                                               use_4=self.use_4connectivity,
                                                               nodes_id=pixel_id,
                                                               dtype=np.float32,
                                                               nest=self.isNest,
                                                               )
            index = torch.from_numpy(graph_structure[0])
            weight = torch.from_numpy(graph_structure[1])
            if self.folder:
                print(f"Saving file {file_address}")
                torch.save({"index": index, "weight": weight}, file_address)
            return index, weight

        index, weight = self.getGraph(sampling_res=sampling_res)

        # patch_res is not None
        if num_hops is None:
            raise ValueError("num_hops must be given when we are splitting the graph to patches")

        n_patches, nPix_per_patch = self.getPatchesInfo(sampling_res, patch_res)
        assert (patch_id >=0) and (patch_id < n_patches), f"patch_id={patch_id} is not in valid range [0, {n_patches})"

        # https://github.com/rusty1s/pytorch_geometric/issues/1205
        # https://github.com/rusty1s/pytorch_geometric/issues/973
        interested_nodes = torch.arange(nPix_per_patch * patch_id, nPix_per_patch * (patch_id + 1), dtype=torch.long)
        subset, sub_edge_index, mapping, edge_mask = torch_g_utils.k_hop_subgraph(interested_nodes,
                                                                                          edge_index=index,
                                                                                          num_hops=num_hops,
                                                                                          relabel_nodes=True)
        sub_edge_weight = weight[edge_mask]
        if self.folder:
            print(f"Saving file {file_address}")
            torch.save({"index": sub_edge_index,
                        "weight": sub_edge_weight,
                        "nodes": subset,
                        "mapping": mapping},
                       file_address)
        return sub_edge_index, sub_edge_weight, subset, mapping

    def getMapHopToHop(self, sampling_res, patch_res, exterior_hop_number, patch_id, interior_hop_number):
        assert exterior_hop_number >= interior_hop_number, "num_hops_larger must be greater than num_hops_smaller"
        if self.folder:
            # Note self.weight_type, self.use_geodesic does not change anything in the mapping. So we can use
            filename = f"hopToHop_{self.use_4connectivity}_{sampling_res}_{patch_res}_{patch_id}_{exterior_hop_number}"
            filename += ".pth"
            file_address = os.path.join(self.folder, filename)
            if os.path.isfile(file_address):
                # print(f"Loading hop to hop file {file_address}")
                data_dict = torch.load(file_address)
                mapping = data_dict.get("map_hop_to_hop", None)
                return mapping

        _, _, interested_nodes, _ = self.getGraph(sampling_res=sampling_res, patch_res=patch_res, num_hops=interior_hop_number, patch_id=patch_id)
        index, _ = self.getGraph(sampling_res=sampling_res)
        _, _, mapping, _ = torch_g_utils.k_hop_subgraph(interested_nodes, edge_index=index, # type: ignore
                                                                num_hops=exterior_hop_number - interior_hop_number,
                                                                relabel_nodes=True)
        if self.folder:
            print(f"Saving hop to hop file {file_address}")
            torch.save({"map_hop_to_hop": mapping}, file_address)
        return mapping

    def getPatchesInfo(self, sampling_res, patch_res):
        nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution

        if (patch_res is None) or (patch_res < 0):   # Negative value means that the whole sphere is desired
            return 1, hp.nside2npix(nside)

        patch_width = hp.order2nside(patch_res)
        nPix_per_patch = patch_width * patch_width
        nside_patch = nside // patch_width
        n_patches = hp.nside2npix(nside_patch)
        return n_patches, nPix_per_patch

    def getLayerGraphUpsampling(self, scaling_factor_upsampling, hop_upsampling, resolution, patch_resolution=None, patch_id=-1, inputHopFromDownsampling=None):
        # print("starting unsampling graph construction", flush=True)
        assert len(scaling_factor_upsampling) == len(hop_upsampling), "list size for scaling factor and hop numbers must be equal"
        nconv_layers = len(scaling_factor_upsampling)
        list_sampling_res_conv, list_patch_res_conv = [[None] * nconv_layers for i in range(2)]
        list_sampling_res_conv[0] = resolution
        list_patch_res_conv[0] = patch_resolution

        patching = False
        if (patch_id != -1) and (patch_resolution is not None) and (patch_resolution > 0):
            patching = True

        for l in range(1, nconv_layers):
            list_sampling_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[l-1], scaling_factor_upsampling[l-1])
            if patching:
                list_patch_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[l-1], scaling_factor_upsampling[l-1])

        highest_sampling_res = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[-1], scaling_factor_upsampling[-1])
        if patching:
            highest_patch_res = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[-1], scaling_factor_upsampling[-1])

        list_index, list_weight, list_mapping_upsampling = [[None] * nconv_layers for i in range(3)]

        if not patching:
            index, weight = self.getGraph(sampling_res=list_sampling_res_conv[-1])
            l_first = next((i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]), -1) + 1
            list_index[l_first], list_weight[l_first] = index, weight
            for l in reversed(range(nconv_layers - 1)):
                if list_sampling_res_conv[l] != list_sampling_res_conv[l+1]:
                    index, weight = self.getGraph(sampling_res=list_sampling_res_conv[l])
                    l_first = next((i for i in reversed(range(l+1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]), -1) + 1
                    list_index[l_first], list_weight[l_first] = index, weight

            return {"list_sampling_res":list_sampling_res_conv, "list_index":list_index, "list_weight":list_weight, "output_sampling_res":highest_sampling_res}


        if all(v<0 for v in hop_upsampling):    # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
            index, weight, _, _ = self.getGraph(sampling_res=list_sampling_res_conv[-1], patch_res=list_patch_res_conv[-1], num_hops=0, patch_id=patch_id)
            l_first = next( (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]), -1) + 1
            list_index[l_first], list_weight[l_first] = index, weight
            for l in reversed(range(nconv_layers - 1)):
                if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
                    index, weight, _, _ = self.getGraph(sampling_res=list_sampling_res_conv[l], patch_res=list_patch_res_conv[l], num_hops=0, patch_id=patch_id)
                    l_first = next( (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]), -1) + 1
                    list_index[l_first], list_weight[l_first] = index, weight

            return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
                    "list_index": list_index, "list_weight": list_weight,
                    "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}

        K = hop_upsampling.copy()
        if inputHopFromDownsampling is not None:
            K[0] += inputHopFromDownsampling




        l_first = next((i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]), -1) + 1
        aggregated_K = np.sum(K[l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
        index, weight, nodes, mapping = self.getGraph(sampling_res=list_sampling_res_conv[-1], patch_res=list_patch_res_conv[-1], num_hops=aggregated_K + 1, patch_id=patch_id)

        if highest_sampling_res != list_sampling_res_conv[-1]:
            n_bitshit = 2 * (highest_sampling_res - list_sampling_res_conv[-1])
            n_children = 1 << n_bitshit
            mapping = mapping << n_bitshit
            mapping = mapping.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
            mapping = mapping.flatten()
        list_mapping_upsampling[-1] = mapping
        list_index[l_first], list_weight[l_first] = index, weight

        for l in reversed(range(nconv_layers-1)):
            if list_sampling_res_conv[l] != list_sampling_res_conv[l+1]:
                n_bitshit = 2 * (list_sampling_res_conv[l+1] - list_sampling_res_conv[l])
                parent_nodes = nodes >> n_bitshit
                # parent_nodes = parent_nodes.unique()

                index, weight = self.getGraph(sampling_res=list_sampling_res_conv[l])
                l_first = next((i for i in reversed(range(l+1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]), -1) + 1
                aggregated_K = np.sum(K[l_first:l+1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop

                parent_nodes, index, _, edge_mask = torch_g_utils.k_hop_subgraph(parent_nodes, edge_index=index,
                                                                                        num_hops=aggregated_K + 1,
                                                                                        relabel_nodes=True)
                weight = weight[edge_mask]

                n_children = 1 << n_bitshit
                generated_children_nodes_next_layer = parent_nodes << n_bitshit
                generated_children_nodes_next_layer = generated_children_nodes_next_layer.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
                generated_children_nodes_next_layer = generated_children_nodes_next_layer.flatten()
                mapping = (nodes.unsqueeze(1) == generated_children_nodes_next_layer).nonzero()[:, 1]

                nodes = parent_nodes

                list_mapping_upsampling[l] = mapping
                list_index[l_first], list_weight[l_first] = index, weight

        # print("ending unsampling graph construction", flush=True)
        return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
                "list_index": list_index, "list_weight": list_weight,
                "list_mapping": list_mapping_upsampling,
                "input_nodes": nodes,
                "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}


    def getLayerGraphs(self, scaling_factor_downsampling, hop_downsampling, scaling_factor_upsampling, hop_upsampling, upsampled_resolution, patch_upsampled_resolution=None, patch_id=-1):
        assert len(scaling_factor_downsampling) == len(hop_downsampling), "number of layers between scale factor and hops must be equal"
        nlayers_downsampling = len(scaling_factor_downsampling)

        assert len(scaling_factor_upsampling) == len(hop_upsampling), "number of layers between scale factor and hops must be equal"


        patching = False
        if (patch_id != -1) and (patch_upsampled_resolution is not None) and (patch_upsampled_resolution > 0):
            patching = True

        list_downsampling_res_conv, list_downsampling_patch_res_conv = [[None] * nlayers_downsampling for i in range(2)]
        list_downsampling_res_conv[0] = upsampled_resolution
        list_downsampling_patch_res_conv[0] = patch_upsampled_resolution

        for l in range(1, nlayers_downsampling):
            list_downsampling_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[l-1], scaling_factor_downsampling[l-1])
            if patching:
                list_downsampling_patch_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(list_downsampling_patch_res_conv[l-1], scaling_factor_downsampling[l-1])

        lowest_sampling_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[-1], scaling_factor_downsampling[-1])
        if patching:
            lowest_patch_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_patch_res_conv[-1], scaling_factor_downsampling[-1])

        list_index_downsampling, list_weight_downsampling, list_mapping_downsampling = [[None] * nlayers_downsampling for i in range(3)]

        assert all(v < 0 for v in hop_downsampling) == all(v < 0 for v in hop_upsampling), "for cutting graph both downsampling and upsamling hops must have negative elements"

        if not patching:
            dict_graphs = dict()
            dict_graphs["upsampling"] = self.getLayerGraphUpsampling(scaling_factor_upsampling, hop_upsampling, lowest_sampling_res)

            index, weight = self.getGraph(sampling_res=list_downsampling_res_conv[-1])
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
            for l in reversed(range(nlayers_downsampling - 1)):
                if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                    index, weight = self.getGraph(sampling_res=list_downsampling_res_conv[l])
                    l_first = next((i for i in reversed(range(l + 1)) if list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                    list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

            dict_graphs["downsampling"] = {"list_sampling_res":list_downsampling_res_conv, "list_index":list_index_downsampling, "list_weight":list_weight_downsampling}
            return dict_graphs

        if all(v < 0 for v in hop_downsampling):  # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
            dict_graphs = dict()
            dict_graphs["upsampling"] = self.getLayerGraphUpsampling(scaling_factor_upsampling, hop_upsampling, lowest_sampling_res, patch_resolution=lowest_patch_res, patch_id=patch_id)

            index, weight, node_ids, _ = self.getGraph(sampling_res=list_downsampling_res_conv[-1], patch_res=list_downsampling_patch_res_conv[-1], num_hops=0, patch_id=patch_id)
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
            for l in reversed(range(nlayers_downsampling - 1)):
                if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                    index, weight, node_ids, _ = self.getGraph(sampling_res=list_downsampling_res_conv[l], patch_res=list_downsampling_patch_res_conv[l], num_hops=0, patch_id=patch_id)
                    l_first = next((i for i in reversed(range(l + 1)) if list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                    list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight


            _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
            range_downsampling_input_to_patch = (int(patch_id*nPixPerPatch), int((patch_id+1)*nPixPerPatch))
            # Maybe later I can remove the next assert check.
            assert torch.all(torch.eq(node_ids, torch.arange(range_downsampling_input_to_patch[0], range_downsampling_input_to_patch[1], dtype=node_ids.dtype))), "node_ids must match range"

            dict_graphs["downsampling"] = {"list_sampling_res":list_downsampling_res_conv, "list_patch_res":list_downsampling_patch_res_conv,
                                           "list_index": list_index_downsampling, "list_weight": list_weight_downsampling,
                                           "range_downsampling_input_to_patch":range_downsampling_input_to_patch}
            return dict_graphs

        lowest_res_aggregated_hop = 0
        if list_downsampling_res_conv[-1] == lowest_sampling_res:
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            lowest_res_aggregated_hop = np.sum(hop_downsampling[l_first:])

        dict_graphs = dict()
        dict_graphs["upsampling"] = self.getLayerGraphUpsampling(scaling_factor_upsampling, hop_upsampling, lowest_sampling_res, patch_resolution=lowest_patch_res, patch_id=patch_id, inputHopFromDownsampling=lowest_res_aggregated_hop)

        # print("starting downsampling graph construction", flush=True)

        nodes = dict_graphs["upsampling"]["input_nodes"]
        index = dict_graphs["upsampling"]["list_index"][0]
        weight = dict_graphs["upsampling"]["list_weight"][0]

        _, nPixPerPatch = self.getPatchesInfo(lowest_sampling_res, lowest_patch_res)
        ind_start = (nodes == patch_id*nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
        # Maybe later I can remove the next assert check.
        assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch), torch.arange(patch_id*nPixPerPatch, (patch_id+1)*nPixPerPatch, dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
        range_downsampling_output_to_patch = (ind_start, ind_start+nPixPerPatch)


        if list_downsampling_res_conv[-1] == lowest_sampling_res:   # This means that last conv layer of downsampling has same size of first conv layer of upsampling
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            list_mapping_downsampling[-1] = None # This means that we are in the middle of layer so no mapping is needed
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
        else:
            n_bitshit = 2 * (list_downsampling_res_conv[-1] - lowest_sampling_res)
            n_children = 1 << n_bitshit
            nodes = nodes << n_bitshit
            nodes = nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
            nodes = nodes.flatten()
            index, weight = self.getGraph(sampling_res=list_downsampling_res_conv[-1])

            l_first = next((i for i in reversed(range(nlayers_downsampling)) if list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            aggregated_K = np.sum(hop_downsampling[l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop

            nodes, index, mapping, edge_mask = torch_g_utils.k_hop_subgraph(nodes, edge_index=index,
                                                                                    num_hops=aggregated_K + 1,
                                                                                    relabel_nodes=True)
            weight = weight[edge_mask]
            list_mapping_downsampling[-1] = mapping
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

        for l in reversed(range(nlayers_downsampling - 1)):
            if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                n_bitshit = 2 * (list_downsampling_res_conv[l] - list_downsampling_res_conv[l+1])
                n_children = 1 << n_bitshit
                nodes = nodes << n_bitshit
                nodes = nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
                nodes = nodes.flatten()
                index, weight = self.getGraph(sampling_res=list_downsampling_res_conv[l])

                l_first = next((i for i in reversed(range(l + 1)) if list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                aggregated_K = np.sum(hop_downsampling[l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop

                nodes, index, mapping, edge_mask = torch_g_utils.k_hop_subgraph(nodes, edge_index=index,
                                                                                        num_hops=aggregated_K + 1,
                                                                                        relabel_nodes=True)
                weight = weight[edge_mask]
                list_mapping_downsampling[l] = mapping
                list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

        _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
        ind_start = (nodes == patch_id * nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
        # Maybe later I can remove the next assert check.
        assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch), torch.arange(patch_id * nPixPerPatch, (patch_id + 1) * nPixPerPatch, dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
        range_downsampling_input_to_patch = (ind_start, ind_start+nPixPerPatch)

        # print("ending downsampling graph construction", flush=True)
        dict_graphs["downsampling"] = {"list_sampling_res":list_downsampling_res_conv, "list_patch_res":list_downsampling_patch_res_conv,
                                       "list_index":list_index_downsampling, "list_weight":list_weight_downsampling,
                                       "input_nodes":nodes, "list_mapping":list_mapping_downsampling,
                                       "range_downsampling_output_to_patch":range_downsampling_output_to_patch,
                                       "range_downsampling_input_to_patch":range_downsampling_input_to_patch}

        return dict_graphs