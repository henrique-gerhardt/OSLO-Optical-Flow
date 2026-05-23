import argparse
import healpix_graph_loader
import time

parser = argparse.ArgumentParser()
parser.add_argument("--healpix-res", "-hr", type=int, default=10, help="Resolution of the healpix for sampling.")
parser.add_argument("--patch-res", "-pr", type=int, default=6, help="Resolution of the healpix patches.")
parser.add_argument("--out-graph-dir", "-ogd", type=str, default="../GraphData", help="Directory to save graph and subgraph structures.")
parser.add_argument("--use-4connectivity", action='store_true', help='use 4 neighboring for graph construction')
parser.add_argument("--use-euclidean", action='store_true', help='Use geodesic distance for graph weights')
parser.add_argument("--weight-type", '-w', type=str, default='gaussian', help="Weighting function on distances between nodes of the graph")
parser.add_argument("--num-hops", "-nh", type=int, default=0, help="Considered num of hops for each patch.")

args = parser.parse_args()

if __name__ == '__main__':

    print("=========== printing args ===========")
    print("--healpix-res", args.healpix_res)
    print("--patch-res", args.patch_res)
    print("--out-graph-dir", args.out_graph_dir)
    print("--use-4connectivity", args.use_4connectivity)
    print("--use-euclidean", args.use_euclidean)
    print("--weight-type", args.weight_type)
    print("--num-hops", args.num_hops)


    graph_loader = healpix_graph_loader.HealpixGraphLoader(weight_type=args.weight_type, use_geodesic=not args.use_euclidean, use_4connectivity=args.use_4connectivity, load_save_folder=args.out_graph_dir)
    n_patches, _ = graph_loader.getPatchesInfo(sampling_res=args.healpix_res, patch_res=args.patch_res)
    print(f"# patches={n_patches}")
    start_time_total = time.time()
    for patch_id in range(n_patches):
        start_time_patch = time.time()
        graph_loader.getGraph(sampling_res=args.healpix_res, patch_res=args.patch_res, num_hops=args.num_hops, patch_id=patch_id)
        end_time_patch = time.time()
        print(f"Patch {patch_id} finished in {end_time_patch-start_time_patch} seconds", flush=True)
    end_time_total = time.time()
    print(f"--- {end_time_total - start_time_total} seconds in total ---")
