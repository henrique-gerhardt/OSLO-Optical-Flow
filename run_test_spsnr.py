import os, sys
import subprocess

def script_args_from_model(model_key:str):
    script_args = {}
    if 'factorizedPrior' in model_key:
        script_args['model'] = 'SphereFactorizedPrior'
    elif 'scaleHyperprior' in model_key:
        script_args['model'] = 'SphereScaleHyperprior'
    elif 'meanScaleHyperprior' in model_key:
        script_args['model'] = 'SphereMeanScaleHyperprior'
    if 'attention' in model_key: script_args['attention'] = ''
    if 'RB' in model_key: script_args['nonlinearity'] = 'RB'
    if 'cm_ar' in model_key:
        script_args['context-model'] = 'autoregressive'
        if 'no_relu' not in model_key: script_args['autoregressive-relu'] = ''
        if '_abs_' in model_key: script_args['autoregressive-abs'] = ''
        if '_arhops_' in model_key: script_args['autoregressive-hops'] = int(model_key.split('_arhops_')[1][0])
    return script_args

def get_test_dir_suff(data:str, checkpoint:str):
    test_dir_suff = ''
    if data != 'test':
        test_dir_suff = '_val' if data=='validation' else '_'+data 
    try:
        test_dir_suff += f'_ep_{int(checkpoint)}'
    except ValueError:
        if checkpoint!='best_loss': test_dir_suff += f'_{checkpoint}'
    return test_dir_suff

def get_chp_path(exp_path:str, key:str, checkpoint:str):
    return os.path.join(exp_path, key, f"{'checkpoint_' if checkpoint!='final' else ''}{checkpoint}.pth.tar")

def _create_cmd_str(script_name:str, script_args:dict):
    cmd = f'python {script_name} '
    for key, val in script_args.items():
        cmd += f'--{key} {val} '
    return cmd

def run_commands_seq(commands:list):
    run_str = '; '.join(commands)
    subprocess.Popen(run_str, shell=True)

def get_lam_q_from_key(key:str):
    lbd = float(key.split('_lambda_')[1].split('_')[0])
    q = int(key.split('_q_')[1][:1])
    return lbd, q

def create_test_script_args(key:str, exp_path:str, data_path:str, data:str, checkpoint:str, gid, exp_path_out=None):
    lbd, q = get_lam_q_from_key(key)
    if exp_path_out is None: exp_path_out = exp_path
    hr = 8 if 'hr_8' in key else 10
    test_dir_suff = get_test_dir_suff(data, checkpoint)
    script_args = {
        'out-dir': os.path.join(exp_path_out, key),
        'checkpoint-file': get_chp_path(exp_path, key, checkpoint),
        'test-data': os.path.join(data_path, f'Healpix_sampling_SUN360_res_{hr}', f'{data}.txt'),
        'neighbor-struct-dir': os.path.join(data_path, 'neighbor_structure'),
        'foldername-valtest': 'reconstruction'+test_dir_suff,
        'filename-test-results': 'test_results'+test_dir_suff,
        'batch-size-valtest': 10,
        'healpix-res': hr,
        'patch-res-valtest': hr-2,
        'print-freq': 1,
        'dataloader-num-workers': 2,
        'validation-start': 801,
        'gpu': '',
        'gpu_id': gid,
        'only-npy-valtest': '',
        'lambda': lbd,
        'quality': q,
    }
    script_args.update(script_args_from_model(key))
    return script_args

def create_spsnr_script_args(key:str, exp_path_out, data_path:str, data:str, checkpoint:str):
    hr = 8 if 'hr_8' in key else 10
    test_dir_suff = get_test_dir_suff(data, checkpoint)
    script_args = {
        'reconstruction-dir': os.path.join(exp_path_out, key),
        'original-dir': os.path.join(data_path, 'SUN360_pano9104x4552'),
        'test-data': os.path.join(data_path, f'Healpix_sampling_SUN360_res_{hr}', f'{data}.txt'),
        'reconstruction-subfolder': 'reconstruction'+test_dir_suff,
        'original-ext': 'jpg',
        'projection-original': 'erp',
        'test-files-prefix': f'healpix_sampling_res_{hr}_',
        'reconstruction-ext': 'npy',
        'reconstruction-prefix': f'healpix_sampling_res_{hr}_',
        'reconstruction-suffix': '_reconstructed',
        'projection': 'healpix',
        'rate-prefix': f'healpix_sampling_res_{hr}_',
        'sphere-points': os.path.join('misc', 'sphere_655362.txt'),
    }
    return script_args

def run_test_spsnr_seq(key, gid, exp_path, data_path, data, checkpoint, exp_path_out=None):
    if exp_path_out is None: exp_path_out = exp_path
    test_args = create_test_script_args(key, exp_path, data_path, data, checkpoint, gid, exp_path_out)
    spsnr_args = create_spsnr_script_args(key, exp_path_out, data_path, data, checkpoint)
    cmds = [
        _create_cmd_str('main_sphere_compression.py', test_args),
        _create_cmd_str(os.path.join('misc', 'compute_spherical_psnr.py'), spsnr_args),
    ]
    run_commands_seq(cmds)


if __name__ == '__main__':
    assert len(sys.argv[1:]) in [6,7], f"Usage: {sys.argv[0]} <exp_key> <gid> <exp_path> <data_path> <test_data> <checkpoint> [<exp_path_out>]"
    run_test_spsnr_seq(*sys.argv[1:])