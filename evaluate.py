
import os
import torch
import numpy as np
import argparse

import joblib
import math
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from scipy.spatial.distance import jensenshannon

from eval import build_model, MODEL2BATCHSIZE, DATASET_CONFIG, DATA_CONFIG, VOXEL_SIZE
from eval.metric_utils import compute_batch_logits, volume_sum_update, voxelize_pcd, compute_pairwise_cd_batch
from eval.fid_score import calculate_frechet_distance

MODEL2METRICS = {
    'rangevit_wo_i': 'FRID',
    'rangevit_w_i': 'FRD',
    'minkowskinet': 'FSVD',
    'spvcnn': 'FPVD',
    'liploc': 'FLD',
}

def save(results, results_path, latex_path):
    results = results.copy()
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if os.path.exists(latex_path):
        os.remove(latex_path)

    results.drop(columns=['samples path'], inplace=True)

    model_col = results.columns[0]
    metric_cols = results.columns[1:]

    exponents = {}
    scaled_cols = {}

    for col in metric_cols:
        col_vals = results[col].astype(float)
        col_vals_nonzero = col_vals.replace(0, np.nan).dropna()
        if len(col_vals_nonzero) == 0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(np.abs(col_vals_nonzero)).min()))
        exponents[col] = exp
        scaled_cols[col] = col_vals / (10.0 ** exp)

    scaled_df = results.copy()
    for col in metric_cols:
        scaled_df[col] = scaled_cols[col]

    header_main = [model_col] + [col for col in metric_cols]
    header_exp = [""] + [f"$\\times 10^{{{exponents[col]}}}$" for col in metric_cols]

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}"
        return x

    formatted = scaled_df.applymap(fmt)

    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n\\small\n")
        f.write("\\begin{tabular}{l" + "c" * len(metric_cols) + "}\n")
        f.write("\\toprule\n")
        
        # Write column names
        f.write(" & ".join(header_main) + " \\\\\n")
        f.write(" & ".join(header_exp) + " \\\\\n")
        f.write("\\midrule\n")

        # Write data rows
        for row in formatted.itertuples(index=False):
            f.write(" & ".join(str(cell) for cell in row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{State-of-the-art comparison with per-row scaled metrics.}\n")
        f.write("\\label{tab:sota_comparison}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to {latex_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ['minkowskinet', 'spvcnn', 'liploc', 'rangevit_wo_i', 'rangevit_w_i']

    #Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluation script for Repa3D")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--sanity-check', action='store_true', help='Perform sanity check on the samples')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    
    model_names = config.models.keys()
    
    # Dataframe to store results
    results_all = pd.DataFrame(columns=['Method', 'FRD', 'FRID', 'FLD', 'FSVD', 'FPVD', 'JSD', 'TV', 'MMD', 'samples path'])
    results_val = pd.DataFrame(columns=['Method', 'FRD', 'FRID', 'FLD', 'FSVD', 'FPVD', 'JSD', 'TV', 'MMD', 'samples path'])

    time_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    models_string = "_".join(list(config.models.keys()))

    results_all_path = os.path.join(config.save_path, "trainval", f'eval_{time_stamp}_{models_string}.csv')
    results_val_path = os.path.join(config.save_path, "valonly", f'eval_{time_stamp}_{models_string}.csv')
    latex_all_path = os.path.join(config.save_path, "trainval", f'eval_{time_stamp}_{models_string}.tex')
    latex_val_path = os.path.join(config.save_path, "valonly", f'eval_{time_stamp}_{models_string}.tex')

    # Load the dataset statistics
    dataset_stats = np.load(config.dataset_stat_path, allow_pickle=True).item()

    for model_name in config.models.keys():
        sample_path = config.models[model_name].samples_path
        metrics = config.models[model_name].metrics
        print(f"Evaluating model {model_name} with samples from {sample_path}")

        results_all_row = {'Method': model_name, 'samples path': sample_path}    
        results_val_row = {'Method': model_name, 'samples path': sample_path}    

        # Load the model and samples
        if sample_path.endswith('.npz'):
            all_imgs = np.load(sample_path)['arr_0']
            if all_imgs.ndim == 3:
                all_imgs = np.expand_dims(all_imgs, axis=1)
            depth = rearrange(all_imgs[:, 0],'b w h -> b (w h)')
            depth_range = DATASET_CONFIG['kitti']['depth_range']
            mask = (depth > depth_range[0]) & (depth < depth_range[1])

            if all_imgs.shape[1] < 4:
                pcd_path = sample_path.replace('img.npz', 'samples.pcd')
                all_pcds = joblib.load(pcd_path)
            else:
                pcd = rearrange(all_imgs[:, 1:4],'b c w h -> b (w h) c') #xyz
                all_pcds = [pcd[i][mask[i]] for i in range(pcd.shape[0])]
            
            mask = rearrange(mask,'b (h w) -> b h w',w=all_imgs.shape[-1])
            depth = all_imgs[:, 0]*mask - (1-mask)
            print(depth.min(), depth.max())
            intensity = all_imgs[:, -1]*mask
            all_range_imgs = np.concatenate([depth[:,None], intensity[:,None]], axis=1) # depth and intensity
            #remove from ram all_imgs
            del all_imgs
            del mask
            del depth
            del intensity
        
        if sample_path.endswith('.pcd'):
            all_pcds = joblib.load(sample_path)
            all_range_imgs = [0]*len(all_pcds)  # Placeholder for range images

        if args.sanity_check:
            all_pcds = all_pcds[:10]
            all_range_imgs = all_range_imgs[:10]
        else:
            assert len(all_pcds) >= 10000, "Prepare 10000 samples before evaluation"
            all_pcds = all_pcds[:10000]
            all_range_imgs = all_range_imgs[:10000]

        #Compute Metrics
        for m in models:
            if m not in metrics:
                print(f"Model {m} not in metrics for {model_name}, skipping...")
                continue
            is_voxel = m in ['minkowskinet', 'spvcnn']

            bs = MODEL2BATCHSIZE[m]

            print(f"loading model {m}...")
            model = build_model('kitti',m, device)
            print(f"model {m} loaded.")
            
            print(f"Computing logits of {model_name} with model {m}...")
            all_logits_list = []
            if is_voxel:
                for i in range(math.ceil(len(all_pcds) / bs)):
                    batch = all_pcds[i * bs:(i + 1) * bs]
                    logits = compute_batch_logits(batch, model, is_voxel=is_voxel, dataset_config=DATASET_CONFIG['kitti'])
                    all_logits_list.extend(logits)
                all_logits = np.stack(all_logits_list)
            else:
                for i in range(math.ceil(len(all_range_imgs) / bs)):
                    batch = all_range_imgs[i * bs:(i + 1) * bs]
                    logits = compute_batch_logits(batch, model, is_voxel=is_voxel, dataset_config=DATASET_CONFIG['kitti'])
                    all_logits_list.append(logits)
                all_logits = np.vstack(all_logits_list)
            
            mu = np.mean(all_logits, axis=0)
            sigma = np.cov(all_logits, rowvar=False)

            mu_all_dataset = dataset_stats['all'][m]['mu']
            sigma_all_dataset = dataset_stats['all'][m]['sigma']

            mu_val_dataset = dataset_stats['val'][m]['mu']
            sigma_val_dataset = dataset_stats['val'][m]['sigma']

            fid_all = calculate_frechet_distance(mu, sigma, mu_all_dataset, sigma_all_dataset)
            fid_val = calculate_frechet_distance(mu, sigma, mu_val_dataset, sigma_val_dataset)

            # Store the results in the dataframe
            results_all_row[MODEL2METRICS[m]] = fid_all
            results_val_row[MODEL2METRICS[m]] = fid_val

            print(f"###--- Computed {MODEL2METRICS[m]}: {fid_all:.4f} for model {model_name} ---###")

        x_range, y_range = DATA_CONFIG['64']['x'], DATA_CONFIG['64']['y']
        vol_shape = (math.ceil((x_range[1] - x_range[0]) / VOXEL_SIZE), math.ceil((y_range[1] - y_range[0]) / VOXEL_SIZE))
        min_bound = (math.ceil((x_range[0]) / VOXEL_SIZE), math.ceil((y_range[0]) / VOXEL_SIZE))
        
        #Compute voxelized point cloud bev
        if 'mmd' in metrics:
            print("Computing dataset histogram...")


            all_pcds_hists = []
            for pcd in all_pcds[:2000]:
                pcd_voxel = voxelize_pcd(pcd, x_range, y_range, VOXEL_SIZE, min_bound, vol_shape)
                all_pcds_hists.append(pcd_voxel)

            print("Dataset histogram computed.")
            print("Computing Maximum Mean Discrepancy (MMD)...")
            
            val_histograms = dataset_stats['val']['valset_voxelized_bev']['points']

            val_histograms = val_histograms[:1000]  # Use a subset of 1000 samples for MMD computation

            if args.sanity_check:
                mmd = 0.0
            else:
                results_dist = []
                for r in tqdm(val_histograms):
                    dists = compute_pairwise_cd_batch(r, all_pcds_hists)
                    results_dist.append(min(dists))
                mmd = sum(results_dist) / len(results_dist)
            
            results_all_row['MMD'] = mmd
            results_val_row['MMD'] = mmd

            print(f"###--- Computed MMD: {mmd:.4f} for model {model_name} ---###")

        ## Compute bev histogram ##
        if 'jsd' in metrics or 'tv' in metrics:
            print("Computing histograms...")
            volume_sum = np.zeros(vol_shape, np.float32)

            for pcd in all_pcds:
                volume_sum = volume_sum_update(volume_sum, pcd, x_range, y_range, VOXEL_SIZE, min_bound)

            all_dataset_volume_sum = dataset_stats['all']['dataset_bev_hist']
            val_dataset_volume_sum = dataset_stats['val']['dataset_bev_hist']

            print("Validation computed.")

            all_dataset_volume_sum = (all_dataset_volume_sum / np.sum(all_dataset_volume_sum)).flatten()
            val_dataset_volume_sum = (val_dataset_volume_sum / np.sum(val_dataset_volume_sum)).flatten()
            volume_sum = (volume_sum / np.sum(volume_sum)).flatten()
            
            
            jsd_all = jensenshannon(all_dataset_volume_sum, volume_sum)
            tv_all = np.abs(all_dataset_volume_sum - volume_sum).sum() 
            jsd_val = jensenshannon(val_dataset_volume_sum, volume_sum)
            tv_val = np.abs(val_dataset_volume_sum - volume_sum).sum()

            results_all_row['JSD'] = jsd_all
            results_all_row['TV'] = tv_all
            results_val_row['JSD'] = jsd_val
            results_val_row['TV'] = tv_val

            print(f"###--- Computed JSD: {jsd_all:.4f} for model {model_name} ---###")
            print(f"###--- Computed TV: {tv_all:.4f} for model {model_name} ---###")
        
        results_all.loc[len(results_all)] = results_all_row
        results_val.loc[len(results_val)] = results_val_row

        save(results_all,results_all_path, latex_all_path)
        save(results_val,results_val_path, latex_val_path)

            



    