#!/usr/bin/env python3

import argparse
import logging
import os
import pprint
import random
import gc

import warnings
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from util.dist_helper import setup_distributed
from util.utils import init_log


import depth_pro

import csv
import os
import sys
sys.path.append('/home/grand_tour_depth_benchmark/utils')
from depth_alignment_utils import align_depth_least_squares

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)
    mae = torch.mean(torch.abs(diff))

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'mae':mae.item()}


parser = argparse.ArgumentParser(description='Depth Pro for Metric Depth Estimation')

parser.add_argument('--dataset', default='grandtour')
parser.add_argument('--dataset_file_path', type=str, help='the path pointing to the dataset')
parser.add_argument('--depth_alignment', type=str, default='TRUE', choices=['TRUE', 'FALSE'], help='Activate Depth Alignment or Not')
parser.add_argument('--csv_file', type=str, default="metric.csv", help='Save Metric to CSV file')
parser.add_argument('--vis_res', type=str,default='FALSE', choices=['TRUE', 'FALSE'], help='Activate Saving Visualization Result')
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--min_depth', default=0.1, type=float)
parser.add_argument('--max_depth', default=60, type=float)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def main():
    args = parser.parse_args()
    
    # warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)

    local_rank = int(os.environ["LOCAL_RANK"])
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)  
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/test.txt', 'test', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'grandtour':
        from dataset.grandtour import GRANDTOUR
        valset = GRANDTOUR(args.dataset_file_path, 'val', device=local_rank, parent_data_dir='/mnt/GrandTour')
    elif args.dataset == 'kitti':
        from dataset.kitti import KITTI
        valset = KITTI(args.dataset_file_path, 'val', size=size) # parent_data_dir='/'.join(args.dataset_file_path.split('/')[:-2])
    
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=1, drop_last=True, sampler=valsampler)
    
    
    model, transform = depth_pro.create_model_and_transforms(device=local_rank)
    
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                   output_device=local_rank, find_unused_parameters=True)
    
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100, 'mae':100}
        
    model.eval()
    
    results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
               'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
               'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda(),
               'mae': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    
    data_iter = tqdm(enumerate(valloader), total=len(valloader)) if rank == 0 else enumerate(valloader)

    for i, sample in data_iter:
        
        img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
        depth = depth.squeeze()
        valid_mask = valid_mask.squeeze()
        img = img.squeeze()

        img_np = img.permute(1, 2, 0).contiguous().cpu().numpy()
        # img_np = img.cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        img_pil = Image.fromarray(img_np)
        image, _, f_px = depth_pro.load_rgb(img_pil)
        image = transform(image)

        # cv2.imwrite("temp.png", image.detach().cpu().numpy().astype(np.uint8))
        with torch.no_grad():
            pred = model.infer(image, f_px=f_px)["depth"]
        

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

        if args.depth_alignment == 'TRUE':
            aligned_pred, _, _ = align_depth_least_squares(pred.cpu().numpy(), depth.cpu().numpy(), valid_mask.cpu().numpy())
            aligned_pred = np.clip(aligned_pred, a_min=args.min_depth, a_max=args.max_depth)
            pred = torch.tensor(aligned_pred, dtype=torch.float32, device=local_rank)

        if valid_mask.sum() < 10:
            continue
            
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        if (rank == 0 and i % 10 == 0) and args.vis_res == 'TRUE':    # Visualize every 10th sample, and i % 10 == 0
            import cv2
            import matplotlib.pyplot as plt

            print(f"saving!!! {args.csv_file.split('/')[-1].split('.')[-2]}")

            img_np = img.cpu().numpy()#.transpose(1, 2, 0)
            print(f"img_np shape: {img_np.shape}")
            # img_np = img_np * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
            img_np = np.clip(img_np, 0, 1)
        
            pred_np = pred.cpu().numpy()
            depth_np = depth.cpu().numpy()
            # depth_np[depth_np == 0] = np.nan
            
            valid_pred = pred[valid_mask].cpu().numpy()
            valid_depth = depth[valid_mask].cpu().numpy()
            print(f"pred depth: min: {np.min(valid_pred)}, max: {np.max(valid_pred)}")
            print(f"depth_np: min: {np.min(valid_depth)}, max: {np.max(valid_depth)}")
            

            # Calculate error only on valid regions
            error_values = np.abs(valid_pred - valid_depth)
            cmap = plt.get_cmap("turbo_r")
            norm_error_values = (error_values - np.min(error_values)) / (np.max(error_values) - np.min(error_values))
            color_norm_error_values = (cmap(norm_error_values)[..., :3] * 255).astype(np.uint8)
            
            # error_map = np.full_like(pred_np, fill_value=60)  
            error_map = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
            error_map[valid_mask.cpu().numpy()] = (0.8 * color_norm_error_values + (1 - 0.8) * error_map[valid_mask.cpu().numpy()]).astype(np.uint8)  
            print(f"valid_mask_shape: {np.sum(valid_mask.cpu().numpy())}, error_values shape: {error_values.shape}")

            min_error = np.nanmin(error_values)
            max_error = np.nanmax(error_values)
            print(f"min_error: {min_error}; max_error: {max_error}")

            print(f"cur_results: {cur_results}")
                
            
            # Create output dir
            os.makedirs(f"/mnt/GrandTour/visualizations/{args.csv_file.split('/')[-1].split('.')[-2]}", exist_ok=True) # args.dataset_file_path.split('/')[-1].split('.')[-2]
                        
            # Add prediction visualization to the plot
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in cur_results.items()])

            fig, axes = plt.subplots(2, 3, figsize=(36, 20))
            fig.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing between subplots

            # First row
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')

            im = axes[0, 1].imshow(error_map, cmap='turbo_r', vmin=min_error, vmax=max_error)
            fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[0, 1].set_title("Error Map")
            axes[0, 1].axis('off')

            norm_pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np))
            im = axes[0, 2].imshow(norm_pred_np, cmap='turbo_r', vmin=np.min(norm_pred_np), vmax=np.max(norm_pred_np))
            fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[0, 2].set_title("Normalized Predicted Depth")
            axes[0, 2].axis('off')

            # Second row
            im = axes[1, 0].imshow(pred_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[1, 0].set_title("Predicted Depth")
            axes[1, 0].axis('off')

            im = axes[1, 1].imshow(depth_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[1, 1].set_title("GT Depth")
            axes[1, 1].axis('off')

            norm_depth_np = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
            cmap = plt.get_cmap('turbo_r')
            colored_depth = cmap(norm_depth_np)[..., :3] 

            # Alpha blend with RGB image
            overlay_img = np.copy(img_np)
            overlay_img[valid_mask.cpu().numpy()] = colored_depth[valid_mask.cpu().numpy()]

            axes[1, 2].imshow(overlay_img)
            axes[1, 2].set_title("GT Depth Overlay")
            axes[1, 2].axis('off')

            plt.figtext(0.5, 0.05,  # Center bottom
            metrics_text,
            ha='center',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8))

            
            image_path = sample['image_path'][0]
            print(f"image_path: {image_path}")
            # timestamp = image_path.split()[0].split('/')[-1].split('.')[0]
            # plt.savefig(f"/mnt/GrandTour/visualizations/{args.csv_file.split('/')[-1].split('.')[-2]}/{timestamp}.png")
            # plt.close()
            plt.close('all')

            # Root directory for saving
            base_vis_dir = f"/mnt/GrandTour/visualizations/{args.csv_file.split('/')[-1].split('.')[-2]}"
            os.makedirs(base_vis_dir, exist_ok=True)

            # Prepare timestamp
            image_path = sample['image_path'][0]
            timestamp =  image_path.split()[0].split('/')[-1].split('.')[0]

            # Visualization mappings
            visuals = {
                "original_image": (img_np, None, "Original Image"),
                "error_map": (error_map, (min_error, max_error), "Error Map"),
                "normalized_pred_depth": (norm_pred_np, (np.min(norm_pred_np), np.max(norm_pred_np)), "Normalized Predicted Depth"),
                "predicted_depth": (pred_np, (np.min(depth_np), np.max(depth_np)), "Predicted Depth"),
                "gt_depth": (depth_np, (np.min(depth_np), np.max(depth_np)), "GT Depth"),
                "gt_overlay": (overlay_img, None, "GT Depth Overlay"),
            }

            for key, (data, vrange, title) in visuals.items():
                fig, ax = plt.subplots(figsize=(12, 6))
                if vrange:
                    im = ax.imshow(data, cmap='turbo_r', vmin=vrange[0], vmax=vrange[1])
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Depth (m)')
                else:
                    ax.imshow(data)
                ax.set_title(title)
                ax.axis('off')

                out_dir = os.path.join(base_vis_dir, key)
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, f"{timestamp}.png"))
                plt.close()


        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

        if rank == 0:
            data_iter.set_description(f"mae: {cur_results['mae']:.3f}")
    
    torch.distributed.barrier()
    
    for k in results.keys():
        dist.reduce(results[k], dst=0)
    dist.reduce(nsamples, dst=0)
    
    if rank == 0:
        averaged_metrics = {k: (v / nsamples).item() for k, v in results.items()}
    
        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        logger.info('==========================================================================================')
        print()
    
        csv_file = args.csv_file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            if f.tell() == 0:
                writer.writeheader()
            
            writer.writerow(averaged_metrics)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())



if __name__ == '__main__':
    main()