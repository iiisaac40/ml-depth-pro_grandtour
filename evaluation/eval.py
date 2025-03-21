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


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--dataset', default='grandtour', choices=['hypersim', 'vkitti', 'grandtour'])
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
        valset = GRANDTOUR(args.dataset_file_path, 'val', device=local_rank, parent_data_dir='/'.join(args.dataset_file_path.split('/')[:-2]))
    elif args.dataset == 'kitti':
        from dataset.kitti import KITTI
        valset = KITTI(args.dataset_file_path, 'val', size=size) # parent_data_dir='/'.join(args.dataset_file_path.split('/')[:-2])
    
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=0, drop_last=True, sampler=valsampler)
    
    
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

        img_np = img.cpu().numpy()
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
        
        # Add this after the cur_results line
        if i % 10 == 0 and args.vis_res == 'TRUE':  # Visualize every 10th sample
            # import cv2
            import matplotlib.pyplot as plt

            # img_np = img.detach().cpu().numpy()
            # img_np = np.clip(img_np, 0, 1)

            valid_mask_np = valid_mask.cpu().numpy().astype(np.uint8)

            pred_np = pred.cpu().numpy()
            depth_np = depth.cpu().numpy()
            
            print(f"pred depth: min: {np.min(pred_np)}, max: {np.max(pred_np)}")
            print(f"depth_np: min: {np.min(depth_np)}, max: {np.max(depth_np)}")
            
            valid_mask_vis = np.zeros_like(pred_np)
            valid_mask_vis[valid_mask_np == 1] = 1
        
            
            # Create output dir
            os.makedirs("/home/output/visualizations/GrandTour_depthpro", exist_ok=True)
                        
            # Add prediction visualization to the plot
            plt.figure(figsize=(30, 20))
            
            # Original Image
            plt.subplot(221)
            plt.imshow(img_np)
            plt.title("Original Image")
            
            # Valid Mask
            plt.subplot(222)
            plt.imshow(valid_mask_vis)
            plt.title("Valid Regions")
            
            # Prediction
            plt.subplot(223)
            plt.imshow(pred_np, cmap='turbo_r', vmin=np.min(pred_np), vmax=np.max(pred_np))
            plt.colorbar(label='Depth (m)')
            plt.title("Predicted Depth")

            # GT depth
            plt.subplot(224)
            plt.imshow(depth_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            plt.colorbar(label='Depth (m)')
            plt.title("GT Depth")
            
            image_path = sample['image_path'][0]
            print(f"image_path: {image_path}")
            timestamp = image_path.split()[0].split('/')[-1].split('.')[0]
            plt.savefig(f"/home/output/visualizations/GrandTour_depthpro/sample_{timestamp}.png")
            plt.close()
        
        if valid_mask.sum() < 10:
            continue
        print(f"pred shape: {pred.shape}, depth shape: {depth.shape}, img shape: {img.shape} ")
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
        
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