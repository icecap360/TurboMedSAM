from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from framework import import_module

from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime


def preprocess_img(img_3c, target_length, normalize, resize_longest,
    means = [0.2482501, 0.21106622, 0.20026337],     
    stds = [0.3038128, 0.27170245, 0.26680432]):
    if resize_longest:
        img_256 = resize_longest_side(img_3c, target_length)
    else:
        img_256 = cv2.resize(img_3c, 
                             (target_length,target_length), 
                             interpolation=cv2.INTER_AREA)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    if normalize:
        img_256_norm = (img_256_norm-np.array(means))/np.array(stds) 
    if resize_longest:
        img_256_padded = pad_image(img_256_norm, target_length)
        return img_256_padded, newh, neww
    else:
        return img_256_norm, newh, neww

def resize_longest_side(image, target_length):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def get_bbox(mask, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes

def resize_box(box, original_size, target_length):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = target_length / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box
    
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box, target_length, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, target_length, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou

def MedSAM_infer_npz_2D(model, img_npz_file, target_length, normalize, resize_longest):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256, newh, neww = preprocess_img(img_3c, target_length=target_length, 
                                         normalize=normalize, 
                                         resize_longest=resize_longest)
    img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box(box, original_size=(H, W), target_length=target_length)
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(model, image_embedding, box256, target_length, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
   
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


def MedSAM_infer_npz_3D(model, img_npz_file, target_length, normalize, resize_longest):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256, new_H, new_W = preprocess_img(img_3c, target_length=target_length, 
                                         normalize=normalize, 
                                         resize_longest=resize_longest)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box(mid_slice_bbox_2d, original_size=(H, W), target_length=target_length)
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg, target_length)
                    pre_seg256 = pad_image(pre_seg256, target_length)
                    box_256 = get_bbox(pre_seg256)
                else:
                    box_256 = resize_box(mid_slice_bbox_2d, original_size=(H, W), target_length=target_length)
            img_2d_seg, iou_pred = medsam_inference(model, image_embedding, box_256, target_length, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256, new_H, new_W = preprocess_img(img_3c, target_length=target_length, 
                                         normalize=normalize, 
                                         resize_longest=resize_longest)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg, target_length)
                pre_seg256 = pad_image(pre_seg256, target_length)
                box_256 = get_bbox(pre_seg256)
            else:
                scale_256 = target_length / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            img_2d_seg, iou_pred = medsam_inference(model, image_embedding, box_256, target_length, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
        
        # from PIL import Image, ImageDraw
        # im = Image.fromarray(segs_3d_temp)
        # im = im.convert('RGB')
        # im.save("segs_3d_temp_idx{i}.png".format(idx))
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    data_root = '/data/qasim/MedSAM/split_npzs_3chnl/official_val'
    pred_save_dir = '/home/qasim/Projects/TurboMedSAM/results/RepViTm11_epoch4-Distill_ViTB_BasicAugmentation_epoch_1_20000/results_npz_basic'
    png_save_dir = '/home/qasim/Projects/TurboMedSAM/results/RepViTm11_epoch4-Distill_ViTB_BasicAugmentation_epoch_1_20000/results_overlay_basic'
    # checkpoint_path = '/home/qasim/Projects/TurboMedSAM/checkpoints/RepViTm11_epoch4-Distill_ViTB_AggressiveAugmentation_epoch_2.pth'
    # checkpoint_path = '/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_vit_b.pth'
    checkpoint_path = '/home/qasim/Projects/TurboMedSAM/results/RepViTm11_epoch4-Distill_ViTB_BasicAugmentation_epoch_1_20000/RepViTm11_epoch4-Distill_ViTB_BasicAugmentation_epoch_1_20000.pth'
    # checkpoint_path = '/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_vit_b.pth'
    config = '/home/qasim/Projects/TurboMedSAM/configs/CVPRMedSAMRepViTm11.py'
    device = 'cuda'
    img_size =  1024
    normalize = True
    resize_longest = True
    save_overlay = False
    measure_efficiency = True
    num_workers = 4
    img_npz_files = glob(join(data_root, '*.npz'), recursive=True)
    # img_npz_files = [os.path.join(data_root, '3DBox_MR_0173.npz')]
    # img_npz_files = missing_files
    
    abs_config_path = os.path.join("configs", config)
    cfg = import_module(os.path.basename(config), 
                        abs_config_path)
    model = cfg.model

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        metadata = getattr(state_dict, '_metadata', ())
        state_dict = state_dict['state_dict']
        # state_dict._metadata = metadata
    revise_keys = [(r'^module.', ''), (r'^_orig_mod.', '')]
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
            for k, v in state_dict.items()})
    # Keep metadata in state_dict
    model.load_checkpoint(state_dict)
    model.to(device)
    model.eval()
    
    if save_overlay:
        makedirs(png_save_dir, exist_ok=True)
    #%% set seeds
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    makedirs(pred_save_dir, exist_ok=True)
    device = torch.device(device)
    
    if measure_efficiency:
        efficiency = OrderedDict()
        efficiency['case'] = []
        efficiency['time'] = []
        for img_npz_file in tqdm(img_npz_files):
            start_time = time()
            if basename(img_npz_file).startswith('3D'):
                MedSAM_infer_npz_3D(model, img_npz_file, img_size, normalize, resize_longest)
            else:
                MedSAM_infer_npz_2D(model, img_npz_file, img_size, normalize, resize_longest)
            end_time = time()
            efficiency['case'].append(basename(img_npz_file))
            efficiency['time'].append(end_time - start_time)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
        efficiency_df = pd.DataFrame(efficiency)
        efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
    else:
        raise Exception('Use other script')
