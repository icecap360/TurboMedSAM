import glob
import os
import numpy as np

data_root = '/data/qasim/MedSAM/split_npzs_3chnl/official_val/'
result_dir = '/home/qasim/Projects/TurboMedSAM/work_dir/CVPRMedSAMRepViTm11/results_npz/'
search_prefix = '*.npz'

files = glob.glob("{}{}".format(data_root,search_prefix))
for input_file in files:
    pred_file = os.path.join(
        result_dir, os.path.basename(input_file)
    )
    inputs = np.load(input_file)
    n_boxes = len(inputs['boxes'])
    in_shape = inputs['imgs'].shape
    outputs = np.load(pred_file)
    pred_shape = outputs['segs'].shape
    max_idx = np.max(outputs['segs'])
    if len(pred_shape) == 2 and pred_shape != in_shape[:-1]:
        print('shape error', pred_file, pred_shape, in_shape)
    elif len(pred_shape) == 3 and pred_shape != in_shape:
        print('shape error', pred_file, pred_shape, in_shape)
    if max_idx != n_boxes:
        print('idx error', os.path.basename(input_file), max_idx, n_boxes)