import numpy as np
import cv2, os
from PIL import Image

fname = '2DBox_MR_0133.npz'
input_path = os.path.join('/data/qasim/MedSAM/official_val/', fname)
pred_path = os.path.join(
        '/home/qasim/Projects/TurboMedSAM/work_dir/CVPRMedSAMRepViTm11/results_npz/', 
        fname)
predicted_mask = np.load(pred_path)['segs']


npz = np.load(input_path)
img = npz['imgs']
boxes = npz['boxes']
img[np.where(predicted_mask)] = np.array([255,0,0], np.uint8)

def highlight_boxes(arr, boxes):
  """
  Highlights bounding boxes on a grayscale NumPy array.

  Args:
      arr: Grayscale NumPy array representing the image.
      boxes: A NumPy array of shape (num_boxes, 4) where each row represents 
             a bounding box in XYXY format (xmin, ymin, xmax, ymax).

  Returns:
      A new NumPy array with bounding boxes highlighted.
  """
  arr_copy = arr.copy()  # Avoid modifying original array
  for box in boxes:
    xmin, ymin, xmax, ymax = box.astype(np.int64)
    # Draw a rectangle with thickness 2 and color (0, 255, 0) - green
    cv2.rectangle(arr_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
  return arr_copy

highlighted_image = highlight_boxes(img, boxes)
fname = os.path.basename(pred_path)[:-4] + '.png'
Image.fromarray(highlighted_image).save(fname)

