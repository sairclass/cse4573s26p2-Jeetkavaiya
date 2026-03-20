'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
# Helper function to convert image to float in range [0, 1] and also return the scale factor to restore the original range.
def _to_float01(img: torch.Tensor):
    x = img.float()
    scale_back = 1.0
    if x.max() > 1.5:
        x = x / 255.0
        scale_back = 255.0
    return x.clamp(0.0, 1.0), scale_back

def _restore_range(img: torch.Tensor, scale_back: float):
    out = img.clamp(0.0, 1.0)
    if scale_back > 1.5:
        out = out * scale_back
    return out

def _gray(img: torch.Tensor):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[1] == 1:
        return img
    return K.color.rgb_to_grayscale(img)


# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
