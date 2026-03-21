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

# Helper function to compute Harris corner response for a grayscale image.
def _harris_response(gray: torch.Tensor):
    grad = K.filters.spatial_gradient(gray)
    ix = grad[:, :, 0]
    iy = grad[:, :, 1]

    ix2 = K.filters.gaussian_blur2d(ix * ix, (7, 7), (1.5, 1.5))
    iy2 = K.filters.gaussian_blur2d(iy * iy, (7, 7), (1.5, 1.5))
    ixy = K.filters.gaussian_blur2d(ix * iy, (7, 7), (1.5, 1.5))

    k = 0.04
    det = ix2 * iy2 - ixy * ixy
    tr = ix2 + iy2
    r = det - k * tr * tr
    return r

# Helper function to detect keypoints using Harris corner response and non-maximum suppression.
def _detect_keypoints(img: torch.Tensor, max_pts=700, patch_size=11, nms_size=9):
    gray = _gray(img)
    r = _harris_response(gray)

    margin = patch_size // 2 + 2
    if r.shape[-2] <= 2 * margin or r.shape[-1] <= 2 * margin:
        return torch.empty((0, 2), device=img.device)

    r[:, :, :margin, :] = -1e9
    r[:, :, -margin:, :] = -1e9
    r[:, :, :, :margin] = -1e9
    r[:, :, :, -margin:] = -1e9

    mx = torch.nn.functional.max_pool2d(r, kernel_size=nms_size, stride=1, padding=nms_size // 2)
    keep = (r == mx)

    rmax = r.max()
    if rmax <= 0:
        return torch.empty((0, 2), device=img.device)

    keep = keep & (r > 0.01 * rmax)

    pts_yx = torch.nonzero(keep[0, 0], as_tuple=False)
    if pts_yx.shape[0] == 0:
        return torch.empty((0, 2), device=img.device)

    vals = r[0, 0, pts_yx[:, 0], pts_yx[:, 1]]
    vals, order = torch.sort(vals, descending=True)
    pts_yx = pts_yx[order]

    if pts_yx.shape[0] > max_pts:
        pts_yx = pts_yx[:max_pts]

    pts_xy = torch.stack([pts_yx[:, 1].float(), pts_yx[:, 0].float()], dim=1)
    return pts_xy

# Helper function to compute descriptors for keypoints using normalized patches.
def _describe_patches(img: torch.Tensor, pts_xy: torch.Tensor, patch_size=11):
    if pts_xy.shape[0] == 0:
        return torch.empty((0, patch_size * patch_size), device=img.device)

    gray = _gray(img)
    b, c, h, w = gray.shape
    rad = patch_size // 2

    padded = torch.nn.functional.pad(gray, (rad, rad, rad, rad), mode='reflect')
    unfolded = torch.nn.functional.unfold(padded, kernel_size=(patch_size, patch_size))
    unfolded = unfolded[0].t()  # (H*W, patch_size*patch_size)

    x = pts_xy[:, 0].long().clamp(0, w - 1)
    y = pts_xy[:, 1].long().clamp(0, h - 1)
    idx = y * w + x

    desc = unfolded[idx]
    desc = desc - desc.mean(dim=1, keepdim=True)
    desc = desc / (desc.std(dim=1, keepdim=True) + 1e-6)
    desc = torch.nn.functional.normalize(desc, p=2, dim=1)
    return desc

def _extract_features(img: torch.Tensor, max_pts=700, patch_size=11):
    pts = _detect_keypoints(img, max_pts=max_pts, patch_size=patch_size)
    desc = _describe_patches(img, pts, patch_size=patch_size)
    return pts, desc

# Helper function to match descriptors between two sets using ratio test and mutual nearest neighbor check.
def _match_descriptors(desc1: torch.Tensor, desc2: torch.Tensor, ratio=0.82):
    if desc1.shape[0] < 4 or desc2.shape[0] < 4:
        return torch.empty((0, 2), dtype=torch.long, device=desc1.device)

    dmat = torch.cdist(desc1, desc2, p=2)

    if desc2.shape[0] < 2:
        return torch.empty((0, 2), dtype=torch.long, device=desc1.device)

    vals12, idx12 = torch.topk(dmat, k=2, largest=False, dim=1)
    best12 = idx12[:, 0]
    ratio_ok = vals12[:, 0] / (vals12[:, 1] + 1e-8) < ratio

    best21 = torch.argmin(dmat, dim=0)
    ids1 = torch.arange(desc1.shape[0], device=desc1.device)
    mutual = best21[best12] == ids1

    keep = ratio_ok & mutual
    if keep.sum() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=desc1.device)

    return torch.stack([ids1[keep], best12[keep]], dim=1)

def _normalize_points(pts: torch.Tensor):
    mean = pts.mean(dim=0)
    d = torch.sqrt(((pts - mean) ** 2).sum(dim=1) + 1e-8).mean()
    s = torch.sqrt(torch.tensor(2.0, device=pts.device)) / (d + 1e-8)

    t = torch.eye(3, device=pts.device, dtype=pts.dtype)
    t[0, 0] = s
    t[1, 1] = s
    t[0, 2] = -s * mean[0]
    t[1, 2] = -s * mean[1]

    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    pts_n = (t @ pts_h.t()).t()
    return pts_n[:, :2], t

# Helper function to compute homography using DLT
def _dlt_homography(src: torch.Tensor, dst: torch.Tensor):
    if src.shape[0] < 4:
        return None

    src_n, t1 = _normalize_points(src)
    dst_n, t2 = _normalize_points(dst)

    n = src.shape[0]
    a = torch.zeros((2 * n, 9), device=src.device, dtype=src.dtype)

    x = src_n[:, 0]
    y = src_n[:, 1]
    u = dst_n[:, 0]
    v = dst_n[:, 1]

    a[0::2, 0:3] = torch.stack([-x, -y, -torch.ones_like(x)], dim=1)
    a[1::2, 3:6] = torch.stack([-x, -y, -torch.ones_like(x)], dim=1)
    a[0::2, 6:9] = torch.stack([u * x, u * y, u], dim=1)
    a[1::2, 6:9] = torch.stack([v * x, v * y, v], dim=1)

    try:
        _, _, vh = torch.linalg.svd(a)
    except RuntimeError:
        return None

    h = vh[-1].view(3, 3)
    h = torch.linalg.inv(t2) @ h @ t1

    if torch.abs(h[2, 2]) < 1e-8:
        return None

    h = h / h[2, 2]
    return h

def _project_points(h: torch.Tensor, pts: torch.Tensor):
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    warped = (h @ pts_h.t()).t()
    z = warped[:, 2:3].clamp(min=1e-8)
    return warped[:, :2] / z

# Helper function to compute homography using RANSAC
def _ransac_homography(src: torch.Tensor, dst: torch.Tensor, thresh=4.0, iters=1200):
    if src.shape[0] < 4:
        return None, None

    best_h = None
    best_inliers = None
    best_count = 0
    n = src.shape[0]

    for _ in range(iters):
        ids = torch.randperm(n, device=src.device)[:4]
        h = _dlt_homography(src[ids], dst[ids])
        if h is None:
            continue

        pred = _project_points(h, src)
        err = torch.sqrt(((pred - dst) ** 2).sum(dim=1) + 1e-8)
        inliers = err < thresh
        cnt = int(inliers.sum().item())

        if cnt > best_count:
            best_count = cnt
            best_h = h
            best_inliers = inliers

    if best_h is None or best_inliers is None or best_count < 4:
        return None, None

    refined = _dlt_homography(src[best_inliers], dst[best_inliers])
    if refined is not None:
        best_h = refined
        pred = _project_points(best_h, src)
        err = torch.sqrt(((pred - dst) ** 2).sum(dim=1) + 1e-8)
        best_inliers = err < thresh

    return best_h, best_inliers

# Helper function to estimate homography between two images using feature matching and RANSAC.
def _estimate_pairwise_h(img1: torch.Tensor, img2: torch.Tensor):
    pts1, desc1 = _extract_features(img1)
    pts2, desc2 = _extract_features(img2)

    matches = _match_descriptors(desc1, desc2)
    if matches.shape[0] < 8:
        return None, 0, 0.0

    m1 = pts1[matches[:, 0]]
    m2 = pts2[matches[:, 1]]

    h, inliers = _ransac_homography(m1, m2)
    if h is None or inliers is None:
        return None, 0, 0.0

    inlier_count = int(inliers.sum().item())
    inlier_ratio = float(inlier_count / max(matches.shape[0], 1))
    return h, inlier_count, inlier_ratio

def _corners_of(img: torch.Tensor):
    _, h, w = img.shape
    return torch.tensor(
        [[0.0, 0.0],
         [w - 1.0, 0.0],
         [w - 1.0, h - 1.0],
         [0.0, h - 1.0]],
        device=img.device,
        dtype=img.dtype
    )

# Helper function to compute the canvas size and translation transform for a set of images and their homographies.
def _canvas_from_transforms(images, transforms):
    all_pts = []
    for img, h in zip(images, transforms):
        pts = _project_points(h, _corners_of(img))
        all_pts.append(pts)

    all_pts = torch.cat(all_pts, dim=0)
    min_xy = torch.floor(all_pts.min(dim=0).values)
    max_xy = torch.ceil(all_pts.max(dim=0).values)

    tx = -min_xy[0]
    ty = -min_xy[1]

    width = int((max_xy[0] - min_xy[0] + 1).item())
    height = int((max_xy[1] - min_xy[1] + 1).item())

    t = torch.eye(3, device=images[0].device, dtype=images[0].dtype)
    t[0, 2] = tx
    t[1, 2] = ty

    return t, height, width

# Helper function to warp an image
def _warp_image_and_mask(img: torch.Tensor, h: torch.Tensor, out_h: int, out_w: int):
    if img.dim() == 3:
        img_b = img.unsqueeze(0)
    else:
        img_b = img

    h_b = h.unsqueeze(0)
    warped = K.geometry.transform.warp_perspective(
        img_b, h_b, (out_h, out_w), mode='bilinear', padding_mode='zeros', align_corners=True
    )

    mask = torch.ones((1, 1, img.shape[1], img.shape[2]), device=img.device, dtype=img.dtype)
    warped_mask = K.geometry.transform.warp_perspective(
        mask, h_b, (out_h, out_w), mode='nearest', padding_mode='zeros', align_corners=True
    )

    return warped[0], warped_mask[0:1]

# Helper function to blend two warped images using their masks
def _blend_pair_for_background(w1, m1, w2, m2):
    both = (m1 > 0.5) & (m2 > 0.5)
    only1 = (m1 > 0.5) & (~both)
    only2 = (m2 > 0.5) & (~both)

    out = torch.zeros_like(w1)
    out[:, only1[0]] = w1[:, only1[0]]
    out[:, only2[0]] = w2[:, only2[0]]

    if both.any():
        g1 = _gray(w1.unsqueeze(0))
        g2 = _gray(w2.unsqueeze(0))
        diff = (g1 - g2).abs()

        grad1 = K.filters.spatial_gradient(g1)
        grad2 = K.filters.spatial_gradient(g2)
        mag1 = torch.sqrt(grad1[:, :, 0] ** 2 + grad1[:, :, 1] ** 2 + 1e-8)
        mag2 = torch.sqrt(grad2[:, :, 0] ** 2 + grad2[:, :, 1] ** 2 + 1e-8)

        avg = 0.5 * (w1 + w2)
        choose1 = mag1 <= mag2
        choose2 = ~choose1

        overlap_hard = both & (diff > 0.12)
        overlap_soft = both & (~overlap_hard)

        out[:, overlap_soft[0, 0]] = avg[:, overlap_soft[0, 0]]
        out[:, overlap_hard[0, 0] & choose1[0, 0]] = w1[:, overlap_hard[0, 0] & choose1[0, 0]]
        out[:, overlap_hard[0, 0] & choose2[0, 0]] = w2[:, overlap_hard[0, 0] & choose2[0, 0]]

    return out



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
    names = sorted(list(imgs.keys()))
    if len(names) == 0:
        return img

    img1_raw = imgs[names[0]]
    if len(names) == 1:
        img = img1_raw
        return img

    img2_raw = imgs[names[1]]

    img1, scale1 = _to_float01(img1_raw)
    img2, scale2 = _to_float01(img2_raw)
    scale_back = scale1 if scale1 >= scale2 else scale2

    h12, inliers, ratio = _estimate_pairwise_h(img1, img2)

    if h12 is None or inliers < 10:
        out_h = max(img1.shape[1], img2.shape[1])
        out_w = img1.shape[2] + img2.shape[2]
        img = torch.zeros((3, out_h, out_w), device=img1.device, dtype=img1.dtype)
        img[:, :img1.shape[1], :img1.shape[2]] = img1
        img[:, :img2.shape[1], img1.shape[2]:img1.shape[2] + img2.shape[2]] = img2
        return _restore_range(img, scale_back)

    eye = torch.eye(3, device=img1.device, dtype=img1.dtype)
    transforms = [eye, h12]
    shift, out_h, out_w = _canvas_from_transforms([img1, img2], transforms)

    w1, m1 = _warp_image_and_mask(img1, shift @ eye, out_h, out_w)
    w2, m2 = _warp_image_and_mask(img2, shift @ h12, out_h, out_w)

    img = _blend_pair_for_background(w1, m1, w2, m2)
    img = _restore_range(img, scale_back)

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
