import SimpleITK as sitk
import numpy as np
from numba import njit  # For speed


@njit
def get_edge_of_mask(mask: np.ndarray) -> np.ndarray:
    """
    Mask must only contain 0 for background and 1 for mask.
    :param mask:
    :return:
    """
    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                sum = np.sum(mask[z, y - 1:y + 2, x - 1:x + 2])
                if sum < 9:
                    edge[z, y, x] = mask[z, y, x]
    return edge


def calculate_added_path_length(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
    gt_edge = get_edge_of_mask(sitk.GetArrayFromImage(gt_image == label_int))
    pred_edge = get_edge_of_mask(sitk.GetArrayFromImage(pred_image == label_int))

    if np.count_nonzero(gt_edge) == 0:      # Edge case if prediction is all false and should not be.
                                            # If so, return full size of prediction
        apl = np.count_nonzero(pred_edge)
    else:
        apl = (gt_edge > pred_edge).astype(int).sum()  # Regular APL

    return apl
