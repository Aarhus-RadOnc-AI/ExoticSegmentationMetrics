from pprint import pprint

import SimpleITK as sitk
import numpy as np
from numba import njit  # For speed


# @njit
def find_contour_z_boundaries(path: str,
                              label_int: int):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img == label_int)

    contour_range = []
    for z in range(arr.shape[0]):
        if bool(np.count_nonzero(arr[z, :, :])):
            contour_range.append(z)

    return {
        "path": path,
        "contour_range": contour_range,
        "top": contour_range[-1],
        "bottom": contour_range[0]}



if __name__ == "__main__":
    p = "/home/mathis/Documents/Studies/1_bounds/OAR_bounds/ESTRO_OAR_interactive/model_eval/investigative_model/contours/Task5041_OARBoundsMergedDSCTOnly/test/pred/HNCDL_005&20150427.nii.gz"
    for i in range(5):
        print(i)
        res = find_contour_z_boundaries(path=p, label_int=i)
        pprint(res)
