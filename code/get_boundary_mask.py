import os
import numpy as np
import SimpleITK as sitk

connectivity_6 = [
    # 6 个面邻居
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
]

connectivity_18 = [
    # 6 个面邻居
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    # 12 个棱邻居
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)
]

connectivity_26 = [
    (dx, dy, dz)
    for dx in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dz in [-1, 0, 1]
    if not (dx == 0 and dy == 0 and dz == 0)
]

connectivity_map = {
    "connectivity_6": connectivity_6,
    "connectivity_18": connectivity_18,
    "connectivity_26": connectivity_26
}


if __name__ == "__main__":
    src_mask_file = r'..\data\MR\mask.nii.gz'

    part = 18
    offsets = connectivity_map["connectivity_%d"%part]
    mask_itk = sitk.ReadImage(src_mask_file)
    mask_np = sitk.GetArrayFromImage(mask_itk)

    label_femur = 1
    label_femur_cartilage = 2
    idxs_femur = np.where(mask_np == label_femur)
    idxs_femur_cartilage = np.where(mask_np == label_femur_cartilage)

    mask_np_femur = np.zeros_like(mask_np)
    mask_np_femur[idxs_femur] = 1

    mask_np_femur_cartilage = np.zeros_like(mask_np)
    mask_np_femur_cartilage[idxs_femur_cartilage] = 1
    sizeZ, sizeY, sizeX = mask_np.shape

    mask_np_boundary = np.zeros_like(mask_np)

    for i in range(len(idxs_femur_cartilage[0])):
        curZ = idxs_femur_cartilage[0][i]
        curY = idxs_femur_cartilage[1][i]
        curX = idxs_femur_cartilage[2][i]

        if curX >1 and (curX < sizeX-1) and curY > 1 and (curY < sizeY-1) and curZ >1 and (curZ < sizeZ -1):
            cur_mask = mask_np_femur[curZ-1:curZ+2, curY-1:curY+2, curX-1:curX+2]

            cur_sum = 0.0
            for dx, dy, dz in offsets:
                nx, ny, nz = curX + dx, curY + dy, curZ + dz
                cur_sum += mask_np_femur[nz, ny, nx]
            if cur_sum > 0:
                mask_np_boundary[curZ, curY, curX] = 2

    mask_itk_boundary = sitk.GetImageFromArray(mask_np_boundary)
    mask_itk_boundary.CopyInformation(mask_itk)

    save_mask_file = r'..\data\MR\mask_femur_cartilage_boundary_%d.nii.gz'%part
    sitk.WriteImage(mask_itk_boundary, save_mask_file)





