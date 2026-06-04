import os

import numpy as np
from vtk_tools import *

def convertNII2STL(nii_file, save_stl_dir):
    nii_dir, nii_name = os.path.split(nii_file)
    shot_name = nii_name.replace(".nii.gz", "")
    cur_file_path = nii_file
    img_array, origin, spacing, direction = getImageFromNII(cur_file_path)
    unique_values = np.unique(img_array)

    # print(origin)
    # print(spacing)
    # print(direction)
    # print(unique_values)

    tmp_array = np.zeros_like(img_array)
    for i in range(len(unique_values)):
        cur_value = unique_values[i]
        tmp_array[:] = 0
        if cur_value < 1:
            continue

        save_stl_file = os.path.join(save_stl_dir, shot_name + "_label_%02d.stl" % cur_value)
        if os.path.exists(save_stl_file):
            continue

        cur_idx = np.where(img_array == cur_value)
        tmp_array[cur_idx] = cur_value

        cur_polydata_normal, cur_polydata = createPolyDataNormalsFromArray(tmp_array, spacing, origin, get_largest_connect_region=False)

        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.FeatureEdgesOff()
        featureEdges.BoundaryEdgesOn()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.SetInputData(cur_polydata)
        featureEdges.Update()

        numberOfOpenEdges = featureEdges.GetOutput().GetNumberOfCells()
        # if numberOfOpenEdges > 0:
        #     print("The label is partially clipped because of the size of dicom, "
        #           "coordinate creation for this kind of label might be incorrect")
        #     continue

        saveSTLFile(save_stl_file, cur_polydata_normal)
        #print(save_stl_file_path, " saved done!")
    print(nii_name)


if __name__ == '__main__':

    nii_file = r'..\data\MR\mask_femur_cartilage_boundary_18.nii.gz'

    save_stl_dir = r'..\data\MR\stl'
    os.makedirs(save_stl_dir, exist_ok=True)
    convertNII2STL(nii_file, save_stl_dir)
    # from multiprocessing import Pool
    # list_args = []
    # nii_dir = "/media/hurwa/data3/DeepSpineData/VerSe20/normal/verse_labels"
    # out_dir = "/media/hurwa/data3/DeepSpineData/VerSe20/normal/verse_stl"
    # os.makedirs(out_dir, exist_ok=True)
    # for nii_name in os.listdir(nii_dir):
    #     nii_file = os.path.join(nii_dir, nii_name)
    #     list_args.append([nii_file, out_dir])
    #
    # num_thread = 8
    # p = Pool(num_thread)
    # p.starmap_async(convertNII2STL, list_args)
    # p.close()
    # p.join()


