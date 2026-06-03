#! /usr/bin/env python3

#  Copyright (c) Thiago Franco de Moraes
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import vtk
import numpy as np
import SimpleITK as sitk
from vtkmodules.all import vtkClipClosedSurface, vtkCutter
from vtkmodules.all import vtkClipPolyData
from vtkmodules.all import vtkDataSetMapper
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)

from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes,
    vtkWindowedSincPolyDataFilter,
    vtkPolyDataNormals,
)

from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR
from vtkmodules.all import vtkOBBTree, vtkSmoothPolyDataFilter
from vtkmodules.all import vtkCylinderSource
from vtkmodules.all import vtkPolyDataMapper, vtkActor
from vtkmodules.all import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkInteractorStyleTrackballCamera
import math
import sys

import vtk
import SimpleITK as sitk
#from SpineSeg.SpineSurgicalPlanning import SpineLabelDict_int2str

# index2label={1:"C1", 2:"C2", 3:"C3", 4:"C4", 5:"C5", 6:"C6", 7:"C7", 8:"T1", 9:"T2", 10:"T3",
#              11:"T4", 12:"T5", 13:"T6", 14:"T7", 15:"T8", 16:"T9", 17:"T10", 18:"T11", 19:"T12",
#              20:"L1", 21:"L2", 22:"L3", 23:"L4", 24:"L5", 25:"L6", 26:"Sacrum" }


VTKColorLegend_int2list =\
{   0:     [0,   0,   0],
    1:     [255, 0,   0],
    2:     [0,   255, 0],
    3:     [0,   0,   255],
    4:     [255, 255, 0],
    5:     [0,   255, 255],
    6:     [255, 0,   255],
    7:     [255, 239, 213],
    8:     [0,   0,   205],
    9:     [205, 133, 63],
   10:     [210, 180, 140],
   11:     [102, 205, 170],
   12:     [0,   0,   128],
   13:     [0,   139, 139],
   14:     [46,  139, 87],
   15:     [255, 228, 225],
   16:     [106, 90,  205],
   17:     [221, 160, 221],
   18:     [233, 150, 122],
   19:     [165, 42,  42],
   20:     [255, 250, 250],
   21:     [147, 112, 219],
   22:     [218, 112, 214],
   23:     [75,  0,   130],
   24:     [255, 182, 193],
   25:     [60,  179, 113],
   26:     [255, 205, 235],
}


def numpy2VTK(img, spacing=[1.0, 1.0, 1.0], origin=[0, 0, 0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tobytes()  # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(origin[0], origin[1], origin[2])

    return importer

def createPolyDataFromSTL(stl_file):
    """
    :param stl_file:
    :return:
    """
    reader = vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data

def createPolyDataNormalsFromArray(img_array, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], use_flying_edges=False):
    """

    :param img_array:
    :param spacing:
    :param origin:
    :param use_flying_edges:
    :return:
    """

    importer = numpy2VTK(img_array, spacing=spacing, origin=origin)

    if not use_flying_edges:
        try:
            skin_extractor = vtkFlyingEdges3D()

        except AttributeError:
            skin_extractor = vtkMarchingCubes()
    else:
        skin_extractor = vtkMarchingCubes()

        # femur process #
    skin_extractor.ComputeGradientsOff()
    skin_extractor.ComputeNormalsOff()
    skin_extractor.SetInputConnection(importer.GetOutputPort())
    skin_extractor.SetValue(0, 1)
    skin_extractor.Update()

    smooth = vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(skin_extractor.GetOutput())
    smooth.SetNumberOfIterations(20)
    pass_band = 0.001
    smooth.SetPassBand(pass_band)
    #smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOn()
    smooth.Update()
    normal_gen = vtkPolyDataNormals()
    normal_gen.ConsistencyOn()  # discreate marching cubes may generate inconsistent surface
    # we almost always perform smoothing, so aplitting would not be able to preserve any sharp features
    # (and sharp edges would look like artifacts in the smooth surface).
    # normal_gen.ComputePointNormalsOn()
    # normal_gen.ComputeCellNormalsOn()
    normal_gen.SplittingOff()
    normal_gen.SetInputData(smooth.GetOutput())
    normal_gen.Update()

    return normal_gen, smooth.GetOutput()

def createActorFromMask(mask, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], color=[0, 0, 0], opacity=1.0):
    """
    :param stl_file:
    :return:
    """
    colors = vtkNamedColors()
    cur_polydata_normal = createPolyDataNormalsFromArray(mask, spacing, origin)

    # smooth_filter = vtkSmoothPolyDataFilter()
    # smooth_filter.SetInputConnection(cur_polydata_normal.GetOutputPort())
    # smooth_filter.SetNumberOfIterations(3)
    #
    # smooth_filter.SetRelaxationFactor(0.5)
    # smooth_filter.FeatureEdgeSmoothingOff()
    # smooth_filter.BoundarySmoothingOff()


    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cur_polydata_normal.GetOutputPort())
    mapper.ScalarVisibilityOff()
    cur_color = np.array(color)/255.0
    #cur_color = colors.GetColor3d('red')

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(cur_color[0], cur_color[1], cur_color[2])
    actor.GetProperty().SetOpacity(opacity)
    return actor


def showActors(actors, window_name=""):
    ren = vtkRenderer()
    for cur_actor in actors:
        ren.AddActor(cur_actor)

    win = vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetWindowName("show spine "+window_name)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    ren.ResetCamera()
    win.Render()
    iren.Initialize()
    iren.Start()


def saveSTLFile(save_stl_file_name, poly_data_normals):
    """

    :param save_stl_file_name:
    :param poly_data_normals:
    :return:
    """
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(save_stl_file_name)
    stl_writer.SetInputConnection(poly_data_normals.GetOutputPort())
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()
    stl_writer.Update()


# def saveSTLFile(save_stl_file_name, poly_data_normals):
#     """
#
#     :param save_stl_file_name:
#     :param poly_data_normals:
#     :return:
#     """
#     stl_writer = vtk.vtkSTLWriter()
#     stl_writer.SetFileName(save_stl_file_name)
#     stl_writer.SetInputConnection(poly_data_normals.GetOutputPort())
#     stl_writer.SetFileTypeToBinary()
#     stl_writer.Write()
#     stl_writer.Update()

def read_mesh_file(filename):
    if filename.lower().endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif filename.lower().endswith(".ply"):
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError("Only reads STL and PLY")
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def polydata_to_imagedata(polydata, dimensions=(100, 100, 100), spacings=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    from vtk.util import numpy_support
    #xi, xf, yi, yf, zi, zf = polydata.GetBounds()
    dx, dy, dz = dimensions

    # Calculating spacing
    sx = spacings[0] #(xf - xi) / dx
    sy = spacings[1] #(yf - yi) / dy
    sz = spacings[2] #(zf - zi) / dz

    # Calculating Origin
    ox = origin[0] #xi + sx / 2.0
    oy = origin[1] #yi + sy / 2.0
    oz = origin[2] #zi + sz / 2.0


    image = vtk.vtkImageData()
    image.SetSpacing((sx, sy, sz))
    image.SetDimensions((dx, dy, dz))
    image.SetExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    image.SetOrigin((ox, oy, oz))

    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    inval = 255
    outval = 0

    # for i in range(image.GetNumberOfPoints()):
    #     image.GetPointData().GetScalars().SetTuple1(i, inval)

    data_np = np.ones(dimensions, dtype=np.uint8) * inval

    vtk_array = numpy_support.numpy_to_vtk(data_np.ravel(), array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(vtk_array)

    # # 将NumPy数组转换为vtkDoubleArray
    # vtk_data = vtk.vtkUnsignedCharArray()
    # vtk_data.SetNumberOfComponents(1)
    # vtk_data.SetArray(data_np, n_points * data_np.dtype.itemsize)
    #
    # # 设置标量数据到vtkImageData对象
    # image.GetPointData().SetScalars(vtk_data)


    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin((ox, oy, oz))
    pol2stenc.SetOutputSpacing((sx, sy, sz))
    pol2stenc.SetOutputWholeExtent(image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()


# def save(imagedata, filename):
#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(filename)
#     writer.SetInputData(imagedata)
#     writer.Write()


def save(imagedata, filename):
    # writer = vtk.vtkXMLImageDataWriter()
    # writer.SetFileName(filename)
    # writer.SetInputData(imagedata)
    # writer.Write()
    writer = vtk.vtkNIFTIImageWriter()
    #writer = vtk.vtkTeemNRRDWriter()

    # 设置输入数据（假设 imgstenc 是一个已经配置好的算法）
    # 在 Python 中，通常使用 SetInputConnection 而不是直接设置 Output
    writer.SetInputData(imagedata)

    # 设置输出文件名
    writer.SetFileName(filename)

    # 执行写入操作
    writer.Write()

def convertNII2NRRD(src_nii_file, dst_nrrd_file):
    mask_itk = sitk.ReadImage(src_nii_file)
    sitk.WriteImage(mask_itk, dst_nrrd_file, useCompression=True)



def main():
    input_filename = r'E:\DeepData\CTMR\CT\stl\LFemur_cartilage_new_5000_icp_18.stl'
    output_filename = r'E:\DeepData\CTMR\CT\mask\LFemurCartilage.nii.gz'

    aim_mask_file = r'E:\DeepData\CTMR\CT\mask\mask_graphcut.nii.gz'
    aim_mask_itk = sitk.ReadImage(aim_mask_file)
    aim_mask_np = sitk.GetArrayFromImage(aim_mask_itk)
    dimension = aim_mask_np.shape[::-1]
    spacing = aim_mask_itk.GetSpacing()
    origin = aim_mask_itk.GetOrigin()

    polydata = read_mesh_file(input_filename)
    imagedata = polydata_to_imagedata(polydata, dimension, spacing, origin)
    save(imagedata, output_filename)



if __name__ == "__main__":
    main()

    # src_mask_file = r'E:\DeepData\CTMR\CT\mask\LFemurCartilage.nii'
    # tar_mask_file = r'E:\DeepData\CTMR\CT\mask\mask_graphcut.nii.gz'
    #
    # src_mask_itk = sitk.ReadImage(src_mask_file)
    # src_mask_np = sitk.GetArrayFromImage(src_mask_itk)
    # src_idxs = np.where(src_mask_np > 0)
    #
    # tar_mask_itk = sitk.ReadImage(tar_mask_file)
    # tar_mask_np = sitk.GetArrayFromImage(tar_mask_itk)
    # tar_idxs_femur = np.where(tar_mask_np == 1)
    # tar_mask_np[src_idxs] = 5
    # # tar_mask_np[tar_idxs_femur] = 1
    #
    # save_mask_file = r'E:\DeepData\CTMR\CT\mask\mask_graphcut_cartilage.nii.gz'
    # save_mask_itk = sitk.GetImageFromArray(tar_mask_np)
    # save_mask_itk.CopyInformation(tar_mask_itk)
    #
    # sitk.WriteImage(save_mask_itk, save_mask_file)