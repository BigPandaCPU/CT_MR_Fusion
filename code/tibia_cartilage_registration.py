# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import vtk
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

from vtk_tools import getPointsFromSTL
from vtk_tools import createPointsActor
from vtk_tools import showActors
from vtk_tools import createSpineAxisActor
from vtk_tools import createPolyDataFromSTL
from vtk_tools import createActorFromPolydata
from vtk_tools import saveSTLFileFromPolyData


def vectorCross(v1, v2):
    """
    func:计算两个向量的叉乘
    :param v1: numpy array,xyz
    :param v2: numpy array,xyz
    :return:
    """
    a1 = v1[0]
    b1 = v1[1]
    c1 = v1[2]

    a2 = v2[0]
    b2 = v2[1]
    c2 = v2[2]

    return np.array([b1*c2-b2*c1, c1*a2-a1*c2, a1*b2-a2*b1])


def PCA(data, sort=True):
    center = np.mean(data, axis=0)
    # 将点云平移到原点
    aligned_point_cloud = data - center
    H = np.cov(aligned_point_cloud, rowvar=False)
    eigenvalues,eigenvectors = np.linalg.eig(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]

        #特征向量是按照列进行存储的
        eigenvectors = eigenvectors[:, sort]

    #将特征向量统一到右手坐标系
    eigenvectors[:, 2] = vectorCross(eigenvectors[:, 0], eigenvectors[:, 1])
    return eigenvalues, eigenvectors, center


def prealigned_two_point_clouds(source_points, target_points, N=16):
    from sklearn.neighbors import NearestNeighbors
    """
    func：将source_points对齐到target_points
    算法描述: A.首先将中心归零，计算点云的中心点，然后减去这个中心点
            B.利用PCA算法分别计算每个点云的pca轴，按照特征值由大到小进行排列
            C.将source和target点云分别变换到对应的pca轴下，此时就实现了点云的粗对齐
            source_points_aligned = (source_points - source_center)*source_vectors
            target_points_aligned = (target_points - target_center)*target_vectors
            D.进一步验证对齐是否正确。（将两个pca坐标系对齐之后，由于存在正反向的问题，需要进一步验证是否真正对齐）
            在x、y、z这三个方向上，分别变换方向，然后再计算两个点云之间的最小距离，取距离最小的那个变换T
            E.得到最终的对齐结果
            source_points_aligned_final = source_points_aligned * (target_vectors.T) * T+target_canter
    :param source_points:
    :param target_points:
    :return:
    """
    # num_source_points = source_points_vtk.GetNumberOfPoints()
    # source_points = np.zeros((num_source_points, 3))
    # # 将vtkPoints中的点复制到NumPy数组
    # for i in range(num_source_points):
    #     source_points[i] = source_points_vtk.GetPoint(i)
    #
    # num_target_points = target_points_vtk.GetNumberOfPoints()
    # target_points = np.zeros((num_target_points, 3))
    # # 将vtkPoints中的点复制到NumPy数组
    # for i in range(num_target_points):
    #     target_points[i] = target_points_vtk.GetPoint(i)

    # source_points_actors = createPointsActor(source_points,radius=0.5, opacity=0.8, color='red')
    # target_points_actors = createPointsActor(target_points, radius=0.5, opacity=0.8, color='green')
    # all_actors = []
    # all_actors.extend(source_points_actors)
    # all_actors.extend(target_points_actors)
    #showActors(all_actors)


    source_eigen_values, source_eigen_vectors, source_center = PCA(source_points, sort=True)
    target_eigen_values, target_eigen_vectors, target_center = PCA(target_points, sort=True)
    print("source center:", source_center)
    print("target center:", target_center)

    source_points_decenter = source_points - source_center
    target_points_decenter = target_points - target_center
    prealigned_source_points = np.dot(source_points_decenter, source_eigen_vectors)
    prealigned_target_points = np.dot(target_points_decenter, target_eigen_vectors)

    source_points_actors = createPointsActor(prealigned_source_points, 0.3, 1.0, "yellow")
    target_points_actors = createPointsActor(prealigned_target_points, 0.3, 1.0, "red")
    all_actors = []
    all_actors.extend(source_points_actors)
    #all_actors.extend(target_points_actors)

    # source_points_axis_actors = createSpineAxisActor([0, 0, 0], source_eigen_vectors, len=100)
    # all_actors.extend(source_points_axis_actors)

    source_points_axis_actors2 = createSpineAxisActor([0, 0, 0], source_eigen_vectors.T, len=200)
    all_actors.extend(source_points_axis_actors2)
    # showActors(all_actors)



    max_target = np.max(prealigned_target_points, axis=0) - np.min(prealigned_target_points, axis=0)

    R = np.zeros([N, 3, 3])
    T = np.sqrt(2.0) / 2.0
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0, ], [0.0, 0.0, 1.0]])
    R1 = np.array([[T, T, 0.0], [-T, T, 0.0, ], [0.0, 0.0, 1.0]]) # 45度
    R2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, ], [0.0, 0.0, 1.0]]) # 90度
    R3 = np.array([[-T, T, 0.0], [-T, -T, 0.0, ], [0.0, 0.0, 1.0]])  #135度
    R4 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0, ], [0.0, 0.0, 1.0]]) #180度
    R5 = np.array([[-T, -T, 0.0], [T, -T, 0.0], [0.0, 0.0, 1.0]]) #225度
    R6 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  #270度
    R7 = np.array([[T, -T, 0.0], [T, T, 0.0], [0.0, 0.0, 1.0]])  #315度

    R8 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0, ], [0.0, 0.0, -1.0]])  # 反向0度
    R9 = np.array([[T, T, 0.0], [T, -T, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转45度
    R10 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转90度
    R11 = np.array([[T, -T, 0.0], [-T, -T, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转135度
    R12 = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转180度
    R13 = np.array([[-T, -T, 0.0], [-T, T, 0.0, ], [0.0, 0.0, -1.0]])  #旋转225度
    R14 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转270度
    R15 = np.array([[-T, T, 0.0], [T, T, 0.0, ], [0.0, 0.0, -1.0]])  # 旋转315度

    if N == 8:
        R = np.array([R0,R2, R4, R6, R8,R10, R12, R14])
    else:
        R = np.array([R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15])

    k = 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(prealigned_target_points)

    distances = np.zeros([N])

    for i in range(N):
        T = R[i]
        prealigned_source_points_T = np.dot(prealigned_source_points, T)

        max_source = np.max(prealigned_source_points_T, axis=0) - np.min(prealigned_source_points_T, axis=0)
        D = max_target / max_source
        D_new = np.array([
            [D[0], 0, 0],
            [0, D[1], 0],
            [0, 0, D[2]],
        ])

        prealigned_source_points_T = np.dot(prealigned_source_points_T, D_new)


        dis, indices = nbrs.kneighbors(prealigned_source_points_T)
        distances[i] = np.sum(dis)

    index = np.argmin(distances)
    Trans = R[index]

    matrix1 = np.eye(4)
    matrix1[0:3, 3] = -source_center

    print("\nmatrix1:")
    print(matrix1)

    matrix2 = np.eye(4)
    matrix2[0:3, 0:3] = source_eigen_vectors.T

    print("\nmatrix2:")
    print(matrix2)


    matrix3 = np.eye(4)
    matrix3[0:3, 0:3] = Trans
    print("\nmatrix3:")
    print(matrix3)

    matrix4 = np.eye(4)
    matrix4[0:3, 0:3] = target_eigen_vectors

    print("\nmatrix4:")
    print(matrix4)

    matrix5 = np.eye(4)
    matrix5[0:3, 3] = target_center

    print("\nmatrix5:")
    print(matrix5)

    #matrix_np = np.dot(matrix5, np.dot(matrix4, np.dot(matrix3, np.dot(matrix2, matrix1))))
    matrix_np = np.dot(np.dot(np.dot(np.dot(matrix5, matrix4), matrix3), matrix2), matrix1)
   #这里构造左乘的矩阵

    matrix_vtk = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix_vtk.SetElement(i, j, matrix_np[i, j])

    # source_points_new = np.dot(np.dot(np.dot(source_points, source_eigen_vectors), Trans), target_eigen_vectors.T) + target_center
    # source_points_new_actors = createPointsActor(source_points_new, radius=0.5, opacity=0.8, color='red')
    # all_actors.extend(source_points_new_actors)
    #
    # showActors(all_actors)
    return matrix_vtk


def ICP(source_poly_data, target_poly_data):
    """
    func:使用ICP算法将source_poly_data配准到target_poly_data
    """
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly_data)
    icp.SetTarget(target_poly_data)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    #icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    icp_matrix = icp.GetMatrix()
    return icp_matrix


def registration_two_stl(src_stl_file, tar_stl_file, num_points=5000):
    """

    :return:
    """
    source_points = getPointsFromSTL(src_stl_file, num_points)
    target_points = getPointsFromSTL(tar_stl_file, num_points)

    source_polydata = createPolyDataFromSTL(src_stl_file)
    target_polydata = createPolyDataFromSTL(tar_stl_file)

    prealligned_matrix_vtk = prealigned_two_point_clouds(source_points, target_points, N=16)

    prealligned_trans = vtkTransform()
    prealligned_trans.SetMatrix(prealligned_matrix_vtk)
    prealligned_transform = vtkTransformPolyDataFilter()

    prealligned_transform.SetTransform(prealligned_trans)
    prealligned_transform.SetInputData(source_polydata)
    prealligned_transform.Update()

    icp_matrix_vtk = ICP(prealligned_transform.GetOutput(), target_polydata)
    final_matrix_vtk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(icp_matrix_vtk, prealligned_matrix_vtk, final_matrix_vtk)

    #final_matrix_np = np.array(coarse_matrix_vtk.GetData()).reshape(4, 4)
    return final_matrix_vtk, target_polydata

def rotate_image_data(input_image, target_image, rot_matrix4x4):
    """
    """
    ###############  方法1：AffineTransform    ##################
    # rot_matrix = rot_matrix4x4[:3, :3]
    # translation = rot_matrix4x4[0:3, 3]
    # forward_transform = sitk.AffineTransform(3)
    # forward_transform.SetMatrix(rot_matrix.flatten().tolist())
    # forward_transform.SetTranslation(translation.tolist())
    #
    # # 3. 计算反向变换 (Fixed -> Moving)
    # # 重采样管线需要从目标(Fixed)像素点去找源(Moving)像素点
    # inverse_transform = forward_transform.GetInverse()
    #
    # # 4. 配置重采样器
    # resampler = sitk.ResampleImageFilter()
    #
    # # 设定目标图像的空间几何信息（大小、分辨率、原点、方向）
    # resampler.SetReferenceImage(target_image)
    #
    # # 设定变换关系与插值方式
    # resampler.SetTransform(inverse_transform)
    # resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值，若为标签/Mask请用 sitkNearestNeighbor
    #
    # # 设定超出边界时的默认填充值（通常为背景色 0）
    # resampler.SetDefaultPixelValue(0)
    #
    # # 5. 执行变换
    # transformed_img = resampler.Execute(input_image)
    # return transformed_img
    ########################################################

    ############  方法二： Eular3DTransform    ###############
    rot_matrix3x3 = rot_matrix4x4[:3, :3]
    matrix_flat = np.array(rot_matrix3x3).flatten().tolist()

    transform = sitk.Euler3DTransform()
    transform.SetMatrix(matrix_flat)

    translation = rot_matrix4x4[0:3, 3]
    transform.SetTranslation(translation)

    inverse_transform = transform.GetInverse()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(inverse_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    resampler.SetOutputOrigin(target_image.GetOrigin())
    resampler.SetOutputSpacing(target_image.GetSpacing())
    resampler.SetOutputDirection(target_image.GetDirection())
    resampler.SetSize(target_image.GetSize())

    rotated_image = resampler.Execute(input_image)
    ##########################################################
    return rotated_image


if __name__ == "__main__":
    src_stl_file = "../data/MR/stl/LTibia.stl"
    tar_stl_file = "../data/CT/stl/LTibia.stl"
    tar_cartilage_stl_file = "../data/CT/stl/LTibia_cartilage.stl"
    #tar_cartilage_stl_file2 = "../data/CT/stl/LTibia_outside_cartilage.stl"

    src_cartilage_file = "../data/MR/stl/LTibia_inside_cartilage.stl"
    src_cartilage_file2 = "../data/MR/stl/LTibia_outside_cartilage.stl"
    src_cartilage_boundary_file = "../data/MR/stl/LTibia_cartilage_boundary_18.stl"

    src_mr_file = r'..\data\MR\img.nii.gz'
    tar_ct_file = r'..\data\CT\img.nii.gz'

    src_mr_itk = sitk.ReadImage(src_mr_file)
    tar_ct_itk = sitk.ReadImage(tar_ct_file)



    source_cartilage_polydata = createPolyDataFromSTL(src_cartilage_file)
    source_cartilage_polydata2 = createPolyDataFromSTL(src_cartilage_file2)
    source_cartilage_boundary_polydata = createPolyDataFromSTL(src_cartilage_boundary_file)


    coarse_matrix_vtk, target_polydata = registration_two_stl(src_stl_file, tar_stl_file, 5000)


    #########################  coarse transform   ##############################
    coarse_trans = vtkTransform()
    coarse_trans.SetMatrix(coarse_matrix_vtk)
    coarse_transform = vtkTransformPolyDataFilter()

    coarse_transform.SetTransform(coarse_trans)
    coarse_transform.SetInputData(source_cartilage_boundary_polydata)
    coarse_transform.Update()

    # source_coarse_trans_actor = createActorFromPolydata(coarse_transform.GetOutput(), opacity=0.9, color='green')
    # coarse_transform2 = vtkTransformPolyDataFilter()
    # coarse_transform2.SetTransform(coarse_trans)
    # coarse_transform2.SetInputData(source_cartilage_polydata2)
    # coarse_transform2.Update()
    # source_coarse_trans_actor2 = createActorFromPolydata(coarse_transform2.GetOutput(), opacity=0.9, color='green')
    # saveSTLFileFromPolyData(tar_cartilage_stl_file2, final_transform2.GetOutput())
    ############################################################################

    # #############################  fine tune  ##############################
    transform_cartilage_boundary_polydata = coarse_transform.GetOutput()
    icp_matrix_vtk_fine_tune = ICP(transform_cartilage_boundary_polydata, target_polydata)

    final_trans = vtkTransform()
    final_trans.SetMatrix(icp_matrix_vtk_fine_tune)

    final_transform = vtkTransformPolyDataFilter()
    final_transform.SetTransform(final_trans)
    final_transform.SetInputData(coarse_transform.GetOutput())
    final_transform.Update()

    # source_final_actor = createActorFromPolydata(final_transform.GetOutput(), opacity=1.0, color='blue')
    # saveSTLFileFromPolyData(tar_cartilage_stl_file, final_transform.GetOutput())
    # #######################################################################

    ##############################  get final  matrix   #####################
    final_matrix_vtk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(icp_matrix_vtk_fine_tune, coarse_matrix_vtk, final_matrix_vtk)
    #########################################################################

    #############################   rotate mr img   ##########################
    final_matrix_vtk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(icp_matrix_vtk_fine_tune, coarse_matrix_vtk, final_matrix_vtk)
    final_matrix_np = np.array(final_matrix_vtk.GetData()).reshape(4, 4)

    rotated_mr_itk = rotate_image_data(src_mr_itk, tar_ct_itk, final_matrix_np)

    save_mr_file = r'..\data\MR\img_rot_tibia.nii.gz'
    sitk.WriteImage(rotated_mr_itk, save_mr_file)
    ##########################################################################

    #############################    show actors  ##############################

    trans = vtkTransform()
    trans.SetMatrix(final_matrix_vtk)

    transform = vtkTransformPolyDataFilter()
    transform.SetTransform(trans)
    transform.SetInputData(source_cartilage_polydata)
    transform.Update()

    transform2 = vtkTransformPolyDataFilter()
    transform2.SetTransform(trans)
    transform2.SetInputData(source_cartilage_polydata2)
    transform2.Update()

    source_coarse_trans_actor = createActorFromPolydata(transform.GetOutput(), color='red')
    source_coarse_trans_actor2 = createActorFromPolydata(transform2.GetOutput(), color='blue')

    ##################   save transformed tibia cartilage   #################

    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(transform.GetOutput())
    append_filter.AddInputData(transform2.GetOutput())
    append_filter.Update()  # 执行合并
    combined_polydata = append_filter.GetOutput()
    saveSTLFileFromPolyData(tar_cartilage_stl_file, combined_polydata)
    #########################################################################

    all_actors = []
    target_actor = createActorFromPolydata(target_polydata, opacity=0.8)
    all_actors.append(target_actor)

    all_actors.append(source_coarse_trans_actor)
    all_actors.append(source_coarse_trans_actor2)
    showActors(all_actors)
    ############################################################################
