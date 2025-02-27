import numpy as np
import open3d as o3d

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def pcd_transform(ori, new, epsilon, trials, verbose = False):
    mean_ori = np.mean(ori, axis=0)
    translated_ori = ori - mean_ori
    scale_ori = (np.linalg.norm(translated_ori) ** 2 / len(translated_ori)) ** 0.5
    rescaled_ori = translated_ori / scale_ori

    mean_new = np.mean(new, axis=0)
    translated_new = new - mean_new
    scale_new = (np.linalg.norm(translated_new) ** 2 / len(translated_new)) ** 0.5
    rescaled_new = translated_new / scale_new

    # Find the best rotation matrix
    best_rotation = np.eye(3)
    best_inliers = 0

    for _ in range(trials):
        i = np.random.randint(len(rescaled_ori))

        rotation = rotation_matrix_from_vectors(rescaled_new[i], rescaled_ori[i])

        dist = rescaled_ori - rescaled_new @ rotation.T

        inliers = np.mean(np.linalg.norm(dist, axis=1) < epsilon)

        if inliers > best_inliers:
            best_inliers = inliers
            best_rotation = rotation

    if verbose:
        print(f'Inliers Ratio: {best_inliers}')

    return best_rotation

def pcd_matching_tf(coord1, coord2, epsilon, trials, ransac_sample=4, verbose = False):
    best_inliers = 0
    best_tf = np.eye(4)

    for _ in range(trials):
        sample_indices = np.random.choice(len(coord1), ransac_sample, replace=False)
        sample1 = coord1[sample_indices]
        sample2 = coord2[sample_indices]

        # Zero Mean
        sample1_mean = np.mean(sample1, axis=0)
        sample2_mean = np.mean(sample2, axis=0)

        sample1 = sample1 - sample1_mean
        sample2 = sample2 - sample2_mean

        # SVD
        H = sample2.T @ sample1
        try:
            U, D, V_T = np.linalg.svd(H)
        except:
            continue
        
        R = V_T.T @ U.T
        sample2 = sample2 @ R.T

        scale = np.einsum('ij,ij->', sample1, sample2) / np.einsum('ij,ij->', sample2, sample2)
        #print(np.einsum('ij,ij->', sample1, sample2), np.einsum('ij,ij->', sample2, sample2))
        
        if scale < 1e-5:
            continue
            
        test_coord2 = (coord2 - sample2_mean) @ R.T * scale + sample1_mean

        error = np.linalg.norm(test_coord2 - coord1, axis=1)

        inliers = np.mean(error < epsilon)

        if (inliers > best_inliers):
            best_inliers = inliers

            tf = np.eye(4)
            tf[:3,3] = -sample2_mean.T

            new_tf = np.eye(4)
            new_tf[3,3] = 1/scale
            tf = new_tf @ tf

            new_tf = np.eye(4)
            new_tf[:3,:3] = R
            tf = new_tf @ tf

            new_tf = np.eye(4)
            new_tf[:3,3] = sample1_mean.T
            tf = new_tf @ tf

            best_tf = tf

    if verbose:
        print(f'Inliers Ratio: {best_inliers}')

    return best_tf

def ply_from_1x4_coord(coord, filename, color = None, voxel_size = 0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord[:,:3]/coord[:,3][:,None])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
                                                        std_ratio=1.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    #o3d.io.write_point_cloud(filename, pcd)
    o3d.io.write_point_cloud(filename, pcd)
