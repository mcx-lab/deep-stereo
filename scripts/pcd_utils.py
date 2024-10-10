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

def pcd_matching_tf(ori, new, epsilon, trials, verbose = False):
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

    tf = np.eye(4)

    # Translate by -mean_new
    tf[:3,3] = -mean_new.T

    # Rescale by scale_new
    new_tf = np.eye(4)
    new_tf[3,3] = scale_new
    tf = new_tf @ tf
    
    # Rotate by best_rotation
    new_tf = np.eye(4)
    new_tf[:3,:3] = best_rotation
    tf = new_tf @ tf

    # Scale by inverse of scale_ori
    new_tf = np.eye(4)
    new_tf[3,3] = 1/scale_ori
    tf = new_tf @ tf

    # Translate by +mean_ori
    new_tf = np.eye(4)
    new_tf[:3,3] = mean_ori.T
    tf = new_tf @ tf

    return tf

def ply_from_1x4_coord(coord, filename, color = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord[:,:3]/coord[:,3][:,None])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(filename, pcd)
