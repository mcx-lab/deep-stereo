import numpy as np

def topic_to_image(msg):
    img_raw = np.frombuffer(msg.data, dtype=np.uint8)
    img = img_raw.reshape((msg.height, msg.width, 3))
    return img

def topic_to_depth(msg, config):
    depth_raw = np.frombuffer(msg.data, dtype=np.uint16)
    depth = depth_raw.reshape((msg.height, msg.width))
    depth = depth / (2**16-1) * (config["camera"]["max_range"]- config["camera"]["min_range"]) + config["camera"]["min_range"]
    return depth

def interpolate_depth(depth_map, points):
    depth_values = np.zeros(len(points))
    for i, point in enumerate(points):
        x, y = point
        diff_x = x - int(x)
        diff_y = y - int(y)
        depth_values[i] = depth_map[int(y), int(x)] * (1-diff_x) * (1-diff_y) + \
                            depth_map[int(y), int(x+1)] * diff_x * (1-diff_y) + \
                            depth_map[int(y+1), int(x)] * (1-diff_x) * diff_y + \
                            depth_map[int(y+1), int(x+1)] * diff_x * diff_y
    return depth_values

def depth_to_pcd(depth_image, intrinsic, ):
    # Get dimensions of the depth image
    height, width = depth_image.shape

    # Generate a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()

    # Calculate 3D coordinates
    fx, fy, cx, cy = intrinsic[0], intrinsic[5], intrinsic[2], intrinsic[6]
    z = depth

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Create a point cloud
    points = np.vstack((x_3d, y_3d, z)).T
    return points