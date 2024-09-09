import numpy as np

def topic_to_image(msg):
    img_raw = np.frombuffer(msg.data, dtype=np.uint8)
    img = img_raw.reshape((msg.height, msg.width, 3))
    return img

def topic_to_depth(msg, CAMERA_DATA):
    depth_raw = np.frombuffer(msg.data, dtype=np.uint16)
    depth = depth_raw.reshape((msg.height, msg.width))
    depth = depth / (2**16-1) * (CAMERA_DATA["max_range"]- CAMERA_DATA["min_range"]) + CAMERA_DATA["min_range"]
    return depth
