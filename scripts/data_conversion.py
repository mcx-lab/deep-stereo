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
