#!/usr/bin/env python
import os
import sys

script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )

import rospy
import data_conversion
import depth_anything_interface

config = rospy.get_param('/depth_anything_config')
DEVICE = "cuda"
MODEL_PATH = config["model"]["model_path"]
model_type = config["model"]["model_type"]
encoder = config["model"]["model_size"]
max_depth = 20.0

camera_topic = config["camera"]["rgb_topic"]

model = depth_anything_interface.get_model(DEVICE, MODEL_PATH, model_type, encoder, max_depth)

def callback(data):
    print(data)

def get_est_depth():
    rospy.init_node('depth_anything', anonymous=True)
    rospy.Subscriber(camera_topic, Image, callback)
    rospy.spin()

if __name__ == '__main__':
    get_est_depth()