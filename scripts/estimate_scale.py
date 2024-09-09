#!/usr/bin/env python
import os
import sys

script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )

import rospy
import numpy as np
import data_conversion
import depth_anything_interface
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class EstimateScale(object):
    def __init__(self):
        
        self.config = rospy.get_param('/depth_anything_config')

        self.camera_topic = self.config["camera"]["depth_topic"]
        
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.pub = rospy.Publisher('depth_anything_scale', Image,queue_size=10)

        est_depth_sub = message_filters.Subscriber("/depth_anything_est_depth", Image)
        gt_depth_sub = message_filters.Subscriber(self.camera_topic , Image)
        
        ts = message_filters.TimeSynchronizer([est_depth_sub, gt_depth_sub], 10)
        ts.registerCallback(self.callback)

    def callback(self, msg):
        self.image = data_conversion.topic_to_image(msg)
        self.depth = self.model.infer_image(self.image)

    def start(self):
        rospy.spin()
                        
if __name__ == '__main__':
    rospy.init_node("depth_anything", anonymous=True)
    new_node = DepthAnything()
    new_node.start()