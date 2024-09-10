#!/usr/bin/env python
import os
import sys

script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )

import rospy
import numpy as np
import data_conversion
import depth_anything_interface
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthAnything(object):
    def __init__(self):
        
        self.config = rospy.get_param('/depth_anything_config')
        self.device = "cuda"
        self.model_path = self.config["model"]["model_path"]
        self.model_type = self.config["model"]["model_type"]
        self.encoder = self.config["model"]["model_size"]
        self.max_depth = 20.0

        self.camera_topic = self.config["camera"]["rgb_topic"]

        self.model = depth_anything_interface.get_model(self.device, self.model_path, self.model_type, self.encoder, self.max_depth)
        
        # Params
        self.image = None
        self.depth = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.pub = rospy.Publisher('depth_anything_est_depth', Image,queue_size=1)

        # Subscribers
        rospy.Subscriber(self.camera_topic,Image,self.callback)

    def callback(self, msg):
        self.image = data_conversion.topic_to_image(msg)
        self.depth = self.model.infer_image(self.image)

    def start(self):
                                                                                                
        self.time = rospy.get_rostime().to_sec()

        while not rospy.is_shutdown():
            
            if self.depth is not None:
                self.pub.publish(self.br.cv2_to_imgmsg(self.depth))
                self.depth = None
                rospy.loginfo("Time Taken: {}".format(rospy.get_rostime().to_sec()-self.time))
                self.time = rospy.get_rostime().to_sec()
            self.loop_rate.sleep()
                        
if __name__ == '__main__':
    rospy.init_node("depth_anything", anonymous=True)
    new_node = DepthAnything()
    new_node.start()