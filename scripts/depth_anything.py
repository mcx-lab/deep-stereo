#!/usr/bin/env python
import os
import sys

script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )

import rospy
import numpy as np
import data_conversion
import message_filters
import depth_anything_interface
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
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
        self.depth_topic = self.config["camera"]["depth_topic"]

        self.model = depth_anything_interface.get_model(self.device, self.model_path, self.model_type, self.encoder, self.max_depth)
        
        # Params
        self.image = None
        self.depth = None
        self.camera_intrinsic = None
        self.br = CvBridge()
        self.header = Header()
        self.header.frame_id = "depth_anything_optical_frame"

        # Publishers
        self.camera_pub = rospy.Publisher('depth_anything_camera_info', CameraInfo, queue_size=1)
        self.img_pub = rospy.Publisher('depth_anything_img', Image, queue_size=1)
        self.depth_float32_pub = rospy.Publisher('depth_anything_est_depth_float32', Image, queue_size=1)

    def callback(self,img_sub_msg, depth_sub_msg):

        sta = rospy.get_rostime()

        self.header.stamp = rospy.Time.now()

        self.image = data_conversion.topic_to_image(img_sub_msg)
        gt_depth = data_conversion.topic_to_depth(depth_sub_msg, self.config)
        est_depth = self.model.infer_image(self.image)

        self.depth, _ = depth_anything_interface.get_pred_depth(gt_depth, est_depth, self.config, depth_anything_interface.estimated_depth_model, verbose=False)
      
        msg = self.br.cv2_to_imgmsg(self.depth, "32FC1")
        msg.header = self.header
        self.depth_float32_pub.publish(msg)
        img_sub_msg.header = self.header
        self.img_pub.publish(img_sub_msg)
        self.camera_intrinsic.header = self.header
        self.camera_pub.publish(self.camera_intrinsic)

        rospy.loginfo("Time Taken: {}".format((rospy.get_rostime()-sta).to_sec()))

    def start(self):
        # Grab camera intrinsics
        self.camera_intrinsic = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=5)

        # Subscribers
        img_sub = message_filters.Subscriber(self.camera_topic , Image)
        depth_sub = message_filters.Subscriber(self.depth_topic , Image)

        ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_sub], queue_size=1, slop=0.05)
        ts.registerCallback(self.callback)       

        rospy.spin()
                        
if __name__ == '__main__':
    rospy.init_node("depth_anything", anonymous=True)
    new_node = DepthAnything()
    try:
        new_node.start()    
    except rospy.ROSInterruptException:
        pass
