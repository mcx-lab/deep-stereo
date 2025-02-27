import rosbag
import os
from sensor_msgs.msg import CameraInfo
import json

DESTINATION_DIR = "/scratchdata/processed/desk"

# Open the rosbag file
bag = rosbag.Bag('/scratchdata/desk.bag', 'r')

# Camera Info
camera_info = {}
for topic, msg, t in bag.read_messages(topics=['/camera/color/camera_info']):
    print(msg)
    camera_info["D"] = msg.D
    camera_info["K"] = msg.K
    camera_info["R"] = msg.R
    camera_info["P"] = msg.P
    camera_info["height"] = msg.height
    camera_info["width"] = msg.width
    break
        
# Close the ROS bag
bag.close()

# Save the camera info as a JSON file
with open(os.path.join(DESTINATION_DIR,'camera_info.json'), 'w') as json_file:
    json.dump(camera_info, json_file, indent=4)

print("Camera info has been saved to 'camera_info.json'.")
