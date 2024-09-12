# deep-stereo

This repo creates a ROS package to apply the Depth Anything Model to a camera node.

## Create Docker container

`sudo docker build -t deep-stereo .`

`docker run 
-it
-v <local directory>:/scratchdata
--gpus all
--shm-size 16g 
-d 
--network=host 
--restart unless-stopped 
--env="DISPLAY" 
--env="QT_X11_NO_MITSHM=1" 
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  
--device=/dev/ttyUSB0 
-e DISPLAY=unix$DISPLAY 
--privileged 
deep-stereo`
