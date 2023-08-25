# STARTUP THE CAMERA

## Install drivers (only for 1st installation)

This steps are takes from this [guide](https://husarion.com/tutorials/ros-components/ricoh-theta-z1/). 

### v4l2loopback

First download, build and install v4l2loopback. It will allow you to create virtual loopback camera interfaces.

```
mkdir -p ~/your_ws/src/theta_z1
cd ~/your_ws/src/theta_z1/
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback/
make && sudo make install
sudo depmod -a
```

After successful installation run:

```
ls /dev | grep video
```

You should see your video interfaces.

If you don't have any other cameras installed the output should be empty. To start loopback interface and find it's ID run:

```
sudo modprobe v4l2loopback
ls /dev | grep video
```
New /dev/video device should appear. It's your loopback interface you will later assign to your THETA Z1.

**ATTENTION**: if you get an error like
```
rmmod: ERROR: Module v4l2loopback is not currently loaded
modprobe: ERROR: could not insert 'v4l2loopback': Operation not permitted
```

Try to solve it with
```
sudo apt-get install v4l2loopback-dkms
```
### Ricoh Theta dependencies

Install required packages:

```

sudo apt-get install libgstreamer1.0-0 \
     gstreamer1.0-plugins-base \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     gstreamer1.0-doc \
     gstreamer1.0-tools \
     gstreamer1.0-x \
     gstreamer1.0-alsa \
     gstreamer1.0-gl \
     gstreamer1.0-gtk3 \
     gstreamer1.0-qt5 \
     gstreamer1.0-pulseaudio \
     libgstreamer-plugins-base1.0-dev \
     libjpeg-dev
```

After installation building and install libuvc-theta:

```
cd ~/your_ws/src/theta_z1
git clone https://github.com/ricohapi/libuvc-theta.git
cd libuvc-theta
mkdir build
cd build
cmake ..
make
sudo make install
```

### Installation
```
git clone https://github.com/bach05/libuvc-theta-sample.git
cd gst
make
```

## Run the camera

```
cd your_workspace/src/ricoh_theta_z1/src/libuvc-theta-sample/gst
sh streaming_theta.sh
```
```
export ROS_MASTER_URI=http://192.168.53.2:11311
export ROS_IP=192.168.53.10
rosrun auto_calibration_tools camera.py
```

# CALIBRATION

## Intrinsics

### Collect data
Connect the PC to the robot and the camera. Start up the camera. You can use:
```
roslaunch auto_calibration_tools acquireCalibrationData.launch
```
Be careful to set `SAVE_ROOT`  to `~/AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics/images` in the launch file. 
Move around with the chessboard. Default is a [] chessboard.

### Train the board detector

The chessboard is detected automatically in the image to remap it in the right side. You can easily train a detector for your chessboard

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics
python3 train_board_detector.py
```
The scripts read images from `backgrounds/` folder and `target.png` to train a one-shot detector. 

### Run calibration

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics
python3 splitImage.py
```
This step split each panoramic image in the 6 cube projections. It reads the panoramic images from `/images`

```commandline
python3 calibrateIntrinsics.py
```

## Extrinsics

### Collect data
Connect the PC to the robot and the camera. Start up the camera. You can use:
```
roslaunch auto_calibration_tools acquireCalibrationData.launch
```
Be careful to set `SAVE_ROOT`  to `~/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/images_outdoor` in the launch file. 

Move the robot around the board. We suggest to do this in an empty room or in outdoor to avoid spourious detections. 

# AutoLabeling