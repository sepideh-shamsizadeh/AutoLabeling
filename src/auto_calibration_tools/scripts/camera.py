#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2

from cv_bridge import CvBridge, CvBridgeError

cv_bridge = CvBridge()


def to_ros_image(cv2_uint8_image, img_format="bgr"):
    # -- Check input.
    shape = cv2_uint8_image.shape  # (row, col, depth=3)
    #print(shape)
    assert (len(shape) == 3 and shape[2] == 3)

    # -- Convert image to bgr format.
    if img_format == "rgb":  # If rgb, convert to bgr.
        #print('rgb')
        bgr_image = cv2.cvtColor(cv2_uint8_image, cv2.COLOR_RGB2BGR)
    elif img_format == "bgr":
        #print('****')
        bgr_image = cv2_uint8_image
    else:
        raise RuntimeError("Wrong image format: " + img_format)

    # -- Convert to ROS format.
    ros_image = cv_bridge.cv2_to_imgmsg(bgr_image, "bgr8")
    return ros_image


rospy.init_node('VideoPublisher', anonymous=True)

pub = rospy.Publisher('/theta_camera/image_raw', Image, queue_size=5)
# Open the device at the ID 0
# Use the camera ID based on
# /dev/videoID needed
cap = cv2.VideoCapture(5)

# Check if camera was opened correctly
if not (cap.isOpened()):
    print("Could not open video device")
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print("Auto fps: ", fps)
    rate = rospy.Rate(fps)

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Capture frame-by-frame
while (True):
    ret, frame = cap.read()

    # Display the resulting frame

    #cv2.imshow("preview", frame)
    #cv2.imwrite("outputImage.jpg", frame)
    img_msg = to_ros_image(frame)
    img_msg.header.stamp = rospy.Time.now()
    img_msg.header.frame_id = 'camera'
    pub.publish(img_msg)
    # Waits for a user input to quit the application

# When everything done, release the capture
cap.release()
