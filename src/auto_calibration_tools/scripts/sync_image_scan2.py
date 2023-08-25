#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv

class DataSaverNode:

    def __init__(self):
        rospy.init_node('data_saver_node', anonymous=True)

        # Retrieve parameters
        SAVE_ROOT = rospy.get_param("~SAVE_ROOT", "./calibration_data_extrinsics/images")
        num_images_to_save = rospy.get_param("~num_images_to_save", 100)
        save_interval = rospy.get_param("~save_interval", 1)  # Adjust the interval as needed (in seconds)

        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
            print(f"Folder '{SAVE_ROOT}' created.")
        else:
            print(f"Folder '{SAVE_ROOT}' already exists.")

        self.save_period = save_interval  # Seconds
        self.save_root = SAVE_ROOT
        self.max_data = num_images_to_save

        # Initialize variables to store data
        self.image_data = None
        self.scan_data = None

        # Create subscribers for the topics
        self.image_sub = Subscriber("/theta_camera/image_raw", Image)
        self.scan_sub = Subscriber("/scan", LaserScan)

        # Synchronize the messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.scan_sub], queue_size=30, slop=400.0)
        self.sync.registerCallback(self.callback)

        # Create a timer to save data periodically
        self.save_timer = rospy.Timer(rospy.Duration(self.save_period), self.save_data)

        self.image_count = 0
        self.scan_filename = os.path.join(SAVE_ROOT, f"scan.csv")

        # Check if the file exists before attempting to delete it
        if os.path.exists(self.scan_filename):
            os.remove(self.scan_filename)
            print(f"File '{self.scan_filename}' deleted successfully.")
        else:
            print(f"File '{self.scan_filename}' does not exist.")

    def callback(self, image_msg, scan_msg):
        # Store the received data
        self.image_data = image_msg
        self.scan_data = scan_msg
        print("Data received...")

    def save_data(self, event):
        if self.image_data is not None and self.scan_data is not None:
            # Perform data processing here if needed
            # For example, you can save the image as an OpenCV image
            cv_image = self.convert_image_message_to_opencv(self.image_data)
            
            # Save data to a file (e.g., image and scan data)
            timestamp = rospy.get_time()
            image_filename = f"image_{self.image_count}.png"
            self.image_count +=1
            
            # Save the image
            save_path = os.path.join(self.save_root,image_filename)
            cv2.imwrite(save_path, cv_image)


            with open(self.scan_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_filename] + list(self.scan_data.ranges))

            rospy.loginfo(f"Data saved to {image_filename} and {self.scan_filename}")

            if self.image_count > self.max_data:
                rospy.signal_shutdown("Acquired {} images, stopping...".format(self.image_count))


    def convert_image_message_to_opencv(self, image_msg):
        # Convert ROS Image message to an OpenCV image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        return cv_image

if __name__ == '__main__':

    try:
        node = DataSaverNode()
        print("Start data acquisition!")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
