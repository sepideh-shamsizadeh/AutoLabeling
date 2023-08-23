#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import csv
import os
import cv2


class RangeImagePublisher:
    def __init__(self, csv_file_path, image_folder):
        # Initialize ROS node
        rospy.init_node('range_image_publisher', anonymous=True)

        # Create publishers to publish the LaserScan and Image messages
        self.scan_publisher = rospy.Publisher('/scan', LaserScan, queue_size=10)
        self.image_publisher = rospy.Publisher('/theta_camera/image_raw', Image, queue_size=10)

        # Initialize the CvBridge
        self.bridge = CvBridge()

        self.csv_file_path = csv_file_path
        self.image_folder = image_folder
        self.bag_data = {}

        # Define laser specification
        self.laser_spec = {
            'frame_id': "base_link",
            'angle_min': -3.140000104904175,
            'angle_max': 3.140000104904175,
            'angle_increment': 0.005799999926239252,
            'range_min': 0.44999998807907104,
            'range_max': 25.0
        }

        self.read_csv_data()

        self.data_size = len(self.bag_data)
        self.key_list = list(self.bag_data.keys())
        self.pointer = 0


    def read_csv_data(self):
        # Read the CSV file
        with open(self.csv_file_path, mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Skip the header row

            for row in csvreader:
                bag_name = row[0]
                laser_ranges = [float(value) for value in row[1].split(',')]
                self.bag_data[bag_name] = laser_ranges

    def publish_range_and_image(self, bag_name):

        if bag_name in self.bag_data:
            laser_ranges = self.bag_data[bag_name]
            self.laser_ranges = laser_ranges

            # Create a LaserScan message
            scan_msg = LaserScan()
            scan_msg.header.frame_id = self.laser_spec['frame_id']
            scan_msg.angle_min = self.laser_spec['angle_min']
            scan_msg.angle_max = self.laser_spec['angle_max']
            scan_msg.angle_increment = self.laser_spec['angle_increment']
            scan_msg.range_min = self.laser_spec['range_min']
            scan_msg.range_max = self.laser_spec['range_max']
            scan_msg.ranges = laser_ranges

            # Publish the LaserScan message
            self.scan_publisher.publish(scan_msg)

            # Publish the related image
            image_filename = os.path.splitext(bag_name)[0] + '.png'
            image_path = os.path.join(self.image_folder, image_filename)
            if os.path.exists(image_path):
                image_data = cv2.imread(image_path)
                self.image_data = image_data
                image_msg = self.bridge.cv2_to_imgmsg(image_data, encoding="bgr8")
                self.image_publisher.publish(image_msg)
            else:
                rospy.logwarn(f"Image not found: {image_path}")

    def publish_next_scan_image(self):
        if self.pointer < self.data_size:
            bag_name = self.key_list[self.pointer]
            self.publish_range_and_image(bag_name)
            self.pointer = self.pointer+1
            return self.get_current_data(), [self.pointer, self.data_size]
        else:
            print("No more data....")
            return (None, None), None

    def get_current_data(self):
        return self.laser_ranges, self.image_data

    def get_laser_specs(self):
        return self.laser_spec

if __name__ == "__main__":
    csv_file_path = './calibration_data/output.csv'
    image_folder = './calibration_data/images'

    range_image_publisher = RangeImagePublisher(csv_file_path, image_folder)

    cmd = ''
    while cmd.lower() != 'q':
        scan, image = range_image_publisher.publish_next_scan_image()
        cmd = input("Press Enter to publish the next scan message...    [press Q to stop]")
