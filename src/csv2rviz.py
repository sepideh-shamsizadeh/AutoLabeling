import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import csv
import os
import cv2

# Initialize ROS node
rospy.init_node('range_image_publisher', anonymous=True)

# Create publishers to publish the LaserScan and Image messages
scan_publisher = rospy.Publisher('/scan', LaserScan, queue_size=10)
image_publisher = rospy.Publisher('/theta_camera/image_raw', Image, queue_size=10)

# Create a LaserScan message
scan_msg = LaserScan()

# Set the necessary fields of the LaserScan message
scan_msg.header.frame_id = "base_link"  # Set the appropriate frame ID
scan_msg.angle_min = -3.140000104904175  # Set the minimum angle
scan_msg.angle_max = 3.140000104904175  # Set the maximum angle
scan_msg.angle_increment = 0.005799999926239252  # Set the angle increment
scan_msg.range_min = 0.44999998807907104  # Set the minimum range value
scan_msg.range_max = 25.0  # Set the maximum range value

# Initialize the CvBridge
bridge = CvBridge()

# Path to the folder containing images
image_folder = './calibration_data/images'  # Replace with your image folder path

csv_file_path = './calibration_data/output.csv'

# Dictionary to store bag names as keys and laser range data as values
bag_data = {}

# Read the CSV file
with open(csv_file_path, mode='r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)  # Skip the header row

    for row in csvreader:
        bag_name = row[0]
        laser_ranges = [float(value) for value in row[1].split(',')]
        bag_data[bag_name] = laser_ranges

        # Set the range values in the LaserScan message
        scan_msg.ranges = laser_ranges

        # Publish the LaserScan message
        scan_publisher.publish(scan_msg)

        # Publish the related image
        image_filename = os.path.splitext(bag_name)[0] + '.png'
        image_path = os.path.join(image_folder, image_filename)
        if os.path.exists(image_path):
            image_data = cv2.imread(image_path)
            image_msg = bridge.cv2_to_imgmsg(image_data, encoding="bgr8")
            image_publisher.publish(image_msg)
        else:
            rospy.logwarn(f"Image not found: {image_path}")

        input("Press Enter to publish the next scan message...")
