import os
import cv2
import rosbag
import csv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Replace these with your actual paths
input_folder = '/home/sepid/workspace/Thesis/GuidingRobot/aa/'
output_csv = 'output.csv'
output_image_folder = 'images'

# Create the output image folder if it doesn't exist
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)

# List all bag files in the input folder
bag_files = [f for f in os.listdir(input_folder) if f.endswith('.bag')]

# Initialize the CvBridge
bridge = CvBridge()

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header row
    csvwriter.writerow(['Bag Name', 'Laser Range (second 4)'])

    # Iterate through bag files
    for bag_file in bag_files:
        bag_path = os.path.join(input_folder, bag_file)
        try:
            # Open the bag file
            with rosbag.Bag(bag_path, 'r') as bag:
                # Initialize variables to hold laser range data
                laser_ranges = []

                # Initialize variable to hold image data
                image_data = None

                # Iterate through messages in the bag
                for topic, msg, t in bag.read_messages(topics=['/scan', '/theta_camera/image_raw']):
                    if topic == '/scan':
                        # Assuming msg.ranges contains the laser range data
                        laser_ranges = msg.ranges

                        # Only consider the laser scan data from second 4
                        if t.to_sec() >= 4 and t.to_sec() < 8:
                            laser_ranges = msg.ranges
                    elif topic == '/theta_camera/image_raw':
                        # Convert the ROS Image message to OpenCV image
                        image_data = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                # Write data to CSV row
                csvwriter.writerow([bag_file, ','.join(map(str, laser_ranges))])

                # Save the image if available
                if image_data is not None:
                    image_filename = os.path.splitext(bag_file)[0] + '.png'
                    image_path = os.path.join(output_image_folder, image_filename)
                    cv2.imwrite(image_path, image_data)

        except rosbag.ROSBagUnindexedException:
            print("The bag is unindexed.")

print("Data extraction, CSV writing, and image saving complete.")
