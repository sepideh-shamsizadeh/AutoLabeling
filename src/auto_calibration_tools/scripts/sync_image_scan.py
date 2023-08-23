import rospy
import cv2
import csv
import numpy as np
from sensor_msgs.msg import Image, LaserScan
import time
import os

rospy.init_node('sync_image_scan', anonymous=True)

# Retrieve parameters
SAVE_ROOT = rospy.get_param("~SAVE_ROOT", "./calibration_data_extrinsics/images")
num_images_to_save = rospy.get_param("~num_images_to_save", 100)
save_interval = rospy.get_param("~save_interval", 2)  # Adjust the interval as needed (in seconds)

if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)
    print(f"Folder '{SAVE_ROOT}' created.")
else:
    print(f"Folder '{SAVE_ROOT}' already exists.")


image_msg = None
scan_msg = None
image_counter = 0
save_interval = 2.0  # Save data every 1 second
# Create a ROS rate object
rate = rospy.Rate(1.0 / save_interval)
images = []
f_names = []
scans = []

def sync_callback(image_msg, scan_msg):
    global image_counter

    # Access the image and scan messages with the same timestamp
    image_time = image_msg.header.stamp
    scan_time = scan_msg.header.stamp

    # Convert image data to a NumPy array
    image_data = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width, -1))

    # Save image
    f_names.append(os.path.join(SAVE_ROOT,f"image_{image_counter}.jpg"))
    images.append(image_data)
    scans.append(scan_msg)

    # Print the image and scan timestamps
    print("Image timestamp:", image_time)
    print("Scan timestamp:", scan_time)
    print("Image ID:", image_counter)

    image_counter += 1

    return image_data, scan_msg

def image_callback(msg):
    global image_msg
    image_msg = msg

def scan_callback(msg):
    global scan_msg
    scan_msg = msg

rospy.Subscriber('/theta_camera/image_raw', Image, image_callback)
rospy.Subscriber('/scan', LaserScan, scan_callback)

previous_time = time.time()
cont = 0
while not rospy.is_shutdown() and image_counter < num_images_to_save:
    
    current_time = time.time()
    elapsed_time = current_time - previous_time

    # Save data
    if image_msg is not None and scan_msg is not None:
        sync_callback(image_msg, scan_msg)
        cont +=1

    print(f"[Elapsed time: {elapsed_time}] Acquired {cont} pictures...")
    rate.sleep()  # Use ROS rate to control loop frequency

#rospy.spin()
print(len(images), len(scans))
# Save scan with image ID
scan_filename = os.path.join(SAVE_ROOT, f"scan.csv")
with open(scan_filename, 'a') as csvfile:
    for imgc, scan_msg in enumerate(scans):
        writer = csv.writer(csvfile)
        writer.writerow([image_counter] + list(scan_msg.ranges))

for i, img in enumerate(images):
    cv2.imwrite(f_names[i], img)
