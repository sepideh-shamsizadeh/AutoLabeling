import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class CNNChannelExtractor:

    def __init__(self, params_dir, use_only_first_layer=False):
        self.params_dir = params_dir
        self.use_only_first_layer = use_only_first_layer
        self.subnet = self._create_subnet()
        self.num_channels = self._get_num_channels()
        self._load_weights()

    def _create_subnet(self):
        # Define the CNN architecture similar to AhmedSubnet in the C++ code
        input_shape = (160, 60, 3)  # Assuming the input image size is 160x60x3 (RGB)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(25, (5, 5), activation='relu'),
        ])
        return model

    def _load_weights(self):
        # Load weights for conv_t1
        conv_t1_weights = load_weights_from_file(os.path.join(self.params_dir, 'conv_t1.txt'),
                                                 20, 5, 3)
        # Set the weights for conv_t1
        conv_t1_layer = self.subnet.layers[0]
        conv_t1_weights = np.transpose(conv_t1_weights, (1, 2, 3, 0))
        conv_t1_layer.set_weights([conv_t1_weights, np.zeros(20)])

        # Load weights for conv_t2
        conv_t2_weights = load_weights_from_file(os.path.join(self.params_dir, 'conv_t2.txt'),
                                                 25, 5, 20)
        # Set the weights for conv_t2
        conv_t2_layer = self.subnet.layers[2]
        conv_t2_weights = np.transpose(conv_t2_weights, (1, 2, 3, 0))
        conv_t2_layer.set_weights([conv_t2_weights, np.zeros(25)])

    def _get_num_channels(self):
        return self.subnet.layers[-1].output_shape[-1]

    def extract(self, bgr_image, gray_image):
        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Resize the image to the input size expected by the CNN
        resized_image = cv2.resize(rgb_image, (60, 160))
        # Normalize the image pixel values to [0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0

        # Expand the image dimensions to match the CNN input shape
        input_image = np.expand_dims(normalized_image, axis=0)

        # Perform forward pass
        feature_maps = self.subnet.predict(input_image)

        if self.use_only_first_layer:
            output_layer = self.subnet.layers[2].output
            intermediate_model = tf.keras.Model(inputs=self.subnet.input, outputs=output_layer)
            feature_maps = intermediate_model.predict(input_image)

        return feature_maps

    def num_channels(self):
        return self.num_channels

    def channel_names(self):
        names = [f'layer{i}' for i in range(self.num_channels)]
        return names


def load_weights_from_file(file_path, num_filters, kernel_size, num_channels):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    values = [float(val) for val in lines[1].split()]
    weights = np.array(values).reshape(num_filters, kernel_size, kernel_size, num_channels)
    return weights


if __name__ == "__main__":
    params_dir = "data/"  # Replace this with the directory containing the parameter files
    use_only_first_layer = False
    channel_extractor = CNNChannelExtractor(params_dir, use_only_first_layer)

    # Load the BGR image using OpenCV
    bgr_image = cv2.imread("data/n01-01.jpg")  # Replace this with the path to your input image

    # Since the original code expects a grayscale image as well, let's convert the BGR image to grayscale
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Extract feature maps using the CNNChannelExtractor
    feature_maps = channel_extractor.extract(bgr_image, gray_image)

    # print("Number of channels:", channel_extractor.num_channels())
    print("Channel names:", channel_extractor.channel_names())

    # Access individual feature maps
    for i, feature_map in enumerate(feature_maps[0]):
        # Display the first feature map as an example
        print(np.shape(feature_map))
    #     cv2.imshow(f'Feature Map {i}', feature_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Combine feature maps by summing them along the third dimension
    combined_feature_map = np.sum(feature_maps[0], axis=-1)

    # Normalize the combined feature map for visualization
    combined_feature_map -= np.min(combined_feature_map)
    combined_feature_map /= np.max(combined_feature_map)

    # Display the combined feature map
    plt.imshow(combined_feature_map, cmap='viridis')  # Using 'viridis' colormap for better visualization
    plt.axis('off')
    print(np.shape(combined_feature_map))
    plt.title('Combined Feature Map (Sum)')
    plt.show()
