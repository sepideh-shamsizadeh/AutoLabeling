from PIL import Image
from src.util.visual_feature_utils import *
from src.util.laser2image_utils import *

def concatenate_person(bound_left, bound_right, img):
    # Get dimensions of the parts
    width_part1, height_part1 = (bound_left[2] - bound_left[0], bound_left[3] - bound_left[1])
    width_part2, height_part2 = (bound_right[2] - bound_right[0], bound_right[3] - bound_right[1])
    x = min(bound_left[1], bound_right[1])
    y = max(bound_left[3], bound_right[3])
    print(x, y)
    # Crop the image parts using the bounding box coordinates
    part1 = img.crop([bound_left[0], x, bound_left[2], y])
    part2 = img.crop([bound_right[0], x, bound_right[2], y])

    # Determine the dimensions of the concatenated image
    total_width = width_part1 + width_part2
    max_height = y - x

    # Create a new blank image with the calculated dimensions
    concatenated_image = Image.new('RGB', (total_width, max_height))

    # Paste the parts onto the concatenated image
    concatenated_image.paste(part1, (0, 0))
    concatenated_image.paste(part2, (width_part1, 0))
    concatenated_image.show()

    return concatenated_image


def assign_pose2panoramic(image, org_detected, sides_detected, model1):
    print('++++++++++++++++++++++++++++++++++')
    galleries = []
    print(org_detected)
    print(sides_detected)
    people_detected = {}
    sorted_people = sorted(org_detected, key=lambda
        x: x[0])
    print(sorted_people)
    j = 0
    sorted_positions = []
    print(len(sorted_people))
    while j < len(sorted_people):
        if 0 <= sorted_people[j][0] < 240:
            bounding_boxes = sides_detected['back']['bounding_boxes']
            positions = sides_detected['back']['positions']
            if len(positions) > 0:
                if 0 <= sorted_people[j][2] < 240:
                    bnd = sides_detected['back']['bounding_boxes'][0]
                    pos = sides_detected['back']['positions'][0]
                    if 0 <= bnd[0] < 480:
                        if 240 <= bnd[2] < 480:
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            bnnd = sorted_people.pop()
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append([pos[0], pos[1]])
                            people_detected[str(j)]['bounding_box'].append(bnnd)
                            img = concatenate_person(bnnd, sorted_people[j], image)
                            preprocessed_image = load_and_preprocess_image(img, sorted_people[j])
                            vect = extract_feature_single(model1, preprocessed_image, "cpu")
                            vect_features = vect.view((-1)).numpy()
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            galleries.append(vect_features)
                            sorted_positions.append([pos[0], pos[1]])
                            sides_detected['back']['positions'].pop(0)
                            j += 1
                    elif 240 <= bnd[0] < 480:
                        if 240 <= bnd[2] <= 480:
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                            vect = extract_feature_single(model1, preprocessed_image, "cpu")
                            vect_features = vect.view((-1)).numpy()
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            galleries.append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append([pos[0], pos[1]])
                            sorted_positions.append([pos[0], pos[1]])
                            sides_detected['back']['positions'].pop(0)
                            j += 1
                elif 240 < sorted_people[j][2] < 720:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    if len(sides_detected['left']['positions']) > 0:
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        back_pos = sides_detected['back']['positions'].pop()
                        left_pos = sides_detected['left']['positions'].pop(0)
                        pos = find_closest_position(back_pos, left_pos)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
                    else:
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        pos = sides_detected['back']['positions'].pop()
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
            elif 240 < sorted_people[j][2] < 720:
                people_detected[str(j)] = {
                    'bounding_box': [],  # Initialize with an empty list
                    'position': [],  # Initialize with default values, replace with actual values
                    'visual_features': []
                }
                preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                vect = extract_feature_single(model1, preprocessed_image, "cpu")
                vect_features = vect.view((-1)).numpy()
                people_detected[str(j)]['visual_features'].append(vect_features)
                galleries.append(vect_features)
                pos = sides_detected['left']['positions'].pop(0)
                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                people_detected[str(j)]['position'].append([pos[0], pos[1]])
                sorted_positions.append([pos[0], pos[1]])
                j += 1
        elif 240 <= sorted_people[j][0] < 720:
            bounding_boxes = sides_detected['left']['bounding_boxes']
            positions = sides_detected['left']['positions']
            if len(positions) > 0:
                bnd = sides_detected['left']['bounding_boxes'][0]
                pos = sides_detected['left']['positions'][0]
                if 240 <= sorted_people[j][2] < 720:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                    vect = extract_feature_single(model1, preprocessed_image, "cpu")
                    vect_features = vect.view((-1)).numpy()
                    people_detected[str(j)]['visual_features'].append(vect_features)
                    galleries.append(vect_features)
                    people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                    people_detected[str(j)]['position'].append([pos[0], pos[1]])
                    sorted_positions.append([pos[0], pos[1]])
                    sides_detected['left']['positions'].pop(0)
                    j += 1
                elif 720 <= sorted_people[j][2] < 1200:
                    if len(sides_detected['front']['positions']) > 0:
                        left_pos = sides_detected['left']['positions'].pop()
                        front_pos = sides_detected['front']['positions'].pop(0)
                        pos = find_closest_position(front_pos, left_pos)
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
                    else:
                        pos = sides_detected['left']['positions'].pop()
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
            elif 720 <= sorted_people[j][2] < 1200:
                pos = sides_detected['front']['positions'].pop(0)
                people_detected[str(j)] = {
                    'bounding_box': [],  # Initialize with an empty list
                    'position': [],  # Initialize with default values, replace with actual values
                    'visual_features': []
                }
                preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                vect = extract_feature_single(model1, preprocessed_image, "cpu")
                vect_features = vect.view((-1)).numpy()
                people_detected[str(j)]['visual_features'].append(vect_features)
                galleries.append(vect_features)
                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                people_detected[str(j)]['position'].append([pos[0], pos[1]])
                sorted_positions.append([pos[0], pos[1]])
                j += 1
        elif 720 <= sorted_people[j][0] < 1200:
            bounding_boxes = sides_detected['front']['bounding_boxes']
            positions = sides_detected['front']['positions']
            if len(positions) > 0:
                bnd = sides_detected['front']['bounding_boxes'][0]
                pos = sides_detected['front']['positions'][0]
                if 720 <= sorted_people[j][2] < 1200:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                    vect = extract_feature_single(model1, preprocessed_image, "cpu")
                    vect_features = vect.view((-1)).numpy()
                    people_detected[str(j)]['visual_features'].append(vect_features)
                    galleries.append(vect_features)
                    people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                    people_detected[str(j)]['position'].append([pos[0], pos[1]])
                    sorted_positions.append([pos[0], pos[1]])
                    sides_detected['front']['positions'].pop(0)
                    j += 1
                elif 1200 <= sorted_people[j][2] < 1680:
                    if len(sides_detected['right']['positions']) > 0:
                        front_pos = sides_detected['front']['positions'].pop()
                        right_pos = sides_detected['right']['positions'].pop(0)
                        pos = find_closest_position(front_pos, right_pos)
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
                    else:
                        pos = sides_detected['front']['positions'].pop()
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
            elif 1200 <= sorted_people[j][2] < 1680:
                pos = sides_detected['right']['positions'].pop(0)
                people_detected[str(j)] = {
                    'bounding_box': [],  # Initialize with an empty list
                    'position': [],  # Initialize with default values, replace with actual values
                    'visual_features': []
                }
                preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                vect = extract_feature_single(model1, preprocessed_image, "cpu")
                vect_features = vect.view((-1)).numpy()
                people_detected[str(j)]['visual_features'].append(vect_features)
                galleries.append(vect_features)
                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                people_detected[str(j)]['position'].append([pos[0], pos[1]])
                sorted_positions.append([pos[0], pos[1]])
                j += 1
        elif 1200 <= sorted_people[j][0] < 1680:
            bounding_boxes = sides_detected['right']['bounding_boxes']
            positions = sides_detected['right']['positions']
            if len(positions) > 0:
                bnd = sides_detected['right']['bounding_boxes'][0]
                pos = sides_detected['right']['positions'][0]
                if 1200 <= sorted_people[j][2] < 1680:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                    vect = extract_feature_single(model1, preprocessed_image, "cpu")
                    vect_features = vect.view((-1)).numpy()
                    people_detected[str(j)]['visual_features'].append(vect_features)
                    galleries.append(vect_features)
                    people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                    people_detected[str(j)]['position'].append([pos[0], pos[1]])
                    sorted_positions.append([pos[0], pos[1]])
                    sides_detected['right']['positions'].pop(0)
                    j += 1
                elif 1680 <= sorted_people[j][2] < 1920:
                    if len(sides_detected['back']['positions']) > 0:
                        right_pos = sides_detected['right']['positions'].pop()
                        back_pos = sides_detected['back']['positions'].pop(0)
                        pos = find_closest_position(back_pos, right_pos)
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
                    else:
                        pos = sides_detected['right']['positions'].pop()
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                        vect = extract_feature_single(model1, preprocessed_image, "cpu")
                        vect_features = vect.view((-1)).numpy()
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        galleries.append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append([pos[0], pos[1]])
                        sorted_positions.append([pos[0], pos[1]])
                        j += 1
            elif 1680 <= sorted_people[j][2] < 1920:
                pos = sides_detected['back']['positions'].pop(0)
                people_detected[str(j)] = {
                    'bounding_box': [],  # Initialize with an empty list
                    'position': [],  # Initialize with default values, replace with actual values
                    'visual_features': []
                }
                preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                vect = extract_feature_single(model1, preprocessed_image, "cpu")
                vect_features = vect.view((-1)).numpy()
                people_detected[str(j)]['visual_features'].append(vect_features)
                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                people_detected[str(j)]['position'].append([pos[0], pos[1]])
                sorted_positions.append([pos[0], pos[1]])
                j += 1
        elif 1680 <= sorted_people[j][0] <= 1920:
            bounding_boxes = sides_detected['back']['bounding_boxes']
            positions = sides_detected['back']['positions']
            bnd = sides_detected['back']['bounding_boxes'][0]
            pos = sides_detected['back']['positions'][0]
            if 0 <= bnd[0] < 240:
                people_detected[str(j)] = {
                    'bounding_box': [],  # Initialize with an empty list
                    'position': [],  # Initialize with default values, replace with actual values
                    'visual_features': []
                }
                preprocessed_image = load_and_preprocess_image(image, sorted_people[j])
                vect = extract_feature_single(model1, preprocessed_image, "cpu")
                vect_features = vect.view((-1)).numpy()
                people_detected[str(j)]['visual_features'].append(vect_features)
                galleries.append(vect_features)
                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                people_detected[str(j)]['position'].append([pos[0], pos[1]])
                sorted_positions.append([pos[0], pos[1]])
                sides_detected['back']['positions'].pop(0)
                j += 1
    return people_detected, sorted_positions, galleries
