from PIL import Image
from numpy import clip
from src.util.visual_feature_utils import *
from src.util.laser2image_utils import *
from math import pi, atan2, hypot, floor


def concatenate_person(bound_left, bound_right, img):
    # Get dimensions of the parts
    width_part1, height_part1 = (bound_left[2] - bound_left[0], bound_left[3] - bound_left[1])
    width_part2, height_part2 = (bound_right[2] - bound_right[0], bound_right[3] - bound_right[1])
    x = min(bound_left[1], bound_right[1])
    y = max(bound_left[3], bound_right[3])
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
    # concatenated_image.show()

    return concatenated_image


def get_visual_vector(image, model1, person, flag_concat):
    if flag_concat:
        preprocessed_image = load_and_preprocess_image(image, [0, 0, image.width, image.height])
    else:
        preprocessed_image = load_and_preprocess_image(image, person)
    vect = extract_feature_single(model1, preprocessed_image, "cpu")
    vect_features = vect.view((-1)).numpy()
    return vect_features


def handle_borders(facel, facer, sides_detected):
    facel_pos, facer_pos = [], []
    if len(sides_detected[facer]['positions']) > 0:
        facer_bnd = sides_detected[facer]['bounding_boxes'][0]
        if facer_bnd[0] <= 5:
            facer_pos = sides_detected[facer]['positions'][0]
    if len(sides_detected[facel]['positions']) > 0:
        facel_bnd = sides_detected[facel]['bounding_boxes'][-1]
        if 480 - facel_bnd[2] <= 1:
            facel_pos = sides_detected[facel]['positions'][-1]
    if len(facel_pos) > 0 and len(facer_pos) > 0:
        closest = find_closest_position(facel_pos, facer_pos)
        position = closest
        sides_detected[facer]['bounding_boxes'].pop(0)
        sides_detected[facer]['positions'].pop(0)
        sides_detected[facel]['bounding_boxes'].pop()
        sides_detected[facel]['positions'].pop()
    elif len(facer_pos) > 0:
        sides_detected[facer]['bounding_boxes'].pop(0)
        sides_detected[facer]['positions'].pop(0)
        position = facer_pos
    else:
        sides_detected[facel]['bounding_boxes'].pop()
        sides_detected[facel]['positions'].pop()
        position = facel_pos
    return sides_detected, position


def assign_pose2person(people_detected, j, person, image, pos, model1, flag_concat):
    people_detected[str(j)] = {
        'bounding_box': [],  # Initialize with an empty list
        'position': [],  # Initialize with default values, replace with actual values
        'visual_features': []
    }
    people_detected[str(j)]['bounding_box'].append(person)
    people_detected[str(j)]['position'].append([pos[0], pos[1]])
    vect_features = get_visual_vector(image, model1, person, flag_concat)
    people_detected[str(j)]['visual_features'].append(vect_features)
    people_detected[str(j)]['visual_features'].append(vect_features)
    return people_detected


def from_cube2panoramic(face, bnd):
    inSize = (1920, 960)
    a = 2.0 * float(bnd[0]) / 480
    b = 2.0 * float(bnd[1]) / 480
    if face == 'back':  # back
        (x, y, z) = (-1.0, 1.0 - a, 1.0 - b)
    elif face == 'left':  # left
        (x, y, z) = (a - 1.0, -1.0, 1.0 - b)
    elif face == 'front':  # front
        (x, y, z) = (1.0, a - 1.0, 1.0 - b)
    elif face == 'right':  # right
        (x, y, z) = (1.0 - a, 1.0, 1.0 - b)

    theta = atan2(y, x)  # range -pi to pi
    r = hypot(x, y)
    phi = atan2(z, r)  # range -pi/2 to pi/2

    # source img coords
    uf = 0.5 * 1920 * (theta + pi) / pi
    vf = 0.5 * 1920 * (pi / 2 - phi) / pi
    # Use bilinear interpolation between the four surrounding pixels
    ui = floor(uf)  # coord of pixel to bottom left
    vi = floor(vf)
    A = int(ui % inSize[0]), int(clip(vi, 0, inSize[1] - 1))
    return A


def assign_pose2panoramic(image, org_detected, sides_detected, model1):
    people_detected = {}
    sorted_people = sorted(org_detected, key=lambda
        x: x[0])
    j = 0
    while j < len(sorted_people):
        flag_concat = False
        person = sorted_people[j]
        if 0 <= person[0] < 240:
            if 0 <= person[2] < 240:
                pos = sides_detected['back']['positions'][0]
                bnd = sides_detected['back']['bounding_boxes']
                for i, back in enumerate(bnd):
                    if 0 <= back[0] < 240:
                        if 240 <= back[2] < 480:
                            if 1680 <= sorted_people[-1][0] <= 1920:
                                pos = sides_detected['back']['positions'].pop(i)
                                sides_detected['back']['bounding_boxes'].pop(i)
                                flag_concat = True
                if flag_concat:
                    bnnd = sorted_people.pop()
                    img = concatenate_person(bnnd, person, image)
                    people_detected = assign_pose2person(
                        people_detected, j, person, img, pos, model1, flag_concat
                    )
                    j += 1
                else:
                    people_detected = assign_pose2person(
                        people_detected, j, person, image, pos, model1, flag_concat
                    )
                    sides_detected['back']['positions'].pop(0)
                    sides_detected['back']['bounding_boxes'].pop(0)
                    j += 1
            elif 240 < person[2] < 720:
                sides_detected, pos = handle_borders('back', 'left', sides_detected)
                people_detected = assign_pose2person(
                    people_detected, j, person, image, pos, model1, flag_concat
                )
                j += 1
        elif 240 <= person[0] < 720:
            if 240 <= person[2] < 720:
                if len(sides_detected['left']['bounding_boxes']) > 0:
                    bnd = sides_detected['left']['bounding_boxes'][0]
                    pos = sides_detected['left']['positions'][0]
                    people_detected = assign_pose2person(
                        people_detected, j, person, image, pos, model1, flag_concat
                    )
                    sides_detected['left']['positions'].pop(0)
                    sides_detected['left']['bounding_boxes'].pop(0)
                j += 1
            elif 720 <= person[2] < 1200:
                sides_detected, pos = handle_borders('left', 'front', sides_detected)
                people_detected = assign_pose2person(
                    people_detected, j, person, image, pos, model1, flag_concat
                )
                j += 1
        elif 720 <= person[0] < 1200:
            if 720 <= person[2] < 1200:
                if len(sides_detected['front']['bounding_boxes']) > 0:
                    bnd = sides_detected['front']['bounding_boxes'][0]
                    pos = sides_detected['front']['positions'][0]

                    people_detected = assign_pose2person(
                        people_detected, j, person, image, pos, model1, flag_concat
                    )
                    sides_detected['front']['positions'].pop(0)
                    sides_detected['front']['bounding_boxes'].pop(0)
                j += 1
            elif 1200 <= person[2] < 1680:
                sides_detected, pos = handle_borders('front', 'right', sides_detected)
                people_detected = assign_pose2person(
                    people_detected, j, person, image, pos, model1, flag_concat
                )
                j += 1
        elif 1200 <= person[0] < 1680:
            if 1200 <= person[2] < 1680:
                if len(sides_detected['right']['bounding_boxes']) > 0:
                    bnd = sides_detected['right']['bounding_boxes'][0]
                    pos = sides_detected['right']['positions'][0]

                    people_detected = assign_pose2person(
                        people_detected, j, person, image, pos, model1, flag_concat
                    )
                    sides_detected['right']['positions'].pop(0)
                    sides_detected['right']['bounding_boxes'].pop(0)
                j += 1
            elif 1680 <= person[2] < 1920:
                sides_detected, pos = handle_borders('right', 'back', sides_detected)
                people_detected = assign_pose2person(
                    people_detected, j, person, image, pos, model1, flag_concat
                )
                j += 1
        elif 1680 <= person[0] <= 1920:
            if len(sides_detected['back']['bounding_boxes']) > 0:
                bnd = sides_detected['back']['bounding_boxes'][0]
                pos = sides_detected['back']['positions'][0]
                people_detected = assign_pose2person(
                    people_detected, j, person, image, pos, model1, flag_concat
                )
                sides_detected['back']['positions'].pop(0)
                sides_detected['back']['bounding_boxes'].pop(0)
            j += 1
    return people_detected
