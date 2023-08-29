from src.util.tracking_utils import *


def tracking(measurements, filters, frame_num, missed_filters, current_id, first_gallery):
    print(measurements)
    if frame_num == 0:
        first_gallery = []
        current_id = 0
        for i in range(0, len(measurements)):
            filter_i = creat_new_filter(measurements[str(i)], i)
            first_gallery.append(measurements[str(i)]['visual_features'][0])
            filters[current_id] = filter_i
            current_id += 1
    else:
        id_d = measurements.keys()
        id_g = [int(key) for key in id_d]
        id_rem = [int(key) for key in id_d]
        attached = []
        positions = []
        galleries = []
        for i in id_g:
            positions.append(measurements[str(i)]['position'][0])
            galleries.append(measurements[str(i)]['visual_features'][0])

        filters, missed_ids, id_rem, id_g, attached, assigned_filters = check_neighbours(
            filters, measurements, id_rem, attached, positions, galleries, id_g
        )

        if len(missed_ids) > 0:
            filters, id_rem, assigned, assigned_filters, missed_id = check_similarity_matrix(
                filters, measurements, id_rem, attached, assigned_filters, missed_ids, galleries, id_g
            )
            filters, missed_filters = add_loss_of_id(filters, missed_ids, missed_filters)

        if len(id_rem) > 0:
            filters, missed_filters, attached, assigned_filters, first_gallery = find_missed_id(
                filters, missed_filters, measurements, galleries,
                attached, id_rem, current_id, first_gallery, assigned_filters
            )
    print('frame number:' + str(frame_num))

    return filters, missed_filters, current_id, first_gallery
