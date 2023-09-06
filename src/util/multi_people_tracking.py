from src.util.tracking_utils import *


def tracking(measurements, filters, frame_num, missed_filters, current_id, first_gallery):
    print(measurements)
    # Initialize the filters and galleries with first frame
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
        # First check all neighbours with radius 0.7 meter and visual feature for neighbours with threshold 0.95
        filters, missed_ids, id_rem, id_g, attached, assigned_filters = check_neighbours(
            filters, measurements, id_rem, attached, positions, galleries, id_g
        )

        filters, missed_filters = add_loss_of_id(filters, missed_ids, missed_filters)
        # Here if there be a value in our measurement and didn't assign to our ID's it checked in missed id's
        if len(id_rem) > 0:
            m_id = missed_filters.keys()
            # First check all neighbours with radius 0.7 meter and visual feature for neighbours with threshold 0.95
            missed_filters, missed_ids, id_rem, id_g, attached, assigned_filters = check_neighbours(
                missed_filters, measurements, id_rem, attached, positions, galleries, id_g
            )

            for id in m_id:
                if id not in missed_ids:
                    filters[id] = missed_filters[id]
                    missed_filters.pop(id)
            # for ids in missed_ids:
            #     m_id.remove(ids)
        if len(id_rem) > 0:
            filters, missed_filters, attached, assigned_filters, first_gallery, id_rem = find_missed_id(
                filters, missed_filters, measurements, galleries,
                attached, id_rem, current_id, first_gallery, assigned_filters, 0.5
            )
            # if couldn't find an id for the value it's going to make a new id for that
            for id in id_rem:
                filter_i = creat_new_filter(measurements[str(id)], current_id)
                filters[current_id] = filter_i
                first_gallery[current_id] = measurements[str(id)]['visual_features'][0]
                current_id += 1
    print('frame number:' + str(frame_num))

    return filters, missed_filters, current_id, first_gallery
