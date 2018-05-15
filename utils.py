def intersect_area(a, b):  # returns None if rectangles don't intersect
    a_xmax = max(a[0], a[2])
    a_xmin = min(a[0], a[2])
    a_ymax = max(a[1], a[3])
    a_ymin = min(a[1], a[3])

    b_xmax = max(b[0], b[2])
    b_xmin = min(b[0], b[2])
    b_ymax = max(b[1], b[3])
    b_ymin = min(b[1], b[3])

    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def union_area(a, b):
    return area(a) + area(b) - intersect_area(a, b)


def area(a):
    a_xmax = max(a[0], a[2])
    a_xmin = min(a[0], a[2])
    a_ymax = max(a[1], a[3])
    a_ymin = min(a[1], a[3])

    width = a_xmax - a_xmin
    height = a_ymax - a_ymin

    return width * height


def intersect_over_min_area(a, b):
    intersect = intersect_area(a, b)
    area_a = area(a)
    area_b = area(b)

    min_area = min(area_a, area_b)

    if min_area == 0:
        return 0

    return intersect / min_area


def intersect_over_union_area(a, b):
    intersect = intersect_area(a, b)
    union = union_area(a, b)
    if union == 0:
        return 0
    return intersect / union


def assumed_feet_movement(a, b):
    x1 = int((a[0] + a[2]) / 2)
    y1 = a[3]

    x2 = int((b[0] + b[2]) / 2)
    y2 = b[3]

    dist = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return dist ** 0.5


def wall_movement(a, b):
    left_movement = abs(a[0] - b[0])
    top_movement = abs(a[1] - b[1]) * 2
    right_movement = abs(a[2] - b[2])
    bottom_movement = abs(a[3] - b[3]) * 3

    total_movement = top_movement + left_movement + bottom_movement + right_movement
    return total_movement
