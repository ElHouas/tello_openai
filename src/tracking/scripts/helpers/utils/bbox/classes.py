import numpy as np

def bbox_to_center(bbox):
    '''Return center of box.
    Args:
        bbox in corner format [x1 y1 x2 y2] where x/yi are the corner pts.
    Returns:
        coordinates, [x y], of center.
    '''
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return  [int(x), int(y)]