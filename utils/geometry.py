import cv2
import numpy as np

def order_points_counter_clockwise(polygon):
    """
    Orders the 4 points of a quadrilateral in counter-clockwise order,
    starting from the top-left point.
    
    Args:
        polygon: A numpy array of shape (4, 2) representing 4 (x, y) points.
    
    Returns:
        A numpy array of shape (4, 2) with points ordered counter-clockwise as:
        [top-left, bottom-left, bottom-right, top-right]
    """
    # Create a copy to avoid modifying the original input
    rect = polygon.copy()
    
    # Calculate the sum and difference of coordinates
    s = rect.sum(axis=1)
    diff = np.diff(rect, axis=1)
    
    # Identify the characteristic points
    top_left = rect[np.argmin(s)]       # Smallest (x+y) sum
    bottom_right = rect[np.argmax(s)]   # Largest (x+y) sum
    top_right = rect[np.argmin(diff)]   # Smallest (x-y) difference
    bottom_left = rect[np.argmax(diff)] # Largest (x-y) difference

    counter_clockwise = np.array([top_left, bottom_left, 
                                  bottom_right, top_right], dtype=np.float32)
    
    return counter_clockwise

def align_polygon_perspective(img, polygons): 
    # Points are ordered anti-clockwise
    pt_A, pt_B, pt_C, pt_D = order_points_counter_clockwise(polygons)
    
    width_AD = np.linalg.norm(pt_A - pt_D)
    width_BC = np.linalg.norm(pt_B - pt_C)
    maxWidth = max(int(width_AD), int(width_BC))
    
    height_AB = np.linalg.norm(pt_A - pt_B)
    height_CD = np.linalg.norm(pt_C - pt_D)
    maxHeight = max(int(height_AB), int(height_CD))
    

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    # Inverse Matrix: Aligned → Original 
    M_inv = cv2.getPerspectiveTransform(output_pts, input_pts)

    # Apply the perspective transformation to the entire input image to get the final transformed image.
    aligned = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_CUBIC)

    return aligned, M_inv

def calculate_trapezoid_angles_positive(points):
    """
    Tính góc trong khoảng [0°, 180°]
    """
    p0, p1, p2, p3 = points

    angle = np.degrees(np.arctan2(p3[1] - p0[1], p3[0] - p0[0]))
    # Chuyển về [0°, 180°]
    angle = abs(angle) % 180
    if angle > 90:
        angle = 180 - angle
    
    return angle

def smart_crop_or_align(img, quadrilateral, polygon, angle_threshold=5):
    """
    Normalize góc về [0, 90] để dễ hiểu.
    0° = thẳng ngang hoặc thẳng dọc
    45° = xoay 45°
    """
    quadrilateral = order_points_counter_clockwise(quadrilateral)
    angle = calculate_trapezoid_angles_positive(quadrilateral)
    
    if angle <= angle_threshold:
        x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
        result = img[y:y+h, x:x+w]
        M_inv = np.array([[1, 0, x],
                        [0, 1, y],
                        [0, 0, 1]], dtype=np.float32)
    else:
        # Xoay nhiều → Align
        result, M_inv = align_polygon_perspective(img, quadrilateral)
    
    return result, M_inv

def transform_boxes_back(boxes, M_inv):
    """
    Transform boxes from aligned image back to original image space.
    
    Args:
        boxes: List of polygons, each with shape (n_points, 2)
        M_inv: Inverse perspective transformation matrix (3x3)
    
    Returns:
        List of transformed polygons in original image coordinates.
    """
    boxes_original = []
    for box in boxes:
        box_np = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(box_np, M_inv)
        boxes_original.append(transformed.reshape(-1, 2))
    return boxes_original