import cv2
import numpy as np
from TextPMs.util.misc import find_bottom, find_long_edges, split_edge_seqence, norm2, split_edge_seqence_by_step
from TextPMs.util.detection import sigmoid_alpha

def make_text_region(img, polygons, scale, mask_cnt, alpha):
    h, w = img.shape[0]//scale, img.shape[1]//scale
    mask_ones = np.ones(img.shape[:2], np.uint8)
    mask_zeros = np.zeros(img.shape[:2], np.uint8)

    train_mask = np.ones((h, w), np.uint8)
    tr_mask = np.zeros((h, w, mask_cnt), np.float32)
    if polygons is None:
        return tr_mask, train_mask

    for polygon in polygons:
        instance_mask = mask_zeros.copy()
        cv2.fillPoly(instance_mask, [polygon.points.astype(np.int32)], color=(1,))
        # dmp = ndimg.distance_transform_edt(instance_mask[::scale, ::scale])  # distance transform
        dmp = cv2.distanceTransform(instance_mask[::scale, ::scale], cv2.DIST_L2, 5)
        for i, k in enumerate(alpha):
            tr_mask[:, :, i] = np.maximum(tr_mask[:, :, i], sigmoid_alpha(dmp, k))

        if polygon.text == '#':
            cv2.fillPoly(mask_ones, [polygon.points.astype(np.int32)], color=(0,))
            continue

    train_mask = mask_ones[::scale, ::scale]

    return tr_mask, train_mask

class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)