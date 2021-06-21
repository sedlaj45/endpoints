'''
Segmentation of handtools from Handtool dataset using Mask R-CNN.

Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard, Josef Sivic:
Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video,
CVPR 2019.
https://arxiv.org/abs/1904.02683

Jiri Sedlar, 2019-2021
Intelligent Machine Perception Project (IMPACT)
http://impact.ciirc.cvut.cz/
CIIRC, Czech Technical University in Prague

Based on implementation of Mask R-CNN by Matterport (see below).
https://github.com/matterport/Mask_RCNN

Usage example:

python handtools_ijcv_recognize_endpoints.py --class_label=spade
'''

############################################################
#  Load libraries
############################################################

import os
import shutil
import cv2
import numpy as np
import pickle
import json


############################################################
#  Specify directories
############################################################

# Root directory of the project
ROOT_DIR = os.path.abspath('Mask_RCNN')

# Handtool dataset directories
HANDTOOLS_DIR = os.path.join('handtools')


############################################################
#  Functions
############################################################

def alphanum_key(s):
    """
    Alphanumeric sorting
    :param s:
    :return:
    """
    import re
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def get_list_of_files(path, sort=True):
    """
    List files in a given path directory
    :param path: Directory path
    :return: List of file basenames in the directory
    """
    basename_list = [basename for basename in os.listdir(path) if os.path.isfile(os.path.join(path, basename))]
    if sort:
        basename_list.sort(key=alphanum_key)
    return basename_list


def swap_arrays(x, y):
    tmp = x.copy()
    x = y
    y = tmp
    return x, y


def load_openpose(openpose_dir, video_name, openpose_basename='handtools_openpose.pkl'):
    """
    Load joints from Openpose
    :param openpose_dir: directory with pose files (.pkl)
    :param video_name: name of video, e.g. 'hammer_1'
    :return: Openpose data ([frame_index, 18 joint indices, [x, y, score]]),
             indices of joints ({'r_wrist': 4, 'l_wrist': 7})
    """
    j2d_path = os.path.join(openpose_dir, openpose_basename)
    with open(j2d_path, 'rb') as f:
        j2d_data = pickle.load(f, encoding='bytes')
        j2d_pos = j2d_data[str.encode(video_name)]
    openpose_index_dict = {
        'r_wrist': 4,
        'l_wrist': 7,
    }
    return j2d_pos, openpose_index_dict


def endpoints_from_mask_and_bbox(fname_mask, fname_bbox=None):
    """
    Compute endpoints as intersections between bounding box and line fitted to segmentation mask
    :param fname_mask: File name of image with binary segmentation mask
    :param fname_bbox: File name of image with binary bounding box (filled rectangle)
    :return: List of endpoints [[y0, x0], [y1, x1]] or None
    """
    # Segmentation mask
    mask = cv2.imread(fname_mask, 0)  # grayscale
    if np.max(mask) <= 0:  # no detection
        return None
    mask = np.asarray(np.round(mask / np.max(mask)), np.uint8)  # binary 0/1

    # Fit line through mask
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    for contour in contours[1:]:
        cnt = np.vstack((cnt, contour))
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # print('vx, vy, x, y:', vx, vy, x, y)

    # Line ax + by + c = 0
    a = vy[0]
    b = -vx[0]
    c = - (a * x[0] + b * y[0])
    # print('a, b, c:', a, b, c)

    # Bounding box
    if fname_bbox is None:
        bbox = mask  # in absence of bounding box use mask instead
    else:
        bbox = cv2.imread(fname_bbox, 0)  # grayscale
        if np.max(bbox) == 0:
            bbox = mask  # in absence of bounding box use mask instead
        else:
            bbox = np.asarray(np.round(bbox / np.max(bbox)), np.uint8)  # binary 0/1
    y0 = np.max(np.argmax(bbox, axis=0))
    y1 = np.shape(bbox)[0] - np.max(np.argmax(bbox[::-1, :], axis=0)) - 1
    x0 = np.max(np.argmax(bbox, axis=1))
    x1 = np.shape(bbox)[1] - np.max(np.argmax(bbox[:, ::-1], axis=1)) - 1
    # print('x0, x1, y0, y1:', x0, x1, y0, y1)

    # Intersection of line and bounding box
    if a != 0:
        xA = - (b * y0 + c) / a
        xB = - (b * y1 + c) / a
        xA = int(np.round(xA))
        xB = int(np.round(xB))
    else:
        xA = np.inf
        xB = np.inf
    if b != 0:
        yA = - (a * x0 + c) / b
        yB = - (a * x1 + c) / b
        yA = int(np.round(yA))
        yB = int(np.round(yB))
    else:
        yA = np.inf
        yB = np.inf
    # print('xA, xB, yA, yB:', xA, xB, yA, yB)
    endpoint_list = []
    if x0 <= xA <= x1:
        # print(xA, y0)
        if [xA, y0] not in endpoint_list:
            endpoint_list.append([y0, xA])
    if x0 <= xB <= x1:
        # print(xB, y1)
        if [xB, y1] not in endpoint_list:
            endpoint_list.append([y1, xB])
    if y0 <= yA <= y1:
        # print(x0, yA)
        if [x0, yA] not in endpoint_list:
            endpoint_list.append([yA, x0])
    if y0 <= yB <= y1:
        # print(x1, yB)
        if [x1, yB] not in endpoint_list:
            endpoint_list.append([yB, x1])
    # print(endpoint_list)
    if len(endpoint_list) > 2:
        dist_max = 0
        endpoint_index_list = []
        for i in range(len(endpoint_list)):
            for j in range(i):
                dist_ij = dist_points(endpoint_list[i], endpoint_list[j])
                if dist_ij > dist_max:
                    dist_max = dist_ij
                    endpoint_index_list = [j, i]
        endpoint_list = [endpoint_list[endpoint_index_list[0]], endpoint_list[endpoint_index_list[1]]]
    if len(endpoint_list) != 2:
        return None
    return endpoint_list


def dist_point_segment(point, point_1, point_2):
    """
    Distance of point from segment between 2 endpoints
    :param point: Coordinates [y, x] of point
    :param point_1: Coordinates [y, x] of one endpoint
    :param point_2: Coordinates [y, x] of the other endpoint
    :return: Distance of the point from the segment (infinity if invalid)
    """
    if np.any(np.isnan([point, point_1, point_2])):
        return np.inf

    point = np.asarray(point, float)
    point_1 = np.asarray(point_1, float)
    point_2 = np.asarray(point_2, float)
    dist_1 = dist_points(point, point_1)
    dist_2 = dist_points(point, point_2)
    dist_1_2 = dist_points(point_1, point_2)

    if min(dist_1, dist_2) == 0:  # point == point_1 or point == point_2
        return 0
    if dist_1_2 == 0:  # point_1 == point_2
        return dist_1
    dot_1 = np.dot((point_2 - point_1) / np.linalg.norm(point_2 - point_1), (point - point_1) / np.linalg.norm(point - point_1))
    dot_2 = np.dot((point_1 - point_2) / np.linalg.norm(point_1 - point_2), (point - point_2) / np.linalg.norm(point - point_2))
    if dot_1 < 0 and dot_2 >= 0:
        dist = dist_1
    elif dot_1 >= 0 and dot_2 < 0:
        dist = dist_2
    elif dot_1 >= 0 and dot_2 >= 0:
        dist = np.abs(np.cross(point_2 - point_1, point - point_1) / np.linalg.norm(point_2 - point_1))
    else:
        print('Warning: Error in dist_point_segment()')
        return np.inf
    return dist


def dist_points(point_1, point_2):
    """
    Euclidean distance between 2 points
    :param point_1: Coordinates [y, x] of one point
    :param point_2: Coordinates [y, x] of the other endpoint
    :return: Distance between point_1 and point_2 (infinity if invalid)
    """
    if np.any(np.isnan([point_1, point_2])):
        return np.inf
    dist = np.linalg.norm(np.asarray(point_1, float) - np.asarray(point_2, float))
    return dist


def create_circular_mask(h, w=None, radius=None):
    """
    Create a binary circular mask of given shape
    :param h: Height
    :param w: Width
    :param radius: Radius
    :return: Binary circular mask of shape (h, w)
    """
    if w is None:
        w = h
    center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1]) + 0.5
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center < radius
    mask = np.asarray(mask, np.uint8)
    return mask


def draw_mask(img, mask=None, mask_rgb=(255, 0, 255), mask_border_rgb=(0, 0, 0),
              mask_alpha=0.75, mask_border_alpha=1, mask_border_thickness=5):
    """
    Visualize binary mask
    :param img: RGB image
    :param mask: Binary mask
    :param mask_rgb: RGB color of mask
    :param mask_border_rgb: RGB color of mask boundary
    :param mask_alpha: Opacity of mask
    :param mask_border_alpha: Opacity of mask boundary
    :param mask_border_thickness: Thickness of mask boundary
    :return: RGB image with visualized mask
    """
    if mask is None or np.max(mask) <= 0:  # no mask
        return img

    mask = mask / np.max(mask)  # binary 0/1
    for k in range(3):  # mask
        img[:, :, k] = (1 - mask) * img[:, :, k] \
                       + mask * mask_alpha * mask_rgb[k]\
                       + mask * (1 - mask_alpha) * img[:, :, k]
    mask_border = cv2.dilate(mask, create_circular_mask(mask_border_thickness)) - mask  # outer boundary
    for k in range(3):  # mask boundary
        img[:, :, k] = (1 - mask_border) * img[:, :, k] \
                       + mask_border * mask_border_alpha * mask_border_rgb[k]\
                       + mask_border * (1 - mask_border_alpha) * img[:, :, k]
    return img


def draw_endpoints(img, endpoint_1, endpoint_2,
                   endpoint_radius=10, endpoint_1_rgb=(255, 255, 0), endpoint_2_rgb=(0, 255, 255)):
    """
    Visualize endpoints
    :param img: RGB image
    :param endpoint_1: Coordinates [y, x] of endpoint 1
    :param endpoint_2: Coordinates [y, x] of endpoint 2
    :param endpoint_radius: Radius of circle marker
    :param endpoint_1_rgb: RGB color of enpoint 1
    :param endpoint_2_rgb: RGB color of endpoint 2
    :return: RGB image with visulized endpoints
    """
    try:
        img = np.asarray(img, float)
        if not np.any(np.isnan(endpoint_2)):
            cv2.circle(img, (int(endpoint_2[1]), int(endpoint_2[0])),
                       radius=endpoint_radius, color=endpoint_2_rgb, thickness=-1)
            cv2.circle(img, (int(endpoint_2[1]), int(endpoint_2[0])),
                       radius=endpoint_radius+1, color=(0, 0, 0), thickness=2)
        if not np.any(np.isnan(endpoint_1)):
            cv2.circle(img, (int(endpoint_1[1]), int(endpoint_1[0])),
                       radius=endpoint_radius, color=endpoint_1_rgb, thickness=-1)
            cv2.circle(img, (int(endpoint_1[1]), int(endpoint_1[0])),
                       radius=endpoint_radius+1, color=(0, 0, 0), thickness=2)
    except Exception:
        print('Warning: Error in draw_endpoints()')
    return img


def frames_dict_to_list(d, n_frames=None, default_value=None):
    """
    Convert dictionary to list
    :param d: Python dictionary
    :param length_min: Minimum length of output list
    :param default_value: Value for list items that are not keys in the dictionary
    :return: Python list
    """
    index_list = [int(key) - 1 for key in d.keys()]
    if n_frames is None:
        n_frames = np.max(index_list) + 1
    l = [default_value] * n_frames
    for key in d.keys():
        l[int(key) - 1] = d[key]
    return l


def frames_to_video(fname_video, path_frames, fps=25):
    """
    Convert image frames to video
    :param fname_video: File name for output video
    :param path_frames: Path of folder with frames
    :param fps: Frames per second
    :return:
    """
    fname_list = get_list_of_files(path_frames, sort=True)
    frame = cv2.imread(os.path.join(path_frames, fname_list[0]))
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(fname_video, fourcc, fps, (width, height))
    for fname in fname_list:
        video.write(cv2.imread(os.path.join(path_frames, fname)))
    cv2.destroyAllWindows()
    video.release()
    return


############################################################
#  Main function
############################################################

if __name__ == '__main__':
    import argparse

    ##########################
    # Command line arguments
    ##########################

    parser = argparse.ArgumentParser(description='Handtool dataset: Extract endpoints.')
    parser.add_argument('--dataset', required=False, default=HANDTOOLS_DIR, metavar="<path/to/dataset/>",
                        help='Dataset directory')
    parser.add_argument('--class_label', required=False, default=None, metavar="<barbell|hammer|scythe|spade|...>",
                        help="Handtool name")
    parser.add_argument('--heuristic', required=False, default=None, metavar="<heuristic for orientation of endpoints>",
                        help='Heuristic for orientation of endpoints')
    parser.add_argument('--wrist_confidence_threshold', required=False, default=None, type=float,
                        metavar="<confidence threshold for wrist joints>",
                        help="Confidence threshold for wrist joints")
    parser.add_argument('--wrist_distance_threshold', required=False, default=None, type=float,
                        metavar="<threshold for distance of wrist from segment between endpoints>",
                        help="Threshold for distance of wrist from segment between endpoints")
    parser.add_argument('--endpoint_1_rgb', required=False, default='255,255,0', type=str,
                        metavar="<comma-separated RGB values for color of endpoint 1 (head/left)>",
                        help="RGB color of endpoint 1 (head/left), e.g. '255,255,0'")
    parser.add_argument('--endpoint_2_rgb', required=False, default='0,255,255', type=str,
                        metavar="<comma-separated RGB value for color of endpoint 2 (handle/right)>",
                        help="RGB color of endpoint 2 (handle/right), e.g. '0,255,255'")
    parser.add_argument('--video_number_list', required=False, default=None, type=str,
                        metavar="<comma-separated video numbers>",
                        help="Numbers of videos for evaluation (e.g. '1,2,3,4,5')")
    parser.add_argument('--frames_dir', required=False, default=None, metavar="<path/to/inferred_frames/>",
                        help='Inferred video frames directory')
    parser.add_argument('--results_dir', required=False, default=None, metavar="<path/to/results/>",
                        help='Results directory')
    parser.add_argument('--openpose_dir', required=False, default=None, metavar="<path/to/openpose/>",
                        help='Openpose directory')
    args = parser.parse_args()

    ##########################
    # Arguments
    ##########################

    video_number_list = [1, 2, 3, 4, 5]
    video_number_list_dict = {
        'barbell': video_number_list,
        'hammer': video_number_list,
        'scythe': video_number_list,
        'spade': video_number_list
    }
    wrist_distance_threshold_dict = {
        'barbell': 35,
        'hammer': 35,
        'scythe': 60,
        'spade': 35,
    }
    if args.class_label is None:
        args.class_label = 'spade'

    if args.video_number_list is None:
        args.video_number_list = video_number_list_dict[args.class_label]
    else:
        args.video_number_list = [int(x) for x in args.video_number_list.split(',')]

    if args.wrist_confidence_threshold is None:
        args.wrist_confidence_threshold = 0.2

    if args.wrist_distance_threshold is None:
        args.wrist_distance_threshold = wrist_distance_threshold_dict[args.class_label]

    # Folders
    if args.frames_dir is None:
        args.frames_dir = os.path.join(args.dataset, 'frames')
    if args.results_dir is None:
        args.results_dir = os.path.join(args.dataset, 'results')
    if args.openpose_dir is None:
        args.openpose_dir = os.path.join(args.dataset, 'openpose')

    # Colors
    args.endpoint_1_rgb = [int(val) for val in str(args.endpoint_1_rgb).split(',')]
    args.endpoint_2_rgb = [int(val) for val in str(args.endpoint_2_rgb).split(',')]

    # Heuristic
    heuristic_dict = {
        'barbell': 'horizontal',
        'hammer': 'fixed_loose',
        'scythe': 'vertical',
        'spade': 'vertical',
    }
    
    if args.heuristic is None:
        args.heuristic = heuristic_dict[args.class_label]

    ##########################
    # Settings
    ##########################

    print("Dataset directory: {}".format(args.dataset))
    print("Class label: {}".format(args.class_label))
    print("Heuristic for orientation: {}".format(args.heuristic))
    print("Confidence threshold for wrist joints: {}".format(args.wrist_confidence_threshold))
    print("Distance threshold from wrist to segment between endpoints: {}".format(args.wrist_distance_threshold))
    print("RGB color of endpoint 1 (head/left): {}".format(args.endpoint_1_rgb))
    print("RGB color of endpoint 2 (handle/right): {}".format(args.endpoint_2_rgb))
    print("Video numbers: {}".format(args.video_number_list))
    print("Videos directory: {}".format(args.frames_dir))
    print("Results directory: {}".format(args.results_dir))
    print("Openpose directory: {}".format(args.openpose_dir))

    ##########################
    # Estimate endpoints from masks and bboxes
    ##########################

    for video_number in args.video_number_list:

        video_name = '{}_{}'.format(args.class_label, video_number)  # e.g. 'hammer_1'

        # Folders
        path_frames = os.path.join(args.frames_dir, video_name)
        path_openpose = os.path.join(args.openpose_dir, video_name)
        path_output = os.path.join(args.results_dir, video_name)
        path_mask = os.path.join(path_output, 'mask')
        path_bbox = os.path.join(path_output, 'bbox')
        path_img_mask_box = os.path.join(path_output, 'img_mask_bbox')
        path_img_endpoints = os.path.join(path_output, 'img_endpoints')
        path_img_openpose_endpoints = os.path.join(path_output, 'img_openpose_endpoints')
        path_new_list = [path_img_endpoints]  #, path_img_openpose_endpoints]
        for path_new in path_new_list:
            if os.path.isdir(path_new):
                shutil.rmtree(path_new)
            os.makedirs(path_new)

        # File names
        basename_list = get_list_of_files(path_frames, sort=True)  # ['0001.png', ...]
        frame_name_list = [os.path.splitext(basename)[0] for basename in basename_list]  # ['0001', ...]
        n_frames = len(basename_list)

        # Load segmentation scores
        fname_scores = os.path.join(path_output, '{}_scores.json'.format(video_name))
        with open(fname_scores) as f:    
            scores_json_dict = json.load(f)
        scores_list = frames_dict_to_list(scores_json_dict)

        # Openpose joints
        openpose_array, openpose_index_dict = load_openpose(args.openpose_dir, video_name)

        # Load wrist coordinates where confidence of Openpose detection is higher than threshold
        wrist_L_array = np.full([n_frames, 2], np.nan)
        wrist_R_array = np.full([n_frames, 2], np.nan)
        conf_L_array = np.full([n_frames], np.nan)
        conf_R_array = np.full([n_frames], np.nan)
        for k in range(n_frames):
            wrist_L = openpose_array[k, openpose_index_dict['l_wrist'], 1::-1]
            wrist_R = openpose_array[k, openpose_index_dict['r_wrist'], 1::-1]
            conf_L = openpose_array[k, openpose_index_dict['l_wrist'], 2]
            conf_R = openpose_array[k, openpose_index_dict['r_wrist'], 2]
            if conf_L >= args.wrist_confidence_threshold:
                wrist_L_array[k] = wrist_L
                conf_L_array[k] = conf_L
            if conf_R >= args.wrist_confidence_threshold:
                wrist_R_array[k] = wrist_R
                conf_R_array[k] = conf_R

        # Extract endpoints using mask, bbox, and wrists
        score_array = np.full([n_frames], np.nan)
        detection_index_array = np.full([n_frames], np.nan)
        dist_wrist_array = np.full([n_frames], np.nan)
        endpoint_1_array = np.full([n_frames, 2], np.nan)
        endpoint_2_array = np.full([n_frames, 2], np.nan)
        for k in range(n_frames):
        
            frame_name_k = frame_name_list[k]
            scores_k = scores_list[k]
            n_detections_k = len(scores_k)
            if n_detections_k == 0:  # no detection in frame
                continue

            # Select detection
            #  - maximize: score_i ... probability score of detection i
            #  - minimize: dist_i_j ... distance of wrist j from detection i
            #  - maximize: conf_j ... confidence of wrist j
            #  - argmin_i((1 - score_i) * min_j(dist_i_j * (1-conf_j)))
            #  - closer wrist: dist_i_0 * (1 - score_i)
            #  - farther wrist: dist_i_1 * (1 - score_i)

            endpoints_k_array = np.full([n_detections_k, 2, 2], np.nan)
            dist_wrist_L_k_array = np.full([n_detections_k], np.nan)
            dist_wrist_R_k_array = np.full([n_detections_k], np.nan)
            dist_wrist_k_array = np.full([n_detections_k], np.nan)
            score_k_array = np.full([n_detections_k], np.nan)
            for i in range(n_detections_k):
                endpoints_k_i = endpoints_from_mask_and_bbox(fname_mask=os.path.join(path_mask, '{}_{}.png'.format(frame_name_k, i)),
                                                             fname_bbox=os.path.join(path_bbox, '{}_{}.png'.format(frame_name_k, i)))
                if endpoints_k_i is not None:
                    endpoints_k_array[i] = endpoints_k_i
                    
                dist_wrist_L_k_array[i] = dist_point_segment(wrist_L_array[k], endpoints_k_array[i, 0], endpoints_k_array[i, 1])
                dist_wrist_R_k_array[i] = dist_point_segment(wrist_R_array[k], endpoints_k_array[i, 0], endpoints_k_array[i, 1])

                if np.any(np.isnan(wrist_L_array[k])) and np.any(np.isnan(wrist_R_array[k])):
                    dist_wrist_k_array[i] = np.inf  # no wrist in contact
                elif np.any(np.isnan(wrist_L_array[k])):
                    dist_wrist_k_array[i] = dist_wrist_R_k_array[i]  # only R wrist in contact
                elif np.any(np.isnan(wrist_R_array[k])):
                    dist_wrist_k_array[i] = dist_wrist_L_k_array[i]  # only L wrist in contact
                else:
                    dist_wrist_k_array[i] = np.max([dist_wrist_L_k_array[i], dist_wrist_R_k_array[i]])  # both L & R wrist in contact
                score_k_array[i] = scores_k[i]  # segmentation score

            detection_index_k = np.argmin(dist_wrist_k_array * (1 - score_k_array))
            score_array[k] = scores_k[detection_index_k]
            dist_wrist_array[k] = dist_wrist_k_array[detection_index_k]
            if dist_wrist_array[k] <= args.wrist_distance_threshold:
                endpoint_1_array[k] = endpoints_k_array[detection_index_k][0]
                endpoint_2_array[k] = endpoints_k_array[detection_index_k][1]
            detection_index_array[k] = detection_index_k

        # Determine orientation of endpoints using heuristics
        if args.heuristic == 'fixed_loose':  # endpoint_1 loose, endpoint_2 fixed
            dist_L_array = np.full([n_frames], np.nan)
            dist_R_array = np.full([n_frames], np.nan)
            # Distance of left/right wrist to closest endpoint
            for i in range(n_frames):
                if not np.any(np.isnan([endpoint_1_array[i], endpoint_2_array[i]])):
                    if not np.any(np.isnan(wrist_L_array[i])):
                        dist_L_array[i] = min(dist_points(wrist_L_array[i], endpoint_1_array[i]), dist_points(wrist_L_array[i], endpoint_2_array[i]))
                    if not np.any(np.isnan(wrist_R_array[i])):
                        dist_R_array[i] = min(dist_points(wrist_R_array[i], endpoint_1_array[i]), dist_points(wrist_R_array[i], endpoint_2_array[i]))
            print('Average distance of L wrist to closest endpoint:', np.round(np.nanmean(dist_L_array), 1))
            print('Average distance of R wrist to closest endpoint:', np.round(np.nanmean(dist_R_array), 1))
            # Label wrists as fixed and loose (for all frames)
            if np.nanmean(dist_L_array) <= np.nanmean(dist_R_array):
                wrist_fixed_array = wrist_L_array
                wrist_loose_array = wrist_R_array
            else:
                wrist_fixed_array = wrist_R_array
                wrist_loose_array = wrist_L_array
            # Label endpoints as fixed and loose (for each frame)
            for i in range(n_frames):
                if np.any(np.isnan(wrist_fixed_array[i])):
                    if not np.any(np.isnan([endpoint_1_array[i], endpoint_2_array[i]])):
                        print('Warning: Fixed wrist not detected - removing detected endpoints from frame {}'.format(frame_name_list[i]))
                    endpoint_1_array[i] = np.nan 
                    endpoint_2_array[i] = np.nan
                else:
                    if dist_points(wrist_fixed_array[i], endpoint_1_array[i]) <= dist_points(wrist_fixed_array[i], endpoint_2_array[i]):
                        endpoint_1_array[i], endpoint_2_array[i] = swap_arrays(endpoint_1_array[i], endpoint_2_array[i])
        elif args.heuristic == 'proximity':
            for i in range(n_frames):  # endpoint_1 closer to right wrist, endpoint_2 closer to left wrist
                if not np.any(np.isnan([endpoint_1_array[i], endpoint_2_array[i]])):
                    if not np.any(np.isnan(wrist_L_array[i])):
                        dist_L_1 = dist_points(wrist_L_array[i], endpoint_1_array[i])
                        dist_L_2 = dist_points(wrist_L_array[i], endpoint_2_array[i])
                    else:
                        dist_L_1 = np.inf
                        dist_L_2 = np.inf
                    if not np.any(np.isnan(wrist_R_array[i])):
                        dist_R_1 = dist_points(wrist_R_array[i], endpoint_1_array[i])
                        dist_R_2 = dist_points(wrist_R_array[i], endpoint_2_array[i])
                    else:
                        dist_R_1 = np.inf
                        dist_R_2 = np.inf
                    if np.argmin([dist_R_1, dist_L_2, dist_L_1, dist_R_2]) >= 2:
                        endpoint_1_array[i], endpoint_2_array[i] = swap_arrays(endpoint_1_array[i], endpoint_2_array[i])
        elif args.heuristic == 'vertical':  # endpoint_1 lower than endpoint_2
            for i in range(n_frames):
                if not np.any(np.isnan([endpoint_1_array[i], endpoint_2_array[i]])):
                    if endpoint_1_array[i][0] < endpoint_2_array[i][0]:
                        endpoint_1_array[i], endpoint_2_array[i] = swap_arrays(endpoint_1_array[i], endpoint_2_array[i])
        elif args.heuristic == 'horizontal':  # endpoint_1 to the left of endpoint_2
            for i in range(n_frames):
                if not np.any(np.isnan([endpoint_1_array[i], endpoint_2_array[i]])):
                    if endpoint_1_array[i][1] > endpoint_2_array[i][1]:
                        endpoint_1_array[i], endpoint_2_array[i] = swap_arrays(endpoint_1_array[i], endpoint_2_array[i])
        else:
            print('Warning: Unknown heuristic: {}', args.heuristic)

        fname_endpoints_json = os.path.join(path_output, '{}_endpoints_mrcnn.json'.format(video_name))
        endpoints_dict = {}
        fname_endpoints_txt = os.path.join(path_output, '{}_endpoints_mrcnn.txt'.format(video_name))
        endpoints_list = []
        n_detections = 0
        for k in range(n_frames):
            basename = basename_list[k]
            fsize_bytes = int(os.path.getsize(os.path.join(path_frames, basename)))
            if not np.any(np.isnan([endpoint_1_array[k], endpoint_2_array[k]])):
                n_detections += 1
                endpoints_list.append([int(frame_name_list[k]), int(endpoint_1_array[k, 1]), int(endpoint_1_array[k, 0])])
                endpoints_list.append([int(frame_name_list[k]), int(endpoint_2_array[k, 1]), int(endpoint_2_array[k, 0])])
                regions_list = [
                    {'shape_attributes': {'cx': int(endpoint_1_array[k, 1]), 'cy': int(endpoint_1_array[k, 0]), 'name': 'point'},
                     'region_attributes': {}
                    },
                    {'shape_attributes': {'cx': int(endpoint_2_array[k, 1]), 'cy': int(endpoint_2_array[k, 0]), 'name': 'point'},
                     'region_attributes': {}
                    },
                ]
            else:
                regions_list = []
            endpoints_dict['{}{}'.format(basename, fsize_bytes)] = {
                'filename': basename,
                'size': fsize_bytes,
                'regions': regions_list,
                'file_attributes': {}
            }
        with open(fname_endpoints_json, 'w', encoding='utf-8') as f:
            json.dump(endpoints_dict, f, ensure_ascii=False, indent=4)
        if len(endpoints_list) > 0:
            np.savetxt(fname_endpoints_txt, endpoints_list, fmt='%04d\t%d\t%d')
        else:
            np.savetxt(fname_endpoints_txt, [])
        fname_endpoints_txt = os.path.join(path_output, '../{}/{}_endpoints.txt'.format(args.heuristic, video_name))
        if len(endpoints_list) > 0:
            np.savetxt(fname_endpoints_txt, endpoints_list, fmt='%04d\t%d\t%d')
        else:
            np.savetxt(fname_endpoints_txt, [])

        # Draw endpoints
        for k in range(n_frames):
            basename = basename_list[k]
            fname_img = os.path.join(path_frames, basename)
            fname_img_mask_box = os.path.join(path_img_mask_box, basename)
            fname_img_openpose = os.path.join(path_openpose, basename)
            fname_img_endpoints = os.path.join(path_img_endpoints, basename)
            fname_img_openpose_endpoints = os.path.join(path_img_openpose_endpoints, basename)

            img = cv2.imread(fname_img, 1)[:, :, ::-1]
            img_mask_box = cv2.imread(fname_img_mask_box, 1)[:, :, ::-1]
            img_openpose = cv2.imread(fname_img_openpose, 1)[:, :, ::-1]

            if not np.isnan(detection_index_array[k]):
                fname_mask = os.path.join(path_mask, '{}_{}.png'.format(frame_name_list[k], int(detection_index_array[k])))
                mask = cv2.imread(fname_mask, 0)
            else:
                mask = None

            img_mask = draw_mask(img=img, mask=mask)
            img_openpose = draw_mask(img=img_openpose, mask=mask)
            img_endpoints = draw_endpoints(img_mask, endpoint_1_array[k], endpoint_2_array[k],
                                           endpoint_1_rgb=args.endpoint_1_rgb, endpoint_2_rgb=args.endpoint_2_rgb)
            img_openpose_endpoints = draw_endpoints(img_openpose, endpoint_1_array[k], endpoint_2_array[k],
                                                    endpoint_1_rgb=args.endpoint_1_rgb, endpoint_2_rgb=args.endpoint_2_rgb)
            cv2.imwrite(fname_img_endpoints, img_endpoints[:, :, ::-1])
            cv2.imwrite(fname_img_openpose_endpoints, img_openpose_endpoints[:, :, ::-1])

        # Save as video
        fname_video_endpoints = os.path.join(path_output, '{}_endpoints_mrcnn.mp4'.format(video_name))
        frames_to_video(fname_video_endpoints, path_img_endpoints)
        fname_video_openpose_endpoints = os.path.join(path_output, '{}_openpose_endpoints_mrcnn.mp4'.format(video_name))
        frames_to_video(fname_video_openpose_endpoints, path_img_openpose_endpoints)
        print('Number of frames with detected endpoints in video {}: {}'.format(video_name, n_detections))

