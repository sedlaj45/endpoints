'''
Segmentation of handtools from Handtool video dataset using Mask R-CNN.

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

python handtools_ijcv_mrcnn_inference.py --class_label=spade
'''

'''
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluation on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
'''


############################################################
#  Load libraries
############################################################

import os
import shutil
import sys
import numpy as np
import json
import skimage.io
import cv2
import matplotlib.pyplot as plt


############################################################
#  Specify directories
############################################################

# Root directory of the project
ROOT_DIR = os.path.abspath('Mask_RCNN')

# Import Mask R-CNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
COCO_MODEL_PATH = os.path.join(WEIGHTS_DIR , "mask_rcnn_coco.h5")

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


def reshape_image(img, h_output, w_output):
    """
    Change image size using zero-padding
    :param img: RGB or grayscale image
    :param h_output: Output image height
    :param w_output: Output image width
    :return: Resized image
    """
    h2 = h_output
    w2 = w_output
    h, w = np.shape(img)[:2]
    dy2 = max(0, (h - h2) // 2)
    dx2 = max(0, (w - w2) // 2)
    dy = max(0, (h2 - h) // 2)
    dx = max(0, (w2 - w) // 2)
    if len(np.shape(img)) == 2:  # grayscale
        img2 = np.zeros((h2, w2), float)
        img2[dy:dy+h, dx:dx+w] = img[dy2:dy2+h2, dx2:dx2+w2]
    else:  # color
        img2 = np.zeros((h2, w2, np.shape(img)[2]), float)
        img2[dy:dy+h, dx:dx+w, :] = img[dy2:dy2+h2, dx2:dx2+w2, :]
    return img2


def create_circular_mask(h, w=None, radius=None):
    if w is None:
        w = h
    center = (w//2, h//2)
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1]) + 0.5
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center < radius
    mask = np.asarray(mask, np.uint8)
    return mask


def outer_contour(mask, thickness=4):
    """
    Compute outer contour of input mask, return as a binary 0/1 mask
    :param mask: Binary mask (background 0, foreground 1)
    :param thickness: Contour thickness
    :return: Outer contour as a binary 0/1 mask
    """
    if np.max(mask) == 0:  # empty mask
        return mask
    mask = mask / np.max(mask)
    mask[mask >= 0.5] = 1
    mask[mask < 1] = 0
    disc = create_circular_mask(2 * thickness + 1)
    mask_outer = cv2.dilate(mask, disc)
    mask_contour = mask_outer - mask
    return mask_contour


def overlay_mask(img, mask=None, bbox=None, score=1, color_mask=None, color_bbox=[0, 0.5, 1], cmap='viridis'):
    """
    Visualize segmentation mask and bounding box in an RGB image
    :param img: RGB image
    :param mask: (Binary) mask
    :param bbox: Bounding box (bbox), as a filled-in (binary) mask
    :param score: Score of detection (between 0 and 1)
    :param color_mask: RGB color of mask
    :param color_bbox: RGB color of bbox
    :return: RGB image with visualized mask, bbox, and GT bbox
    """
    if color_mask is None:
        color_map = plt.get_cmap(cmap)
        color_mask = color_map(score)
    img_contour = np.copy(img)
    if bbox is not None:  # normalize bbox, create outer contour
        bbox_contour = outer_contour(mask=bbox)
        for i in range(3):
            img_contour[:, :, i][bbox_contour>0] = color_bbox[i] * np.max(img)
    if mask is not None:  # normalize mask, create outer contour
        mask_contour = outer_contour(mask=mask)
        for i in range(3):
            img_contour[:, :, i][mask_contour>0] = color_mask[i] * np.max(img)
    return img_contour


def save_mask(path_output, frame_name, r_score_list, r_roi_list, r_mask_list, image_mrcnn, image_height, image_width,
              path_mask=None, path_bbox=None, path_img_mask_bbox=None, save_empty=False):
    """
    Save Mask R-CNN results for inferred image, indexed according to score in descending order:
    - mask (binary image)
    - bounding box (binary image)
    - RGB image with visualized mask and bounding box
    :param path_output: Directory for output
    :param frame_name: Filename base for output
    :param r_score_list: Mask R-CNN inference result: scores
    :param r_roi_list: Mask R-CNN inference result: rois
    :param r_mask_list: Mask R-CNN inference result: masks
    :param image_mrcnn: Inferred RGB image after Mask R-CNN padding
    :param image_height: Height of inferred image
    :param image_width: Width of inferred image
    :param path_mask: Directory for output mask
    :param path_bbox Directory for output bounding box
    :param path_img_mask_bbox: Directory for output mask and bounding box visualized in RGB image
    :param save_empty: Save images even if there's no detection
    :return: List of score values in descending order
    """
    if path_mask is None:
        path_mask = os.path.join(path_output, 'mask')
    if path_bbox is None:
        path_bbox = os.path.join(path_output, 'bbox')
    if path_img_mask_bbox is None:
        path_img_mask_bbox = os.path.join(path_output, 'img_mask_bbox')
    resized = False
    h, w = np.shape(image_mrcnn)[:2]
    h2 = image_height
    w2 = image_width
    dh = 0
    dw = 0
    if h != h2 or w != w2:  # crop image
        resized = True
        dh = int(h - h2) // 2
        dw = int(w - w2) // 2
        image_mrcnn = image_mrcnn[dh:h-dh, dw:w-dw, :]
    img_mask_bbox = image_mrcnn
    n_detections = len(r_score_list)
    if n_detections == 0:  # no detection
        if save_empty:
            mask = np.zeros((h, w))
            bbox = np.zeros((h, w))
            cv2.imwrite(os.path.join(path_mask, '{}.png'.format(frame_name)), mask)
            cv2.imwrite(os.path.join(path_bbox, '{}.png'.format(frame_name)), bbox)
    else:
        for i in range(n_detections):  # process from highest score
            score = r_score_list[i]
            roi = r_roi_list[i]
            mask = 255 * r_mask_list[i]
            if resized:
                h_mask, w_mask = np.shape(mask)[:2]
                mask = mask[dh:h_mask-dh, dw:w_mask-dw]
                roi[0] = max(roi[0] - dh, 0)
                roi[2] = min(roi[2] - dh, h)
                roi[1] = max(roi[1] - dw, 0)
                roi[3] = min(roi[3] - dw, w)
            bbox = np.zeros((h2, w2))
            bbox[roi[0]:roi[2], roi[1]:roi[3]] = 255
            img_mask_bbox = overlay_mask(img_mask_bbox, mask=mask, bbox=bbox, score=score)
            cv2.imwrite(os.path.join(path_mask,  '{}_{}.png'.format(frame_name, i)), mask)
            cv2.imwrite(os.path.join(path_bbox, '{}_{}.png'.format(frame_name, i)), bbox)
    cv2.imwrite(os.path.join(path_img_mask_bbox, '{}.png'.format(frame_name)), img_mask_bbox[:, :, ::-1])
    # ..., [cv2.IMWRITE_PNG_COMPRESSION, 9])  # better, but slower compression
    return


############################################################
#  Configurations
############################################################

class ToolsConfig(Config):
    """
    Configuration for training on own dataset.
    Derives from the base Config class and overrides values specific to the new dataset.
    """
    def __init__(self, name='handtools', gpu_count=1, images_per_gpu=1, num_classes_wo_background=1,
                 h_mrcnn=608, w_mrcnn=608, image_padding=True, rpn_nms_threshold=1.0,
                 use_mini_mask=False, detection_min_confidence=0.7, detection_max_instances=0.7):
        self.NAME = name  # configuration name
        self.GPU_COUNT = gpu_count  # number of GPUs
        self.IMAGES_PER_GPU = images_per_gpu  # images per GPU
        self.NUM_CLASSES = 1 + num_classes_wo_background  # total number of classes, incl. background
        self.IMAGE_MIN_DIM = min(h_mrcnn, w_mrcnn)
        self.IMAGE_MAX_DIM = max(h_mrcnn, w_mrcnn)
        self.IMAGE_PADDING = image_padding
        self.RPN_NMS_THRESHOLD = rpn_nms_threshold
        self.USE_MINI_MASK = use_mini_mask
        self.DETECTION_MIN_CONFIDENCE = detection_min_confidence
        self.DETECTION_MAX_INSTANCES = detection_max_instances
        self.MAX_GT_INSTANCES = 1  # orig. 100
        super().__init__()
        return


############################################################
#  Dataset
############################################################

class ToolsDataset(utils.Dataset):

    def __init__(self, h_mrcnn=608, w_mrcnn=608, images_per_gpu=1):
        super().__init__()
        self.h_mrcnn = h_mrcnn
        self.w_mrcnn = w_mrcnn
        self.images_per_gpu = images_per_gpu
        self.image_height = None
        self.image_width = None
        return

    def load_frames(self, dataset_dir, dataset_name='dataset', class_label='object',
                    image_height=400, image_width=600):
        """Load frames for evaluation.
        :param dataset_dir: Root directory of the dataset.
        :param dataset_name: Name of the dataset, e.g. handtools.
        :param class_label: Name of object class, e.g. 'hammer', 'barbell', 'scythe', or 'spade'.
        :param image_height: Height of dataset images
        :param image_width: Width of dataset images
        :return: Number of loaded images
        """
        self.add_class(dataset_name, class_id=1, class_name=class_label)
        print('Loading folder: {}'.format(dataset_dir))
        basename_list = get_list_of_files(dataset_dir, sort=True)
        n_files = len(basename_list)
        # Add files to dataset
        for image_id in range(len(basename_list)):
            # Note: Loading image size for each image is time-consuming
            # image_size = re.search('(\d+) x (\d+)', magic.from_file(fname_image)).groups()
            # image_height = int(image_size[1])
            # image_width = int(image_size[0])
            self.add_image(dataset_name, image_id=image_id, mask_class_list=[1],
                           path=os.path.join(dataset_dir, basename_list[image_id]),
                           width=image_width, height=image_height)
        print('Number of loaded images: {}'.format(n_files))
        self.image_height = image_height
        self.image_width = image_width
        return n_files  # number of added files

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        image = reshape_image(img=image, h_output=self.h_mrcnn, w_output=self.w_mrcnn)
        return image.astype(np.int32)

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            mask: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        mask_class_list = self.image_info[image_id]['mask_class_list']
        class_name_list = []
        for mask_class in mask_class_list:
            class_name_list.append(self.class_names[mask_class])
        class_ids = np.asarray(range(self.num_classes), np.int32)
        height = self.image_info[image_id]['height']
        width = self.image_info[image_id]['width']
        mask = np.zeros([height, width, self.num_classes], np.float)
        mask = reshape_image(img=mask, h_output=self.h_mrcnn, w_output=self.w_mrcnn)
        return mask.astype(np.bool), class_ids.astype(np.int32)


############################################################
#  Main function
############################################################

if __name__ == '__main__':
    import argparse

    ##########################
    # Command line arguments
    ##########################

    parser = argparse.ArgumentParser(description='Handtool dataset: Mask R-CNN.')
    parser.add_argument('--dataset', required=False, default=HANDTOOLS_DIR, metavar="<path/to/dataset/>",
                        help='Dataset directory')
    parser.add_argument('--class_label', required=False, default=None, metavar="<barbell|hammer|scythe|spade|...>",
                        help="Handtool name")
    parser.add_argument('--model', required=False, default=None, metavar="<path/to/weights.h5>",
                        help="Path to .h5 weights file or {'coco', 'last', 'imagenet'}")
    parser.add_argument('--image_height', required=False, default=400, type=int, metavar="<inferred image height>",
                        help='Inferred image height')
    parser.add_argument('--image_width', required=False, default=600, type=int, metavar="<inferred image width>",
                        help='Inferred image width')
    parser.add_argument('--mrcnn_image_size', required=False, default=608, type=int, metavar="<Mask R-CNN image height==width>",
                        help='Mask R-CNN image height==width')
    parser.add_argument('--detection_max_instances', required=False, default=1, type=int,
                        metavar="<detection max instances>", help='Detection max instances')
    parser.add_argument('--images_per_gpu', required=False, default=1, type=int, metavar="<images per GPU>",
                        help='Number of images per GPU')
    parser.add_argument('--gpu_count', required=False, default=1, type=int, metavar="<number of GPUs>",
                        help='Number of GPUs')
    parser.add_argument('--no_image_padding', required=False, default=False, action='store_true',
                        help="Don't use image padding")
    parser.add_argument('--use_mini_mask', required=False, default=False, action='store_true',
                        help="Use minimasks")
    parser.add_argument('--video_number_list', required=False, default=None, type=str, metavar="<comma-separated video numbers>",
                        help="Numbers of videos for evaluation (e.g. '1,2,3,4,5')")
    parser.add_argument('--no_video_output', required=False, default=False, action='store_true',
                        help="Don't save output video")
    parser.add_argument('--save_empty', required=False, default=False, action='store_true',
                        help="Save also empty detections")
    parser.add_argument('--fps', required=False, default=25, type=float, metavar="<fps for output video>",
                        help="Frames per second for output video")
    parser.add_argument('--frames_dir', required=False, default=None, metavar="<path/to/inferred_frames/>",
                        help='Inferred video frames directory')
    parser.add_argument('--results_dir', required=False, default=None, metavar="<path/to/results/>",
                        help='Results directory')
    parser.add_argument('--models_dir', required=False, default=None, metavar="<path/to/model_weights/>",
                        help='Model weights directory')
    parser.add_argument('--test_rpn_nms', required=False, default=0.7, type=float,
                        metavar="<Inference RPN NMS threshold>", help='Inference RPN NMS threshold')
    parser.add_argument('--test_dmc', required=False, default=0.7, type=float,
                        metavar="<inference detection min confidence>",
                        help='Inference detection min confidence')
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
    if args.class_label is None:
        args.class_label = 'spade'

    if args.video_number_list is None:
        args.video_number_list = video_number_list_dict[args.class_label]
    else:
        args.video_number_list = [int(x) for x in args.video_number_list.split(',')]

    args.save_video = not args.no_video_output
    args.image_padding = not args.no_image_padding

    if args.frames_dir is None:
        args.frames_dir = os.path.join(args.dataset, 'frames')
    if args.results_dir is None:
        args.results_dir = os.path.join(args.dataset, 'results')
    if args.models_dir is None:
        args.models_dir = os.path.join(args.dataset, 'mrcnn_weights')

    if args.model is None:
        args.model = os.path.join(args.models_dir, '{}.h5'.format(args.class_label))

    ##########################
    # Settings
    ##########################

    print('\nParameters:')
    print("Dataset directory: {}".format(args.dataset))
    print("Class label: {}".format(args.class_label))
    print("Model (.h5 file): {}".format(args.model))
    print("Inferred image height: {}".format(args.image_height))
    print("Inferred image width: {}".format(args.image_width))
    print("Mask R-CNN image height==width: {}".format(args.mrcnn_image_size))
    print("Detection max instances: {}".format(args.detection_max_instances))
    print("Images per GPU: {}".format(args.images_per_gpu))
    print("Number of GPUs: {}".format(args.gpu_count))
    print("Use image padding: {}".format(args.image_padding))
    print("Use minimasks: {}".format(args.use_mini_mask))
    print("Video numbers: {}".format(args.video_number_list))
    print("Save also empty detections: {}".format(args.save_empty))
    print("Save video output: {}".format(args.save_video))
    print("Output video fps: {}".format(args.fps))
    print("Frames directory: {}".format(args.frames_dir))
    print("Results directory: {}".format(args.results_dir))
    print("Models directory: {}".format(args.models_dir))
    print('')

    ##########################
    # Configuration
    ##########################

    config = ToolsConfig(name=args.class_label, gpu_count=args.gpu_count,
                         images_per_gpu=args.images_per_gpu,
                         h_mrcnn=args.mrcnn_image_size,
                         w_mrcnn=args.mrcnn_image_size, image_padding=args.image_padding,
                         rpn_nms_threshold=args.test_rpn_nms, use_mini_mask=args.use_mini_mask,
                         detection_min_confidence=args.test_dmc,
                         detection_max_instances=args.detection_max_instances)
    config_str = config.display()  # Show configuration values
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'config.txt'), 'w') as f:
        f.write(config_str)

    ##########################
    # Create model
    ##########################

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.models_dir)

    ##########################
    # Select weights file to load (and continue training from there)
    ##########################

    if args.model.lower() == "coco":  # Start from MS COCO trained weights
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":  # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        model_path = model.get_imagenet_weights()  # Start from ImageNet trained weights
    else:
        model_path = args.model  # .h5 file name

    ##########################
    # Load weights
    ##########################

    print("Loading Mask R-CNN weights from file: {}".format(model_path))
    model.load_weights(model_path, by_name=True)

    ##########################
    # Evaluate
    ##########################

    for video_number in args.video_number_list:

        video_name = '{}_{}'.format(args.class_label, video_number)  # e.g. 'hammer_1'

        # Test dataset
        path_frames = os.path.join(args.frames_dir, video_name)
        dataset_test = ToolsDataset()
        n_frames = dataset_test.load_frames(dataset_dir=path_frames, class_label=args.class_label,
                                            image_height=args.image_height, image_width=args.image_width)
        dataset_test.prepare()
        # print('video name: {}'.format(video_name))
        # print('n_frames: {}'.format(n_frames))

        # Run detection on test dataset
        path_output = os.path.join(args.results_dir, video_name)
        path_mask = os.path.join(path_output, 'mask')
        path_bbox = os.path.join(path_output, 'bbox')
        path_img_mask_bbox = os.path.join(path_output, 'img_mask_bbox')
        path_new_list = [path_mask, path_bbox, path_img_mask_bbox]
        for path_new in path_new_list:
            if os.path.isdir(path_new):
                shutil.rmtree(path_new)
            os.makedirs(path_new)

        basename_list = get_list_of_files(path_frames, sort=True)
        n_detected_frames = 0  # number of frames with detection
        scores_dict = {}

        print('Saving inference results to: {}'.format(path_output))

        print("Frame: [score(s)]:")
        for image_id in range(n_frames):

            # Results from Mask R-CNN inference
            image_mrcnn, image_meta, gt_class_id, gt_bbox, gt_mask\
                = modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=args.use_mini_mask)
            results = model.detect([image_mrcnn], verbose=0)
            r = results[0]

            # Remove results with empty mask
            n_detections = 0
            r_score_list = []
            r_roi_list = []
            r_mask_list = []
            r_class_id_list = []
            for i in range(len(r['scores'])):
                if np.max(r['masks'][:, :, i]):  # mask not empty
                    r_score_list.append(r['scores'][i])
                    r_roi_list.append(r['rois'][i])
                    r_mask_list.append(r['masks'][:, :, i])
                    r_class_id_list.append(r['class_ids'][i])
                    n_detections += 1

            # Sort results from highest score
            index_list = np.argsort(r_score_list)[::-1]
            r_score_list = [r_score_list[i] for i in index_list]
            r_roi_list = [r_roi_list[i] for i in index_list]
            r_mask_list = [r_mask_list[i] for i in index_list]
            r_class_id_list = [r_class_id_list[i] for i in index_list]

            # Visualize and save results
            frame_name = os.path.splitext(basename_list[image_id])[0]  # e.g. '0001'
            print(' {}: {}'.format(frame_name, np.round(100 * np.asarray(r_score_list), 1)))
            save_mask(path_output, frame_name, r_score_list=r_score_list, r_roi_list=r_roi_list, r_mask_list=r_mask_list,
                      image_mrcnn=image_mrcnn, image_height=args.image_height, image_width=args.image_width,
                      path_mask=path_mask, path_bbox=path_bbox, path_img_mask_bbox=path_img_mask_bbox,
                      save_empty=args.save_empty)
            scores_dict[frame_name] = [float(score) for score in r_score_list]
            if n_detections > 0:
                n_detected_frames += 1

        # Save scores
        print('Number of frames with detection(s): {} (out of {})'.format(n_detected_frames, n_frames))
        fname_scores = os.path.join(path_output, '{}_scores.json'.format(video_name))
        with open(fname_scores, 'w', encoding='utf-8') as f:
            json.dump(scores_dict, f, ensure_ascii=False, indent=4)

        # Create a video with visualized masks and bounding boxes
        if args.save_video:
            path_vis_frames = os.path.join(path_output, 'img_mask_bbox')
            fname_vis_video = os.path.join(path_output, '{}.mp4'.format(video_name))
            basename_vis_frame_list = get_list_of_files(path_vis_frames, sort=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vis_video = cv2.VideoWriter(fname_vis_video, fourcc, args.fps, (args.image_width, args.image_height))
            for basename_vis_frame in basename_vis_frame_list:
                vis_video.write(cv2.imread(os.path.join(path_vis_frames, basename_vis_frame)))
            cv2.destroyAllWindows()
            vis_video.release()
            print('Visualization saved to video file: {}'.format(fname_vis_video))

        print('')

