# Code for detection of object endpoints

Project webpage:
https://www.di.ens.fr/willow/research/motionforcesfromvideo/

## Installation

(Used modules: `OpenCV/3.4.1-foss-2018a-Python-3.6.4, cuDNN/7.6.4.38-gcccuda-2019b, CUDA/10.2.89-GCC-8.3.0, Anaconda3/5.0.1`)

1. Install **Mask R-CNN** from **Matterport**:  
https://github.com/matterport/Mask_RCNN

2. Copy [**`handtools_ijcv_mrcnn_inference.py`**](./handtools_ijcv_mrcnn_inference.py) and [**`handtools_ijcv_recognize_endpoints.py`**](./handtools_ijcv_recognize_endpoints.py) to:  
`Mask_RCNN/samples/handtools/`  
(Modify the parameters in code if necessary.)  

3. Copy pre-trained Mask R-CNN weigths [**`barbell.h5`**](https://data.ciirc.cvut.cz/public/projects/2021MotionForces/barbell.h5), [**`hammer.h5`**](https://data.ciirc.cvut.cz/public/projects/2021MotionForces/hammer.h5), [**`scythe.h5`**](https://data.ciirc.cvut.cz/public/projects/2021MotionForces/scythe.h5), [**`spade.h5`**](https://data.ciirc.cvut.cz/public/projects/2021MotionForces/spade.h5) to:  
`Mask_RCNN/samples/handtools/handtools/mrcnn_weights/`  

4. Copy [**`handtools_openpose.pkl`**](./handtools_openpose.pkl) to:  
`Mask_RCNN/samples/handtools/handtools/openpose/`  

5. Copy **Handtool dataset video frames** (from [https://github.com/zongmianli/Handtool-dataset](https://github.com/zongmianli/Handtool-dataset)) to:  
`Mask_RCNN/samples/handtools/handtools/frames/`  
e.g.  
`Mask_RCNN/samples/handtools/handtools/frames/spade_1/0001.png`  
`Mask_RCNN/samples/handtools/handtools/frames/spade_1/0002.png`  
etc.  

6. Copy **Handtool dataset video frames with visualized openpose** (if available) to:  
`Mask_RCNN/samples/handtools/handtools/openpose/`  
e.g.  
`Mask_RCNN/samples/handtools/handtools/openpose/spade_1/0001.png`  
`Mask_RCNN/samples/handtools/handtools/openpose/spade_1/0002.png`  
etc.  

7. Copy (replace the original files) [**`config.py`**](./config.py) and [**`model.py`**](./model.py) to:  
`Mask_RCNN/mrcnn/`  

## Usage

8. Run **Mask R-CNN inference** for each tool, e.g.:  
`cd Mask_RCNN/samples/handtools/`  
`python handtools_ijcv_mrcnn_inference.py --class_label=spade`  

9. **Estimate endpoints** from Mask R-CNN output, e.g.:  
`python handtools_ijcv_recognize_endpoints.py --class_label=spade`  

10. **Output**:  
`Mask_RCNN/samples/handtools/handtools/results/`  
e.g.  
`Mask_RCNN/samples/handtools/handtools/results/spade_1/spade_1_endpoints_mrcnn.txt`  
or  
`Mask_RCNN/samples/handtools/handtools/results/spade_1/spade_1_endpoints_mrcnn.json`  

11. **Endpoints**:  
`0001    435    352`  
`0001    152    216`  
means that in frame 0001.png the pixel coordinates of endpoints 1 (e.g. head) and 2 (e.g. handle) are (x=435, y=352) and (x=152, y=216), respectively.

12. The code is prepared for 400x600 pixel frames (see arguments **`image_height`**, **`image_width`** and **`mrcnn_image_size`** in the [**`handtools_ijcv_mrcnn_inference.py`**](handtools_ijcv_mrcnn_inference.py) code). The easiest solution is to **resize your input** to this size (possibly using cropping or zero-padding) and also linearly **rescale the Openpose data** in the [**`handtools_ijcv_recognize_endpoints.py`**](./handtools_ijcv_recognize_endpoints.py) code before you apply them.

## Image size for Mask R-CNN input:

The input size is important for the Mask R-CNN segmentation. The models were trained on 400x600 pixel images zero-padded to 608x608 (Matterport Mask R-CNN requires square input but doesn't allow 600x600 and the closest larger allowed size -- after modification of [**`model.py`**](./model.py) -- is 608x608). So the pretrained models should theoretically perform best on frames with the same (or similar) properties (400x600 padded to 608x608).

The zero-padding to 608x608 is done by the Matterport Mask R-CNN, so you can simply resize your frames to e.g. 400x600. See the **`image_height`** (400), **`image_width`** (600) and **`mrcnn_image_size`** (608) arguments in the [**`handtools_ijcv_mrcnn_inference.py`**](./handtools_ijcv_mrcnn_inference.py) code.

Q: Will anything break if I just resize the input image when the Openpose `.pkl` file is computed with the original image and is used as an input.  
A: In that case you should also modify the coordinates from Openpose (just a linear rescale) before you apply them in the [`handtools_ijcv_recognize_endpoints.py`](./handtools_ijcv_recognize_endpoints.py) code.
