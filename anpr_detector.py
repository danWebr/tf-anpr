import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
import argparse
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR,draw_ocr


parser = argparse.ArgumentParser(
    description="ANPR detection and OCR script")
parser.add_argument("-i",
                    "--image_path",
                    help="Path to the image to be processed.",
                    type=str)
parser.add_argument("-c",
                    "--custom_model",
                    help="Name of custom model.",
                    type=str)
parser.add_argument("-cp",
                    "--checkpoint_index",
                    help="Index of checkpoint.",
                    type=int)
parser.add_argument("-d",
                    "--detection_threshold",
                    help="Value for detection threshold.",
                    type=float, default=0.7)

args = parser.parse_args()


CUSTOM_MODEL_NAME = args.custom_model
LABEL_MAP_NAME = 'label_map.pbtxt'

ANNOTATION_PATH = os.path.join('Tensorflow', 'workspace','annotations')
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME)

PIPELINE_CONFIG = os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config')
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)


def isnotebook():
    try:
        get_ipython()
        return True   # Using IPython
    except NameError:
        return False      # Probably standard Python interpreter


def main():
    # Check if the script was started using IPython or standard Python
    if isnotebook():
        get_ipython().magic('matplotlib inline') # Needed for visualization


    category_index = label_map_util.create_category_index_from_labelmap(LABELMAP)
    configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)


    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-{}".format(args.checkpoint_index))).expect_partial()


    # Load image
    img = cv2.imread(args.image_path)
    image_np = np.array(img)


    # Start detection
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)


    # Extract results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize detection
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()


    # Prepare OCR
    detection_threshold = args.detection_threshold

    image = image_np_with_detections
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]


    # Apply ROI filtering and OCR, visualize the results
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

        ocr = PaddleOCR(use_angle_cls=True, lang='en') # Need to run only once to download and load model into memory
        result = ocr.ocr(region, cls=True)
        for line in result:
            print("\n\n\nDetected licence plate:")
            print(line)


if __name__ == '__main__':
    main()