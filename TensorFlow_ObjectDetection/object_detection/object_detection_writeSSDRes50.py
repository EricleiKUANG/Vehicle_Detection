# coding: utf-8
 
# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.
import sys
sys.path.append('/nfs/private/tfmodels/research/') # point to your tensorflow dir
sys.path.append('/nfs/private/tfmodels/research/slim') # point ot your slim dir
# # Imports
import numpy as np
import os

print sys.path
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
#cap = cv2.VideoCapture(0)  #


# cap = cv2.VideoCapture("car.mp4")
# cap = cv2.VideoCapture("DJI_0004.MOV")
import datetime
from timeit import default_timer as timer
import time
#if tf.__version__ != '1.4.0':
 # raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
 

# ## Env setup
flags = tf.app.flags 
 
 
# This is needed to display the images.
# get_ipython().magic(u'matplotlib inline')
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#from utils import ops as utils_ops

# ## Object detection imports
# Here are the imports from the object detection module.
 
 
 
 
from utils import label_map_util

from utils import visualization_utils as vis_util
 
 
# # Model preparation
 
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
 
 
 
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
flags.DEFINE_string('model_name','image_tensor',' ')
flags.DEFINE_string('path_to_ckpt',None,' ')
flags.DEFINE_string('outputimagefolder',None,' ')
flags.DEFINE_string('path_to_labels',None,' ')
flags.DEFINE_string('path_to_test_image',None,' ')
flags.DEFINE_string('numofclasses','0',' ')
flags.DEFINE_string('txt_path',None,' ')
FLAGS = flags.FLAGS
#MODEL_NAME = '/nfs/private/tfmodels/research/My_object_detection_Faster_rcnn_resnet101v2'
MODEL_NAME = FLAGS.model_name
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/out/frozen_inference_graph.pb'

PATH_TO_CKPT = FLAGS.path_to_ckpt
#newimagefolder =  '/nfs/private/tfmodels/research/My_object_detection_Faster_rcnn_resnet101v2/testimage/'
newimagefolder = FLAGS.outputimagefolder
# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('/nfs/private/tfmodels/research/My_object_detection_Faster_rcnn_resnet101v2/data', 'BDD100k_label_map.pbtxt')
PATH_TO_LABELS = FLAGS.path_to_labels
NUM_CLASSES = int(FLAGS.numofclasses)
 
 
# ## Download Model
 
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())
 
 
# ## Load a (frozen) Tensorflow model into memory.
 
 
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
 
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
 
 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
 
# ## Helper code
 
 
 
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 
 
# # Detection
 
 
 
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

#PATH_TO_TEST_IMAGES_DIR = '/nfs/project/kuanglei_i/BDD-100k/bdd_100k/images/100k/val'
PATH_TO_TEST_IMAGES_DIR = FLAGS.path_to_test_image
images=os.listdir(PATH_TO_TEST_IMAGES_DIR)
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
TEST_IMAGE_PATHS=[]
for image_name in images:
    if(str(image_name.split(".")[-1])=="jpg"):
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name))
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,images) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
 
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def run_inference_for_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []

            for image in images:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                #print('boxes: '+str(output_dict['detection_boxes']))
                #print(len(output_dict['detection_boxes']))
                #print('class: '+str(output_dict['detection_classes']))
                #print(len(output_dict['detection_scores']))
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                output_dict_array.append(output_dict)

    return output_dict_array


images = []
imagesname = []
width = []
height = []
num = 0
length = len(TEST_IMAGE_PATHS)
for image_path in TEST_IMAGE_PATHS:
  

  image = Image.open(image_path)
  width0, height0 = image.size
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  images.append(image_np)
  imagesname.append(image_path)
  width.append(width0)
  height.append(height0)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  num += 1
  print(str(num)+'/'+str(length))
  #if num >= 5:
     # break
#print len(images)
now = datetime.datetime.now()
print(str(now) + " : count is : " + str(len(images)))
start = timer()
  # Actual detection.
output_dict_array   = run_inference_for_images(images, detection_graph)
end = timer()
avg = (end-start) / len(images)
fps = 1/avg
print("TF inferencing took: "+str(end - start) +" for ["+str(len(images))+"] images, average["+str(avg)+"], FPS["+str(fps)+"]")
boxes = []
classes = []
scores = []
txtpath = FLAGS.txt_path
if not os.path.exists(txtpath):
    os.makedirs(txtpath)
# Visualization of the results of a detection.
for idx in range(len(output_dict_array)):
  output_dict = output_dict_array[idx]
  image_np_org = images[idx]
  #print(image_np_org)
  #print(output_dict)
  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']
  i = 0
  imagesname[idx] = imagesname[idx].split('/')[-1]
  imagesname[idx] = imagesname[idx].split('.')[0]
  
  #print len(classes)
  #print len(boxes)
  #print len(scores)
  #exit(1)
  while(i<len(classes)):
      #print imagesname[idx],scores[i],boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
      if scores[i] > 0.01:
        f = open(txtpath+str(int(classes[i])-1)+'.txt','a+')
        f.write(str(imagesname[idx])+' '+'%.6f'%(scores[i])+' '+'%.6f'%(boxes[i][1]*width[idx])+' '+'%.6f'%(boxes[i][0]*height[idx])+' '+'%.6f'%(boxes[i][3]*width[idx])+' '+'%.6f'%(boxes[i][2]*height[idx])+'\n')
        f.close()
      i = i + 1
  count = idx+1
  print str(count)+'/'+str(len(output_dict_array))+'done'
  #cv2.imwrite(newimagefolder + 'img'+str(idx)+'.jpg', image_np_org)
#plt.figure(figsize=IMAGE_SIZE)
#plt.imshow(image_np)

'''
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
 
        start = time.time()
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        end = time.time()
        # Time elapsed
        seconds = end - start
        print( "Time taken : {0} seconds".format(seconds))
        # Calculate frames per second
        fps  = 1 / seconds;
        print( "Estimated frames per second : {0}".format(fps));
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    '''
