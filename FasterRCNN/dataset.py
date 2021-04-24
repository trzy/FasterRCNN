from .models import region_proposal_network

from collections import defaultdict
import itertools
import os
from operator import itemgetter
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import imageio
from PIL import Image
import numpy as np
import tensorflow as tf

class VOC:
  """
  Loads the VOC dataset at `dataset_dir`. If `scale` is provided, resizes all
  images and associated metadata (e.g., box coordinates) such that the smallest
  dimension is equal to `scale`.
  """
  def __init__(self, dataset_dir, scale = None):
    print("VOC dataset: Parsing metadata...")
    self._dataset_dir = dataset_dir
    self.index_to_class_name, self.class_name_to_index = self._get_index_to_class_name(dataset_dir)
    self.num_classes = len(self.index_to_class_name.keys()) # number of classes including background class
    train_image_paths = self._get_image_paths(dataset_dir, dataset = "train")
    val_image_paths = self._get_image_paths(dataset_dir, dataset = "val")
    self.num_samples = { "train": len(train_image_paths), "val": len(val_image_paths) }
    self._descriptions_per_image_path = {}
    self._descriptions_per_image_path["train"] = { image_path: self._get_image_description(image_path = image_path, scale = scale) for image_path in train_image_paths }
    self._descriptions_per_image_path["val"] = { image_path: self._get_image_description(image_path = image_path, scale = scale) for image_path in val_image_paths }

  def get_full_path(self, filename):
    return os.path.join(self._dataset_dir, "JPEGImages", filename)

  def get_image_description(self, path):
    # Image names are unique, so we don't need to specify the dataset
    if path in self._descriptions_per_image_path["train"]:
      return self._descriptions_per_image_path["train"][path]
    if path in self._descriptions_per_image_path["val"]:
      return self._descriptions_per_image_path["val"][path]
    raise Exception("Image path not found: %s" % path)

  def get_boxes_per_image_path(self, dataset):
    """
    Returns a dictionary where the key is image path and the value is a list of
    Box structures.
    """
    assert dataset == "train" or dataset == "val"
    # For each image, get the values from boxes_by_class_name and join them into a single list
    boxes_per_image_path = { path: image_description.get_boxes() for path, image_description in self._descriptions_per_image_path[dataset].items() }
    return boxes_per_image_path

  class Box:
    def __init__(self, x_min, y_min, x_max, y_max, class_index):
      self.x_min = x_min
      self.x_max = x_max
      self.y_min = y_min
      self.y_max = y_max
      self.class_index = class_index

    def __repr__(self):
      return "[x=%d, y=%d, width=%d, height=%d, class=%d]" % (self.x_min, self.y_min, self.x_max - self.x_min + 1, self.y_max - self.y_min + 1, self.class_index)

    def __str__(self):
      return repr(self)

  #TODO: rename to ImageInfo
  class ImageDescription:
    def __init__(self, name, path, original_width, original_height, width, height, boxes_by_class_name):
      self.name = name
      self.path = path
      self.original_width = original_width
      self.original_height = original_height
      self.width = width
      self.height = height
      self.boxes_by_class_name = boxes_by_class_name

    def load_image_data(self):
      data = imageio.imread(self.path, pilmode = "RGB")
      image = Image.fromarray(data, mode = "RGB").resize((self.width, self.height))
      image = np.array(image)
      return tf.keras.applications.vgg16.preprocess_input(x = image)

    def shape(self):
      return (self.height, self.width, 3)

    def get_boxes(self):
      """
      Returns a list of all object bounding boxes regardless.
      """
      return list(itertools.chain.from_iterable(self.boxes_by_class_name.values()))

    def get_complete_ground_truth_regressions_map(self):
      anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = self.shape())
      ground_truth_regressions, positive_anchors, negative_anchors = region_proposal_network.compute_anchor_label_assignments(ground_truth_object_boxes = self.get_boxes(), anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)
      for anchor_position in positive_anchors + negative_anchors:
        y = anchor_position[0]
        x = anchor_position[1]
        k = anchor_position[2]
        ground_truth_regressions[y,x,k,0] = 1.0
      #print("pos=%d neg=%d count=%f" % (len(positive_anchors), len(negative_anchors), np.sum(ground_truth_regressions[:,:,:,0])))
      return ground_truth_regressions

    def __repr__(self):
      return "[name=%s, (%d, %d), boxes=%s]" % (self.name, self.width, self.height, self.boxes_by_class_name)

  @staticmethod
  def _get_index_to_class_name(dataset_dir):
    """
    Returns mappings between class index and class name. Indices are in the 
    range [1,N], where N is the number of VOC classes, because index 0 is
    reserved for use as a background class in the final classifier network.
    """
    imageset_dir = os.path.join(dataset_dir, "ImageSets", "Main")
    train_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_train.txt") ])
    val_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_val.txt") ])
    assert train_classes == val_classes, "Number of training and validation image sets in ImageSets/Main differs. Does your dataset have missing or extraneous files?"
    assert len(train_classes) > 0, "No classes found in ImageSets/Main"
    index_to_class_name = { (1 + v[0]): v[1] for v in enumerate(sorted(train_classes)) }
    class_name_to_index = { v[1]: (1 + v[0]) for v in enumerate(sorted(train_classes)) }
    index_to_class_name[0] = "background"
    class_name_to_index["background"] = 0
    return index_to_class_name, class_name_to_index

  @staticmethod
  def _get_image_paths(dataset_dir, dataset):
    image_list_file = os.path.join(dataset_dir, "ImageSets", "Main", dataset + ".txt")
    with open(image_list_file) as fp:
      basenames = [ line.strip() for line in fp.readlines() ] # strip newlines
    image_paths = [ os.path.join(dataset_dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
    # Debug: 60 car training images
    image_paths = [
      "2008_000028",
      "2008_000074",
      "2008_000085",
      "2008_000105",
      "2008_000109",
      "2008_000143",
      "2008_000176",
      "2008_000185",
      "2008_000187",
      "2008_000189",
      "2008_000193",
      "2008_000199",
      "2008_000226",
      "2008_000237",
      "2008_000252",
      "2008_000260",
      "2008_000315",
      "2008_000346",
      "2008_000356",
      "2008_000399",
      "2008_000488",
      "2008_000531",
      "2008_000563",
      "2008_000583",
      "2008_000595",
      "2008_000613",
      "2008_000619",
      "2008_000719",
      "2008_000833",
      "2008_000944",
      "2008_000953",
      "2008_000959",
      "2008_000979",
      "2008_001018",
      "2008_001039",
      "2008_001042",
      "2008_001104",
      "2008_001169",
      "2008_001196",
      "2008_001208",
      "2008_001274",
      "2008_001329",
      "2008_001359",
      "2008_001375",
      "2008_001440",
      "2008_001446",
      "2008_001500",
      "2008_001533",
      "2008_001541",
      "2008_001631",
      "2008_001632",
      "2008_001716",
      "2008_001746",
      "2008_001860",
      "2008_001941",
      "2008_002062",
      "2008_002118",
      "2008_002197",
      "2008_002202",
      "2011_003247"
    ]
    return [ os.path.join(dataset_dir, "JPEGImages", path) + ".jpg" for path in image_paths ]
    return image_paths

  @staticmethod
  def _compute_scale_factor(original_width, original_height, new_scale):
    if not new_scale:
      return 1.0
    return (new_scale / original_height) if original_width > original_height else (new_scale / original_width)

  @staticmethod
  def _compute_new_scale(original_width, original_height, new_scale):
    if not new_scale:
      return (original_width, original_height)
    if original_width > original_height:
      new_width = (original_width / original_height) * new_scale
      new_height = new_scale
    else:
      new_height = (original_height / original_width) * new_scale
      new_width = new_scale
    return (int(new_width), int(new_height))

  def _get_image_description(self, image_path, scale):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_file = os.path.join(self._dataset_dir, "Annotations", basename) + ".xml"
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    assert tree != None, "Failed to parse %s" % annotation_file
    assert len(root.findall("size")) == 1
    size = root.find("size")
    assert len(size.findall("width")) == 1
    assert len(size.findall("height")) == 1
    assert len(size.findall("depth")) == 1
    original_width = int(size.find("width").text)
    original_height = int(size.find("height").text)
    width, height = VOC._compute_new_scale(original_width = original_width, original_height = original_height, new_scale = scale)
    scale_factor = VOC._compute_scale_factor(original_width = original_width, original_height = original_height, new_scale = scale)
    depth = int(size.find("depth").text)
    assert depth == 3
    boxes_by_class_name = defaultdict(list)
    for obj in root.findall("object"):
      #TODO: use "difficult" attribute to optionally exclude difficult images?
      assert len(obj.findall("name")) == 1
      assert len(obj.findall("bndbox")) == 1
      class_name = obj.find("name").text
      bndbox = obj.find("bndbox")
      assert len(bndbox.findall("xmin")) == 1
      assert len(bndbox.findall("ymin")) == 1
      assert len(bndbox.findall("xmax")) == 1
      assert len(bndbox.findall("ymax")) == 1
      original_x_min = int(bndbox.find("xmin").text)
      original_y_min = int(bndbox.find("ymin").text)
      original_x_max = int(bndbox.find("xmax").text)
      original_y_max = int(bndbox.find("ymax").text)
      x_min = original_x_min * scale_factor
      y_min = original_y_min * scale_factor
      x_max = original_x_max * scale_factor
      y_max = original_y_max * scale_factor
      #print("width: %d -> %d\theight: %d -> %d\tx_min: %d -> %d\ty_min: %d -> %d" % (original_width, width, original_height, height, original_x_min, x_min, original_y_min, y_min))
      box = VOC.Box(x_min = x_min, y_min = y_min, x_max = x_max, y_max = y_max, class_index = self.class_name_to_index[class_name])
      boxes_by_class_name[class_name].append(box)
    return VOC.ImageDescription(name = basename, path = image_path, original_width = original_width, original_height = original_height, width = width, height = height, boxes_by_class_name = boxes_by_class_name)

  @staticmethod
  def _create_anchor_minibatch(positive_anchors, negative_anchors, mini_batch_size, image_path):
    """
    Returns N=mini_batch_size anchors, trying to get as close to a 1:1 ratio as
    possible with no more than 50% of the samples being positive.
    """
    assert len(positive_anchors) + len(negative_anchors) >= mini_batch_size, "Image has insufficient anchors for mini_batch_size=%d: %s" % (mini_batch_size, image_path)
    assert len(positive_anchors) > 0, "Image does not have any positive anchors: %s" % image_path
    assert mini_batch_size % 2 == 0, "mini_batch_size must be evenly divisible"

    num_positive_anchors = len(positive_anchors)
    num_negative_anchors = len(negative_anchors)

    num_positive_samples = min(mini_batch_size // 2, num_positive_anchors)  # up to half the samples should be positive, if possible
    num_negative_samples = mini_batch_size - num_positive_samples           # the rest should be negative
    positive_sample_indices = random.sample(range(num_positive_anchors), num_positive_samples)
    negative_sample_indices = random.sample(range(num_negative_anchors), num_negative_samples)

    positive_samples = list(map(positive_anchors.__getitem__, positive_sample_indices)) #list(itemgetter(*positive_sample_indices)(positive_anchors))
    negative_samples = list(map(negative_anchors.__getitem__, negative_sample_indices)) #list(itemgetter(*negative_sample_indices)(negative_anchors))

    return positive_samples, negative_samples

  @staticmethod
  def _prepare_data(thread_num, image_paths, descriptions_per_image_path):
    print("VOC dataset: Thread %d started" % thread_num)
    y_per_image_path = {}
    anchor_boxes_per_image_path = {}
    for image_path in image_paths:
      description = descriptions_per_image_path["train"][image_path]
      anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = description.shape())
      ground_truth_regressions, positive_anchors, negative_anchors = region_proposal_network.compute_anchor_label_assignments(ground_truth_object_boxes = description.get_boxes(), anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)
      y_per_image_path[image_path] = (ground_truth_regressions, positive_anchors, negative_anchors)
      anchor_boxes_per_image_path[image_path] = anchor_boxes
    print("VOC dataset: Thread %d finished" % thread_num)
    return y_per_image_path, anchor_boxes_per_image_path

  def _standardize_regressions(self, y_per_image_path):
    ty = []
    tx = []
    th = []
    tw = []

    # Extract all regression components for every single positive anchor
    for _, (ground_truth_regressions, _, _) in y_per_image_path.items():
      for y in range(ground_truth_regressions.shape[0]):
        for x in range(ground_truth_regressions.shape[1]):
          for k in range(ground_truth_regressions.shape[2]):
            is_object = ground_truth_regressions[y,x,k,1] > 0
            if is_object:
              ty.append(ground_truth_regressions[y,x,k,4])
              tx.append(ground_truth_regressions[y,x,k,5])
              th.append(ground_truth_regressions[y,x,k,6])
              tw.append(ground_truth_regressions[y,x,k,7])

    # Compute mean and standard deviation for each
    ty = np.array(ty)
    tx = np.array(tx)
    th = np.array(th)
    tw = np.array(tw)
    ty_mean = np.mean(ty)
    tx_mean = np.mean(tx)
    th_mean = np.mean(th)
    tw_mean = np.mean(tw)
    ty_stdev = np.std(ty)
    tx_stdev = np.std(tx)
    th_stdev = np.std(th)
    tw_stdev = np.std(tw)

    # TODO: figure out where to properly store these
    means = { "tx": tx_mean, "ty": ty_mean, "tw": tw_mean, "th": th_mean }
    stdevs = { "tx": tx_stdev, "ty": ty_stdev, "tw": tw_stdev, "th": th_stdev }
    print("means = ", means)
    print("stdevs = ", stdevs)

    # Standardize the ground truth data
    means = np.array([ ty_mean, tx_mean, th_mean, tw_mean ])
    stdevs = np.array([ ty_stdev, tx_stdev, th_stdev, tw_stdev ])
    for _, (ground_truth_regressions, _, _) in y_per_image_path.items():
      ground_truth_regressions[:,:,:,4:8] -= means
      ground_truth_regressions[:,:,:,4:8] /= stdevs

  # TODO: remove limit_samples. It is not correct because self.num_samples will never match it.
  def train_data(self, mini_batch_size = 256, shuffle = True, num_threads = 16, limit_samples = None, cache_images = False):
    import concurrent.futures

    # Precache anchor label assignments
    y_per_image_path = {}
    anchor_boxes_per_image_path = {}
    image_paths = list(self._descriptions_per_image_path["train"].keys())
    if limit_samples:
      image_paths = image_paths[0:limit_samples]
    batch_size = len(image_paths) // num_threads + 1
    print("VOC dataset: Spawning %d worker threads to process %d training samples..." % (num_threads, len(image_paths)))

    tic = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [ executor.submit(self._prepare_data, i, image_paths[i * batch_size : i * batch_size + batch_size], self._descriptions_per_image_path) for i in range(num_threads) ]
      results = [ f.result() for f in futures ]
      for subset_y_per_image_path, subset_anchor_boxes_per_image_path in results:
        y_per_image_path.update(subset_y_per_image_path)
        anchor_boxes_per_image_path.update(subset_anchor_boxes_per_image_path)
    toc = time.perf_counter()
    print("VOC dataset: Processed %d training samples in %1.1f minutes" % (len(y_per_image_path), ((toc - tic) / 60.0)))

    # Image cache
    cached_image_by_path = {}

    # Iterate
    while True:
      # Shuffle data each epoch
      if shuffle:
        random.shuffle(image_paths)

      # Return one image at a time
      for image_path in image_paths:
        # Load image
        image_data = None
        if cache_images and image_path in cached_image_by_path:
          image_data = cached_image_by_path[image_path]
        if image_data is None:  # NumPy array -- cannot test for == None or "is None"
          image_data = self._descriptions_per_image_path["train"][image_path].load_image_data()
          if cache_images:
            cached_image_by_path[image_path] = image_data

        # Retrieve pre-computed y value and anchor boxes
        ground_truth_regressions, positive_anchors, negative_anchors = y_per_image_path[image_path]
        anchor_boxes = anchor_boxes_per_image_path[image_path]

        # Observed: the maximum number of positive anchors in in a VOC image is 102.
        # Is this a bug in our code? Paper talks about a 1:1 (128:128) ratio.

        # Randomly choose anchors to use in this mini-batch
        positive_anchors, negative_anchors = self._create_anchor_minibatch(positive_anchors = positive_anchors, negative_anchors = negative_anchors, mini_batch_size = mini_batch_size, image_path = image_path)

        # Mark which anchors to use in the map
        ground_truth_regressions[:,:,:,0] = 0.0 # clear out previous (we re-use the same object)
        for anchor_position in positive_anchors + negative_anchors:
          y = anchor_position[0]
          x = anchor_position[1]
          k = anchor_position[2]
          ground_truth_regressions[y,x,k,0] = 1.0

        yield image_path, image_data, ground_truth_regressions, anchor_boxes
