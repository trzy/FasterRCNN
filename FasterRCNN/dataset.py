from .models import region_proposal_network

from collections import defaultdict
import itertools
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

class VOC:
  """
  Loads the VOC dataset at `dataset_dir`. If `scale` is provided, resizes all
  images and associated metadata (e.g., box coordinates) such that the smallest
  dimension is equal to `scale`.
  """
  def __init__(self, dataset_dir, scale = None):
    self._dataset_dir = dataset_dir
    self.index_to_class_name = self._get_index_to_class_name(dataset_dir)
    train_image_paths = self._get_image_paths(dataset_dir, dataset = "train")
    val_image_paths = self._get_image_paths(dataset_dir, dataset = "val")
    self.num_samples = { "train": len(train_image_paths), "val": len(val_image_paths) }
    self._descriptions_per_image_path = {}
    self._descriptions_per_image_path["train"] = { image_path: self._get_image_description(dataset_dir, image_path = image_path, scale = scale) for image_path in train_image_paths }
    self._descriptions_per_image_path["val"] = { image_path: self._get_image_description(dataset_dir, image_path = image_path, scale = scale) for image_path in val_image_paths }

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
    def __init__(self, x_min, y_min, x_max, y_max):
      self.x_min = x_min
      self.x_max = x_max
      self.y_min = y_min
      self.y_max = y_max

    def __repr__(self):
      return "[x=%d, y=%d, width=%d, height=%d]" % (self.x_min, self.y_min, self.x_max - self.x_min + 1, self.y_max - self.y_min + 1)

    def __str__(self):
      return repr(self)

  class ImageDescription:
    def __init__(self,name, original_width, original_height, width, height, boxes_by_class_name):
      self.name = name
      self.original_width = original_width
      self.original_height = original_height
      self.width = width
      self.height = height
      self.boxes_by_class_name = boxes_by_class_name

    def shape(self):
      return (self.height, self.width, 3)

    def get_boxes(self):
      """
      Returns a list of all object bounding boxes regardless.
      """
      return list(itertools.chain.from_iterable(self.boxes_by_class_name.values()))

    def __repr__(self):
      return "[name=%s, (%d, %d), boxes=%s]" % (self.name, self.width, self.height, self.boxes_by_class_name)

  @staticmethod
  def _get_index_to_class_name(dataset_dir):
    imageset_dir = os.path.join(dataset_dir, "ImageSets", "Main")
    train_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_train.txt") ])
    val_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_val.txt") ])
    assert train_classes == val_classes, "Number of training and validation image sets in ImageSets/Main differs. Does your dataset have missing or extraneous files?"
    assert len(train_classes) > 0, "No classes found in ImageSets/Main"
    index_to_class_name = { v[0]: v[1] for v in enumerate(train_classes) }
    return index_to_class_name

  @staticmethod
  def _get_image_paths(dataset_dir, dataset):
    image_list_file = os.path.join(dataset_dir, "ImageSets", "Main", dataset + ".txt")
    with open(image_list_file) as fp:
      basenames = [ line.strip() for line in fp.readlines() ] # strip newlines
    image_paths = [ os.path.join(dataset_dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
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

  @staticmethod
  def _get_image_description(dataset_dir, image_path, scale):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_file = os.path.join(dataset_dir, "Annotations", basename) + ".xml"
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
      box = VOC.Box(x_min = x_min, y_min = y_min, x_max = x_max, y_max = y_max)
      boxes_by_class_name[class_name].append(box)
    return VOC.ImageDescription(name = basename, original_width = original_width, original_height = original_height, width = width, height = height, boxes_by_class_name = boxes_by_class_name)

  @staticmethod
  def _prepare_data(thread_num, image_paths, descriptions_per_image_path):
    print("Thread %d started" % thread_num)
    y_per_image_path = {}
    for image_path in image_paths:
      description = descriptions_per_image_path["train"][image_path]
      anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = description.shape())
      ground_truth_regressions, positive_anchors, negative_anchors = region_proposal_network.compute_anchor_label_assignments(ground_truth_object_boxes = description.get_boxes(), anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)
      y_per_image_path[image_path] = (ground_truth_regressions, positive_anchors, negative_anchors)
    print("Thread %d finished" % thread_num)
    return y_per_image_path

  def train_data(self, num_threads = 32):
    import concurrent.futures

    # Precache everything
    y_per_image_path = {}
    image_paths = list(self._descriptions_per_image_path["train"].keys())
    batch_size = len(image_paths) // num_threads + 1
    print("Spawning %d worker threads to prepare %d training samples..." % (num_threads, len(image_paths)))  

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [ executor.submit(self._prepare_data, i, image_paths[i * batch_size : i * batch_size + batch_size], self._descriptions_per_image_path) for i in range(num_threads) ]   
      results = [ f.result() for f in futures ]
      for subset_y_per_image_path in results:
        y_per_image_path.update(subset_y_per_image_path)
    print("Processed %d training samples" % len(y_per_image_path))

    while True:
      # Shuffle data each epoch
      random.shuffle(image_paths)

      # Return one image at a time 
      for image_path in image_paths:
        yield image_path, y_per_image_path[image_path]


