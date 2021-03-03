from collections import defaultdict
import itertools
import os
from pathlib import Path
import xml.etree.ElementTree as ET

class VOC:
  """
  Loads the VOC dataset at `dataset_dir`. If `scale` is provided, resizes all
  images and associated metadata (e.g., box coordinates) such that the smallest
  dimension is equal to `scale`.
  """
  def __init__(self, dataset_dir, scale = None):
    self.index_to_class_name = self._get_index_to_class_name(dataset_dir)
    train_image_paths = self._get_image_paths(dataset_dir, dataset = "train")
    val_image_paths = self._get_image_paths(dataset_dir, dataset = "val")
    self._descriptions_per_image_path = {}
    self._descriptions_per_image_path["train"] = { image_path: self._get_image_description(dataset_dir, image_path = image_path, scale = scale) for image_path in train_image_paths }
    self._descriptions_per_image_path["val"] = { image_path: self._get_image_description(dataset_dir, image_path = image_path, scale = scale) for image_path in train_image_paths }

  def get_boxes_per_image_path(self, dataset):
    """
    Returns a dictionary where the key is image path and the value is a list of
    Box structures.
    """
    assert dataset == "train" or dataset == "val"
    # For each image, get the values from boxes_by_class_name and join them into a single list
    boxes_per_image_path = { path: list(itertools.chain.from_iterable(image_description.boxes_by_class_name.values())) for path, image_description in self._descriptions_per_image_path[dataset].items() }
    return boxes_per_image_path

  class Box:
    def __init__(self, xmin, ymin, xmax, ymax):
      self.xmin = xmin
      self.xmax = xmax
      self.ymin = ymin
      self.ymax = ymax

    def __repr__(self):
      return "[x=%d, y=%d, width=%d, height=%d]" % (self.xmin, self.ymin, self.xmax - self.xmin + 1, self.ymax - self.ymin + 1)

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
      original_xmin = int(bndbox.find("xmin").text)
      original_ymin = int(bndbox.find("ymin").text)
      original_xmax = int(bndbox.find("xmax").text)
      original_ymax = int(bndbox.find("ymax").text)
      xmin = original_xmin * scale_factor
      ymin = original_ymin * scale_factor
      xmax = original_xmax * scale_factor
      ymax = original_ymax * scale_factor
      print("width: %d -> %d\theight: %d -> %d\txmin: %d -> %d\tymin: %d -> %d" % (original_width, width, original_height, height, original_xmin, xmin, original_ymin, ymin))
      box = VOC.Box(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
      boxes_by_class_name[class_name].append(box)
    return VOC.ImageDescription(name = basename, original_width = original_width, original_height = original_height, width = width, height = height, boxes_by_class_name = boxes_by_class_name)
