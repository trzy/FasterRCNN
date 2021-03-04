from .dataset import VOC

import imageio
from PIL import Image, ImageDraw

def draw_rectangle(image, x_min, y_min, x_max, y_max, color, thickness = 4):
  ctx = ImageDraw.Draw(image)
  ctx.line(xy = [(x_min, y_min), (x_max, y_min)], width = thickness, fill = color)
  ctx.line(xy = [(x_max, y_min), (x_max, y_max)], width = thickness, fill = color)
  ctx.line(xy = [(x_max, y_max), (x_min, y_max)], width = thickness, fill = color)
  ctx.line(xy = [(x_min, y_max), (x_min, y_min)], width = thickness, fill = color)

def show_annotated_image(voc, filename):
  # Load image
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_description(path = filepath)
  data = imageio.imread(filepath, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  
  # Because we loaded it ourselves, we are responsible for rescaling it
  image = image.resize((info.width, info.height), resample = Image.BILINEAR)
  ctx = ImageDraw.Draw(image)

  # Draw ground truth boxes in green
  for box in info.get_boxes():
    draw_rectangle(image, x_min = box.x_min, y_min = box.y_min, x_max = box.x_max, y_max = box.y_max, color = (0, 255, 0))

  image.show()

