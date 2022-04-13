import numpy as np
import os
from PIL import Image
import argparse

from findDerivatives import findDerivatives
from nonMaxSup import nonMaxSup
from edgeLink import edgeLink
import utils, helpers


def cannyEdge(I):
  im_gray = utils.rgb2gray(I)

  Mag, Magx, Magy, Ori = findDerivatives(im_gray)
  grad_Ori, edge_Ori = helpers.get_discrete_orientation(Ori)

  M = nonMaxSup(Mag, Ori, grad_Ori)
  E = edgeLink(M, Mag, edge_Ori)

  return E


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_folder", type=str, help="folder that contains images to compute the Canny Edge over", required=True)
  parser.add_argument("--save_folder", type=str, help="folder to save the Canny Edge to", required=True)
  opt = parser.parse_args()

  image_folder = os.path.abspath(opt.image_folder)
  save_folder = os.path.abspath(opt.save_folder)

  for filename in os.listdir(image_folder):
    im_path = os.path.join(image_folder, filename)
    I = np.array(Image.open(im_path).convert('RGB'))

    E = cannyEdge(I)
    pil_image = Image.fromarray(E.astype(bool)).convert('1')
    pil_image.save(os.path.join(save_folder, "{}_Result.png".format(filename.split(".")[0])))
