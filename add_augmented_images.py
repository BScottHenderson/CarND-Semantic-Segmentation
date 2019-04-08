
import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

import skimage
from skimage import transform
import cv2



def main():

    image_shape = (160, 576)  # KITTI dataset uses 160x576 images

    data_dir = './data'
    data_folder = os.path.join(data_dir, 'data_road/training')

    output_dir = os.path.join(data_folder, 'image_2')
    gt_output_dir = os.path.join(data_folder, 'gt_image_2')

    # Grab image and label paths
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    # For each image ...
    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]

        print('   image file: {}'.format(image_file))
        print('gt image file: {}'.format(gt_image_file))

        # Re-size to image_shape
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        # rotation: random with angle between 0° and 360° (uniform)
        angle = np.random.uniform(0, 360)
        image_aug = scipy.misc.imrotate(image, angle, interp='bilinear')
        gt_image_aug = scipy.misc.imrotate(gt_image, angle, interp='bilinear')
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_rotate.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_rotate.png'), gt_image_aug)

        # translation: random with shift between -10 and 10 pixels (uniform)
        shift = np.random.uniform(-10, 10)
        # M = [1 0 x]
        #     [0 1 y]
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        (rows, cols) = image_shape[:2]
        image_aug = cv2.warpAffine(image, M, (cols, rows))
        gt_image_aug = cv2.warpAffine(gt_image, M, (cols, rows))
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_trans.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_trans.png'), gt_image_aug)

        # rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
        m = (1/1.6 + 1.6) / 2
        s = (1/1.6 + 1.6) / 4
        scale_factor = np.random.lognormal(mean=m, sigma=s)
        image_shape_aug = (int(image_shape[0] * scale_factor), int(image_shape[1] * scale_factor))
        image_aug = cv2.resize(image, dsize=image_shape_aug, interpolation=cv2.INTER_CUBIC)
        gt_image_aug = cv2.resize(gt_image, dsize=image_shape_aug, interpolation=cv2.INTER_CUBIC)
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_scale.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_scale.png'), gt_image_aug)

        # flipping: yes or no (bernoulli)
        flip_axis = np.random.binomial(size=1, n=1, p=0.5)  # n=1 -> Bournoulli distribution
        image_aug = cv2.flip(image, flip_axis)
        gt_image_aug = cv2.flip(gt_image, flip_axis)
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_flip.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_flip.png'), gt_image_aug)

        # shearing: random with angle between -20° and 20° (uniform)
        angle = np.random.uniform(-20, 20)
        angle_radians = angle / 180. * np.pi
        affine_tf = transform.AffineTransform(shear=angle_radians)  # Create Affine transform
        image_aug = transform.warp(image, inverse_map=affine_tf)    # Apply transform to image data
        gt_image_aug = transform.warp(gt_image, inverse_map=affine_tf)
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_shear.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_shear.png'), gt_image_aug)

        # stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)
        m = (1/1.3 + 1.3) / 2
        s = (1/1.3 + 1.3) / 4
        stretch_factor = np.random.lognormal(mean=m, sigma=s)
        a = 1/1.3 + stretch_factor
        b = 1.3 - stretch_factor
        image_aug = skimage.exposure.rescale_intensity(image, in_range=(a, b))
        gt_image_aug = skimage.exposure.rescale_intensity(gt_image, in_range=(a, b))
        scipy.misc.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '_stretch.png'), image_aug)
        scipy.misc.imsave(os.path.join(gt_output_dir, os.path.splitext(os.path.basename(gt_image_file))[0] + '_stretch.png'), gt_image_aug)


if __name__ == '__main__':
    main()
