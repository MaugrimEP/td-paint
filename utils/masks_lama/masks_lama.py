import hashlib
import logging
import math
import random
from dataclasses import asdict
from enum import Enum

import cv2
import numpy as np

from conf.mask_lama_params import (IrregularMaskParams, OutpaintingParams,
                                   RectangleMaskParams, SegmentationParams,
                                   SuperResParams)


# from saicinpainting.utils import LinearRamp
class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


LOGGER = logging.getLogger(__name__)


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)


class RandomSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.impl = None  # will be instantiated in first call (effectively in subprocess)
        self.kwargs = kwargs

    def __call__(self, img, iter_i=None, raw_image=None):
        if self.impl is None:
            self.impl = SegmentationMask(**self.kwargs)

        masks = self.impl.get_masks(np.transpose(img, (1, 2, 0)))
        masks = [m for m in masks if len(np.unique(m)) > 1]
        return np.random.choice(masks)


def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


class RandomSuperresMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img, iter_i=None):
        return make_random_superres_mask(img.shape[1:], **self.kwargs)


class DumbAreaMaskGenerator:
    min_ratio = 0.1
    max_ratio = 0.35
    default_ratio = 0.225

    def __init__(self, is_training):
        #Parameters:
        #    is_training(bool): If true - random rectangular mask, if false - central square mask
        self.is_training = is_training

    def _random_vector(self, dimension):
        if self.is_training:
            lower_limit = math.sqrt(self.min_ratio)
            upper_limit = math.sqrt(self.max_ratio)
            mask_side = round((random.random() * (upper_limit - lower_limit) + lower_limit) * dimension)
            u = random.randint(0, dimension-mask_side-1)
            v = u+mask_side 
        else:
            margin = (math.sqrt(self.default_ratio) / 2) * dimension
            u = round(dimension/2 - margin)
            v = round(dimension/2 + margin)
        return u, v

    def __call__(self, img, iter_i=None, raw_image=None):
        c, height, width = img.shape
        mask = np.zeros((height, width), np.float32)
        x1, x2 = self._random_vector(width)
        y1, y2 = self._random_vector(height)
        mask[x1:x2, y1:y2] = 1
        return mask[None, ...]


class OutpaintingMaskGenerator:
    def __init__(self, min_padding_percent:float=0.04, max_padding_percent:int=0.25, left_padding_prob:float=0.5, top_padding_prob:float=0.5, 
                 right_padding_prob:float=0.5, bottom_padding_prob:float=0.5, is_fixed_randomness:bool=False):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [left_padding_prob, top_padding_prob, right_padding_prob, bottom_padding_prob]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert len([x for x in [self.min_padding_percent, self.max_padding_percent] if (x>=0 and x<=1)]) == 2, f"Padding percentage should be in [0,1]"
        assert sum(self.probs) > 0, f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4, f"At least one of padding probs is not in [0,1] - {self.probs}"
        if len([x for x in self.probs if x > 0]) == 1:
            LOGGER.warning(f"Only one padding prob is greater than zero - {self.probs}. That means that the outpainting masks will be always on the same side")

    def apply_padding(self, mask, coord):
        mask[int(coord[0][0]*self.img_h):int(coord[1][0]*self.img_h),   
             int(coord[0][1]*self.img_w):int(coord[1][1]*self.img_w)] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent*size)
        n2 = int(self.max_padding_percent*size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash)%(2**32)
        return res

    def __call__(self, img, iter_i=None, raw_image=None):
        c, self.img_h, self.img_w = img.shape
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, f"Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [[
                   (0,0), 
                   (1,self.get_padding(size=self.img_h))
                  ],
                  [
                    (0,0),
                    (self.get_padding(size=self.img_w),1)
                  ],
                  [
                    (0,1-self.get_padding(size=self.img_h)),
                    (1,1)
                  ],    
                  [
                    (1-self.get_padding(size=self.img_w),0),
                    (1,1)
                  ]]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(range(len(coords)), p=np.array(self.probs)/sum(self.probs))
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask[None, ...]


class MixedMaskGenerator:
    def __init__(
        self,
        irregular_proba  : float, irregular_kwargs  : IrregularMaskParams,
        box_proba        : float, box_kwargs        : RectangleMaskParams,
        segm_proba       : float, segm_kwargs       : SegmentationParams,
        squares_proba    : float, squares_kwargs    : IrregularMaskParams,
        superres_proba   : float, superres_kwargs   : SuperResParams,
        outpainting_proba: float, outpainting_kwargs: OutpaintingParams,
        # invert_proba     : float,
    ):
        self.probas = []
        self.gens = []
        self.invert_probas = []

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            irregular_kwargs = asdict(irregular_kwargs)
            irregular_kwargs['ramp_kwargs'] = irregular_kwargs.pop('ramp_params', None)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.invert_probas.append(irregular_kwargs.pop('invert_proba'))
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            box_kwargs = asdict(box_kwargs)
            box_kwargs['ramp_kwargs'] = box_kwargs.pop('ramp_params', None)
            self.invert_probas.append(box_kwargs.pop('invert_proba'))
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if segm_proba > 0:
            self.probas.append(segm_proba)
            segm_kwargs = asdict(segm_kwargs)
            self.invert_probas.append(segm_kwargs.pop('invert_proba'))
            self.gens.append(RandomSegmentationMaskGenerator(**segm_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            squares_kwargs = asdict(squares_kwargs)
            squares_kwargs['ramp_kwargs'] = squares_kwargs.pop('ramp_params', None)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.invert_probas.append(squares_kwargs.pop('invert_proba'))
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            superres_kwargs = asdict(superres_kwargs)
            self.invert_probas.append(superres_kwargs.pop('invert_proba'))
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        if outpainting_proba > 0:
            self.probas.append(outpainting_proba)
            outpainting_kwargs = asdict(outpainting_kwargs)
            self.invert_probas.append(outpainting_kwargs.pop('invert_proba'))
            self.gens.append(OutpaintingMaskGenerator(**outpainting_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        # self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        invert_proba = self.invert_probas[kind]
        result = gen(img, iter_i=iter_i)
        if invert_proba > 0 and random.random() < invert_proba:
            result = 1 - result
        return result


def get_mask_generator(kind, kwargs):
    if kind is None:
        kind = "mixed"
    if kwargs is None:
        kwargs = {}

    if kind == "mixed":
        cl = MixedMaskGenerator
    elif kind == "outpainting":
        cl = OutpaintingMaskGenerator
    elif kind == "dumb":
        cl = DumbAreaMaskGenerator
    else:
        raise NotImplementedError(f"No such generator kind = {kind}")
    return cl(**kwargs)
