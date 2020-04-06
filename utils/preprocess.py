import numpy as np
import math
import skimage

class PreprocessImage():
    def __init__(self, config):
        """config: COCO style configuration object for the Mask RCNN.
        """
        self._config = config


    def compose_image_meta(self, image_id, original_image_shape, image_shape,
        window, scale, active_class_ids):
        """Returns input image metadata
        """
        meta = np.array(
            [image_id] + 
            list(original_image_shape) +
            list(image_shape) +
            list(window) +
            [scale] +
            list(active_class_ids)
        )

        return meta


    def mold_image(self, images):
        """Substracts mean pixel from the image and casts dtype to float32.
        """
        return images.astype(np.float32) - self._config.MEAN_PIXEL


    def compute_backbone_shapes(self, image_shape):
        """Computes Mask RCNN backbone shapes
        """
        if callable(self._config.BACKBONE):
            return self._config.COMPUTE_BACKBONE_SHAPE(image_shape)

        assert self._config.BACKBONE in ["resnet50", "resnet101"]
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)),
                int(math.ceil(image_shape[1] / stride))]
                for stride in self._config.BACKBONE_STRIDES])


    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes


    def norm_boxes(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)


    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides,
                                 anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.

        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(self.generate_anchors(scales[i], ratios, feature_shapes[i],
                                            feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)


    def get_anchors(self,image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = self.compute_backbone_shapes(image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = self.generate_pyramid_anchors(
                self._config.RPN_ANCHOR_SCALES,
                self._config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self._config.BACKBONE_STRIDES,
                self._config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = self.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]



    def resize_image(self, image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        """Resizes an image keeping the aspect ratio unchanged.

        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image
                of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None

        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = skimage.transform.resize(image, (round(h * scale), round(w * scale)),
                           preserve_range=True)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:
            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop


    def preprocess_input(self, img):
        """Pre-processes the input image.
        img: Input image of shape (-1,XX,YY,3)
        Returns:
        molded_image: Molded image to be used as model input
        image_meta: Input image metadata
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        """
        molded_image, window, scale, padding, crop = self.resize_image(
            img,
            min_dim=self._config.IMAGE_MIN_DIM,
            min_scale=self._config.IMAGE_MIN_SCALE,
            max_dim=self._config.IMAGE_MAX_DIM,
            mode=self._config.IMAGE_RESIZE_MODE
        )
        molded_image = self.mold_image(molded_image)

        image_meta = self.compose_image_meta(
            0, img.shape, molded_image.shape, window, scale,
            np.zeros([self._config.NUM_CLASSES], dtype=np.int32)
        )

        anchors = self.get_anchors(molded_image.shape)
        #anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
        return molded_image, image_meta, anchors, window
