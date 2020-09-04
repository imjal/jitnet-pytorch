import numpy as np
import cv2

def mask_rcnn_unmold_cls_mask(mask, bbox, image_shape, idx, full_masks,
                              box_masks, cls, compute_box_mask=False,
                              dialate=True, threshold = 0.5):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary or weighted mask with the same size as the original image.
    """
    x1, y1, x2, y2 = bbox # y1, x1, y2, x2 = bbox
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return

    mask = cv2.resize(mask, (x2 - x1, y2 - y1)).astype(np.float32)

    thresh_mask = np.where(np.logical_and(mask >= threshold,
                                          cls > full_masks[y1:y2, x1:x2]),
                                          cls, full_masks[y1:y2, x1:x2]).astype(np.uint8)
    # Put the mask in the right location.
    full_masks[y1:y2, x1:x2] = thresh_mask

    if box_masks is not None:
        if dialate:
            dialate_frac = 0.15
            dy1 = max(int(y1 - dialate_frac * (y2 - y1)), 0)
            dx1 = max(int(x1 - dialate_frac * (x2 - x1)), 0)

            dy2 = min(int(y2 + dialate_frac * (y2 - y1)), image_shape[0])
            dx2 = min(int(x2 + dialate_frac * (x2 - x1)), image_shape[1])

            mask = cv2.resize(mask, (dx2 - dx1, dy2 - dy1)).astype(np.float32)
            box_masks[dy1:dy2, dx1:dx2] = np.where(mask >= 0, 1, 0).astype(np.bool)
        else:
            box_masks[y1:y2, x1:d2] = np.where(mask >= 0, 1, 0).astype(np.bool)

def mask_rcnn_single_mask(boxes, classes, scores, masks, image_shape,
                          box_mask=False, box_threshold=0.5,
                          mask_threshold=0.5):
    N = len(boxes)
    # Resize masks to original image size and set boundary threshold.
    full_masks = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

    box_masks = np.zeros((image_shape[0], image_shape[1]), dtype=np.bool)

    for i in range(N):
        if scores[i] < box_threshold:
            continue
        # Convert neural network mask to full size mask
        mask_rcnn_unmold_cls_mask(masks[i], boxes[i], image_shape,
                                  i, full_masks,
                                  box_masks, classes[i],
                                  compute_box_mask=box_mask,
                                  threshold=mask_threshold)
    return full_masks, box_masks

def batch_segmentation_masks(batch_size,
                             image_shape,
                             batch_boxes,
                             batch_classes,
                             batch_masks,
                             batch_scores,
                             batch_num_objects,
                             compute_weight_masks,
                             class_groups,
                             mask_threshold=0.5,
                             box_threshold=0.5,
                             scale_boxes=True):
    h = image_shape[0]
    w = image_shape[1]
    if h % 32 != 0: # dla only allows mod 32 size images
        h = (int(h/32) + 1) * 32
    if w % 32 != 0:
        w = (int(w/32) + 1) * 32

    image_shape = (h, w)

    seg_masks = np.zeros((batch_size, h, w), np.uint8)
    weight_masks = np.zeros((batch_size, h, w), np.bool)

    class_remap = {}
    for g in range(len(class_groups)):
        for c in class_groups[g]:
            class_remap[c] = g + 1

    batch_boxes = batch_boxes.copy()

    if scale_boxes and len(batch_boxes.shape) == 3:
        batch_boxes[:, :, 0] = batch_boxes[:, :, 0] * h
        batch_boxes[:, :, 2] = batch_boxes[:, :, 2] * h
        batch_boxes[:, :, 1] = batch_boxes[:, :, 1] * w
        batch_boxes[:, :, 3] = batch_boxes[:, :, 3] * w

    batch_boxes = batch_boxes.astype(np.int32)

    for b in range(batch_size):
        N = batch_num_objects[b]
        if N == 0:
            continue
        boxes = batch_boxes[b, :N, :]
        masks = batch_masks[b, :N, :, :]
        scores = batch_scores[b, :N]
        classes = batch_classes[b, :N]

        for i in range(classes.shape[0]):
            if classes[i] in class_remap:
                classes[i] = class_remap[classes[i]]
            else:
                classes[i] = 0

        idx = classes > 0
        boxes = boxes[idx]
        masks = masks[idx]
        classes = classes[idx]
        scores = scores[idx]

        full_masks, box_masks = mask_rcnn_single_mask(boxes, classes,
                                                      scores, masks,
                                                      image_shape,
                                                      box_mask=compute_weight_masks,
                                                      box_threshold=box_threshold,
                                                      mask_threshold=mask_threshold)
        seg_masks[b] = full_masks
        weight_masks[b] = box_masks

    return seg_masks, weight_masks