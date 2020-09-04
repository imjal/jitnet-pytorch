
def get_remap(class_groups):
    remap = {}
    for g in range(len(class_groups)):
        for c in class_groups[g]:
            remap[c] = g
    return remap

detectron_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

coco2bdd_class_groups = [ ['person'], [ 'car', 'bus', 'truck'], [ 'motorcycle', 'bicycle'] ]
coco_class_groups = [ [detectron_classes.index(c) for c in g] for g in coco2bdd_class_groups ]

bdd_semantic_labels = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light', 
    'traffic sign', 
    'vegetation', 
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
]
bdd2compat_class_groups = [ ['person', 'rider'], [ 'car', 'bus', 'truck'], [ 'motorcycle', 'bicycle'] ]
bdd_class_groups = [ [bdd_semantic_labels.index(c) for c in g] for g in bdd2compat_class_groups]

combined_label_names = ['human', '4 wheel vehicle', '2 wheel vechicle']