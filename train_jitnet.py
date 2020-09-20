# import some common libraries
import numpy as np
import os, json, cv2, random, torch, tqdm, torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from progress.bar import Bar
from model.networks.jitnet import JITNet
from model.losses import CrossEntropy2d
import pdb
from PIL import Image
import PIL
import torch.nn as nn
import utils.label_mappings as label_mappings
import torchvision.transforms.functional as F
# from image_trans import get_affine_transforms


class CocoDetection:
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """


    def __init__(self, root, annFile, is_train):
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.categories = [x['name'] for x in self.coco.dataset['categories']]
        self.ignore_index = 255
        
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.is_train = is_train

    def remap(self, x):
        if x in label_mappings.detectron_classes:
            return label_mappings.detectron_classes.index(x)
        else:
            return self.ignore_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target, mask)
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        gt_label = np.zeros(img.size[::-1], dtype=np.int32)
        for i in range(len(target)):
            mask = coco.annToMask(target[i])
            gt_label[np.nonzero(mask)] = self.remap(target[i]['category_id'])
        
        img, gt_label = self.transform(img, gt_label)
        return img, gt_label

    def transform(self, image, gt_seg=None):
        # resize image to 720 x 1280 while maintaining aspect ratio (ugh should random crop at some point)
        width, height = image.size
        rewidth = 1280 / width
        reheight = 720 / height
        ratio = min(rewidth, reheight)
        new_size = (int(height * ratio), int(width * ratio))
        image = F.resize(image, new_size)
        img = np.zeros((720, 1280, 3))
        img[:new_size[0], :new_size[1]] = image
        img = F.to_tensor(img).double()

        # if we have a gt_segmentation, also convert
        if gt_seg is not None:
            gt_seg = Image.fromarray(gt_seg)
            gt_seg = F.resize(gt_seg, new_size, Image.NEAREST)
            label = np.empty((720, 1280))
            label.fill(self.ignore_index)
            label[:new_size[0], :new_size[1]] = gt_seg
            label = F.to_tensor(label)
            return img, label
        
        return img

    def __len__(self):
        return len(self.ids)

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

coco_dataset = CocoDetection('/data2/jl5/centtrack_data/coco/train2017/', '/data2/jl5/centtrack_data/coco/annotations/instances_train2017.json', is_train=True)

phase = 'train'
num_epochs = 5
ignore_index = 255
device = 'cuda:0'

data_loader = DataLoader(coco_dataset, batch_size=4, shuffle=False,
           num_workers=1, pin_memory=False, drop_last=True,
           worker_init_fn=None)

model = JITNet(80).double()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
loss_fn = CrossEntropy2d(ignore_index)

print('Starting training...')
for epoch in range(num_epochs):
    for iter_id, batch in tqdm.tqdm(enumerate(data_loader)):
        input_img = batch[0].to(device=device)
        target_img = torch.squeeze(batch[1]).to(device=device)
        output = model(input_img)
        loss = loss_fn(output, target_img)
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        del output, loss, target_img, input_img

    save_model('model_weights/model_last.pth', epoch, model, optimizer)