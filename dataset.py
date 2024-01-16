## Author: Lishuo Pan 2020/4/18

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import matplotlib.patches as patches
from PIL import ImageColor, ImageFont
import cv2
import os

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # pass
        # TODO: load dataset, make mask list'
        # load the data into data.Dataset
        # imgs_path, masks_path, labels_path, bboxes_path = path
        imgs_path = path[0]
        masks_path = path[1]
        labels_path = path[2]
        bboxes_path = path[3]
        self.imgs_data = h5py.File(imgs_path, 'r')
        self.masks_data = h5py.File(masks_path, 'r')
        self.labels_data = np.load(labels_path, allow_pickle=True)
        self.bboxes_data = np.load(bboxes_path, allow_pickle=True)
        # self.labels_data = self.labels_data.item()
        # self.bboxes_data = self.bboxes_data.item()
        self.imgs_data = self.imgs_data['data']
        self.masks_data = self.masks_data['data']
        self.mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pad = torch.nn.ZeroPad2d((11, 11, 0, 0))

        self.ordered_mask = []
        for label in self.labels_data:
            self.ordered_mask.append(self.masks_data[:label.shape[0]])
            self.masks_data = self.masks_data[label.shape[0]:]

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        transed_mask = self.ordered_mask[index]
        label = torch.tensor(self.labels_data[index])
        transed_bbox = self.bboxes_data[index]
        # print("transed_bbox: ", transed_bbox.shape)
        # print(transed_bbox)
        transed_img = self.imgs_data[index]
        # get the data using pre_process_batch
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(transed_img, transed_mask, transed_bbox)
        # check flag
        # print("transed_bbox: 2", transed_bbox)
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    
    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        #define the transform in 3 steps by defining resize and normalize separetly
        resize = transforms.Resize((800, 1066))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        #transform the image
        transform = transforms.Compose([resize, normalize])
        img_p = self.pad(transform(torch.tensor(img/255.0, dtype = torch.float32)))
        # mask_p = torch.zeros((mask.shape[0], 800, 1066))
        mask_p = torch.zeros((len(mask), 800, 1088))

        for i, m in enumerate(mask):
            # print(m)
            #error uint8 to float
            m = torch.tensor((m/1.0), dtype=torch.float).view(1,300,400)
            m = self.pad(resize(m).view(800,1066))
            # m = self.pad(resize(torch.tensor(m, dtype = torch.float32).view(1, 300, 400)).view(800,1066))
            m = torch.where(m > 0.5, 1, 0)
            mask_p[i] = m

        #box:
        bbox = torch.tensor(bbox, dtype=torch.float)
        box_p = torch.zeros_like((bbox))
        scaling_factors = torch.tensor([800/300, 1066/400, 800/300, 1066/400], dtype=torch.float)
        box_p = bbox * scaling_factors
        box_p[:, [1,3]] += 11 
        # check flag
        assert img_p.shape == (3, 800, 1088)
        # assert box_p.shape[0] == torch.squeeze(mask_p, 0).shape[0]

        return img_p, mask_p, box_p


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        images, labels, masks, bboxes = zip(*batch)
        return torch.stack(images, dim=0), labels, masks, bboxes

    def loader(self):
        # TODO: return a dataloader
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collect_fn)

def visualize_dataset_sample(image, label, mask, bbox):
    mask_converted = mask.bool()
    img = image[:, :, 11:-11]
    #font (str) – A filename containing a TrueType font. 
    # the loader may also search in other directories /Library/Fonts/, 
    # /System/Library/Fonts/ and ~/Library/Fonts/ on macOS.

    #load default font from PIL lib
    # try:
    #     font="/Library/Fonts/Ariale Unicode.ttf"
    # except OSError as error:
    #     font=None
    trans_inv = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.255]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    re_pad = torch.nn.ZeroPad2d((11, 11, 0, 0))
    img_reconstruct = re_pad(trans_inv(img))
    img_reconstruct = (img_reconstruct.clamp(0, 1) * 255).to(torch.uint8)
    img_reconst_visualize = img_reconstruct.clone()
    annotations = []  # List of annotations to draw on the image
    colors = {1: ('Vehicle', 'blue'), 2: ('People', 'green'), 3: ('Animal', 'red')}
    # #use this font to get a proper size of annotation
    # #if does not work; use default font
    # #you can specify your own font path according to your OS/ Directory
    # try:
    #     font_path="/Library/Fonts/"
    #     font = os.listdir(font_path)[0]
    #     img_op = draw_bounding_boxes(img_op, bbox, colors='red', width=2, font=font, font_size=35)
    # except OSError as error:
    #     img_op = draw_bounding_boxes(img_op, bbox, colors='red', width=2)
    if mask_converted.ndim == 2:
        # print("ouch")
        mask_converted = mask_converted[None, :, :]
    # for lab, mask in zip(label,mask_converted):
    #     annotation, color_name = colors.get(lab.item())
    #     annotations.append(annotation)
    #     color = torch.tensor(ImageColor.getrgb(color_name), dtype=torch.uint8)
    #     img_reconst_visualize[:, mask] = color[:, None]
    for mask, _, lab in zip(mask_converted, bbox, label):
        if lab.item() == 1:
            x = ImageColor.getrgb('blue')
            c = torch.tensor(x, dtype=torch.uint8)
            img_reconst_visualize[:, mask] = c[:, None]
            annotations.append("Vehicle")
        elif lab.item() == 2:
            x = ImageColor.getrgb('green')
            c = torch.tensor(x, dtype=torch.uint8)
            img_reconst_visualize[:, mask] = c[:, None]
            annotations.append("People")
        elif lab.item() == 3:
            x = ImageColor.getrgb('red')
            c = torch.tensor(x, dtype=torch.uint8)
            img_reconst_visualize[:, mask] = c[:, None]
            annotations.append("Animal")
    # for i, single_mask in enumerate(mask_converted):
    #     if single_mask.any():
    #         img_reconst_visualize[:, single_mask] = color[:, None]
    img_op = (img_reconstruct * (0.5) + img_reconst_visualize * 0.5).to(torch.uint8)
    #use this font to get a proper size of annotation
    #if does not work; use default font
    #you can specify your own font path according to your OS/ Directory
    try:
        font_path="/Library/Fonts/"
        font = os.listdir(font_path)[0]
        img_op = draw_bounding_boxes(img_op, bbox, labels=annotations, colors='red', width=2, font=font, font_size=35)
    except OSError as error:
        img_op = draw_bounding_boxes(img_op, bbox, labels=annotations, colors='red', width=2)
    img_disp = img_op
    return img_disp.permute(1, 2, 0)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = '/Users/ojasm/Desktop/UPenn/Fall 2023/CIS 680/project 3/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/Users/ojasm/Desktop/UPenn/Fall 2023/CIS 680/project 3/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '/Users/ojasm/Desktop/UPenn/Fall 2023/CIS 680/project 3/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '/Users/ojasm/Desktop/UPenn/Fall 2023/CIS 680/project 3/hw3_mycocodata_bboxes_comp_zlib.npy'
    
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image

    plot_idx = 0
    flag = True
    # fig, ax = plt.(figsize=(10, 20))

    # CUDA for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using the GPU!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using the mps!")
    else:
        device = torch.device("cpu")
        print("WARNING: Could not find GPU! Using CPU only")

    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        img = [img_img.to(device) for img_img in img]
        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        # plot the origin img
        # print(batch_size)
        for i in range(batch_size):
            ## TODO: plot images with annotations
            op_img = visualize_dataset_sample(img[i], label[i], mask[i], bbox[i])
            plt.imshow(op_img)
            plt.savefig("Test visualtrainset"+str(plot_idx)+".png")
            plt.show()
            plot_idx += 1

            if plot_idx == 10:
                flag = False
                break
        if flag == False:
            break
