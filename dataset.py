import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Dataset

DATASET_DIR = '/Users/taehwan/Documents/Dataset/Research/MCI-multiple-drawings-main/'

class Conv_Att_MCI_Dataset(Dataset):
    def __init__(self, img_type='all'):
        self.dataset_raw = self.load_dataset()
        self.load_images(img_type)

    def load_dataset(self):
        image_folder = os.path.join(DATASET_DIR, 'images')
        info_path = os.path.join(DATASET_DIR, 'label.csv')
        info = pd.read_csv(info_path)

        dataset = {}
        for patient_id in os.listdir(image_folder):
            if not patient_id.isdigit():
                continue
            _patient_folder = os.path.join(image_folder, patient_id)
            image_paths = [os.path.join(_patient_folder, img_file) for img_file in os.listdir(_patient_folder) if os.path.splitext(img_file)[-1] == '.png']
            
            mask_patient = info['patient_id'] == int(patient_id)
            assert sum(mask_patient) == 1
            moca_score = info[mask_patient]['MoCA_score'].iloc[0]
            
            dataset[int(patient_id)] = {os.path.basename(_path).split('.')[0]:_path for _path in image_paths}
            dataset[int(patient_id)]['score'] = moca_score
        return dataset

    def load_images(self, img_type):
        img_type = ['clock', 'trail', 'copy'] if img_type == 'all' else [img_type]

        transform_aug = transforms.Compose([
            transforms.Pad([12,12,12,12]), # left, top, right, bottom
            transforms.RandomCrop([256,256])
        ])

        transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        for patient_id in self.dataset_raw:
            for _type in img_type:
                _img_path = self.dataset_raw[patient_id][_type]
                _img = Image.open(_img_path)
                assert _img.size == (256,256)
                _img_aug = _img.copy()

                img = transform(_img)
                img_aug = transform(transform_aug(_img_aug))

                self.dataset_raw[patient_id]['images'] = {'original': img, 'augmented':img_aug}
    
    def make_dataset(self):
        images, images_aug, labels, scores = [],[],[],[]
        for patient_id in self.dataset_raw:
            _img = self.dataset_raw[patient_id]['images']['original']
            _img_aug = self.dataset_raw[patient_id]['images']['augmented']
            _score = self.dataset_raw[patient_id]['score']
            images.append(_img)
            images_aug.append(_img_aug)
            scores.append(_score)
            labels.append(_score < 25)
        dataset = {'images':torch.stack(images), 
                   'images_aug':torch.stack(images_aug),
                   'scores':torch.Tensor(scores),
                   'labels':torch.Tensor(labels)}
        n_tot = len(dataset['images'])
        assert n_tot == len(dataset['images_aug']) == len(dataset['scores']) == len(dataset['labels']), "%d %s %s %s"%(n_tot, dataset['images_aug'].shape, dataset['scores'].shape, dataset['labels'].shape)
        return dataset, n_tot
    
    def random_select_idxs(self, idxs, n):
        idxs_idxs_selected = np.random.choice(len(idxs), n, replace=False)
        idxs_selected = idxs[idxs_idxs_selected]
        idxs = np.delete(idxs, idxs_idxs_selected)
        return idxs, idxs_selected
    
    def split_trn_val_test(self, prob=[0.7,0.15,0.15]): #prob=[trn, val, test]
        _dataset, n_tot = self.make_dataset()
        
        # # for test
        # _dataset['images'] = torch.arange(len(_dataset['images'])) 
        # _dataset['images_aug'] = torch.arange(len(_dataset['images_aug'])) + len(_dataset['images'])

        n_labels = len(_dataset['labels'].unique())

        idxs_per_class = {}
        for _class in range(n_labels):
            _idxs = torch.where(_dataset['labels']==_class)[0]
            idxs_per_class[_class] = _idxs

        dataset = {'trn':{k:[]for k in _dataset.keys()}, 
                'test':{k:[]for k in _dataset.keys()}, 
                'val':{k:[]for k in _dataset.keys()}}

        for _class, _idxs in idxs_per_class.items():
            n_tot = len(_idxs)
            n_trn, n_test = round(n_tot*prob[0]), round(n_tot*prob[-1])
            n_val = n_tot - (n_trn+n_test)
            _idxs, idxs_trn = self.random_select_idxs(_idxs, n_trn)
            _idxs, idxs_val = self.random_select_idxs(_idxs, n_val)
            _idxs, idxs_test = self.random_select_idxs(_idxs, n_test)
            assert len(_idxs) == 0
            
            for k,v in _dataset.items():
                dataset['trn'][k].append(v[idxs_trn])
                dataset['val'][k].append(v[idxs_val])
                dataset['test'][k].append(v[idxs_test])

        for _type in dataset:
            for k in dataset[_type]:
                dataset[_type][k] = torch.cat(dataset[_type][k])

        return self._split(dataset['trn']), self._split(dataset['val']), self._split(dataset['test'])

    def _split(self, _dataset):
        soft_label = self.get_soft_label(_dataset['scores'])
        self.dataset = {
            'images':torch.cat([_dataset['images'], _dataset['images_aug']]),
            'labels':torch.cat([_dataset['labels'], _dataset['labels']]),
            'scores':torch.cat([soft_label, soft_label])
        }
        return deepcopy(self)
    
    def __len__(self):
        return len(self.dataset['labels'])
    
    def get_soft_label(self, scores):
        return 1 - torch.sigmoid(scores - 24.5)

    def __getitem__(self, idx):
        images = self.dataset['images'][idx]
        scores = self.dataset['scores'][idx]
        labels = self.dataset['labels'][idx]
        return images, scores, {'labels':labels}