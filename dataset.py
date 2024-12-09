import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# DATASET_DIR = '/Users/taehwan/Documents/Dataset/CDT/MCI-multiple-drawings-main/' # Mac
DATASET_DIR = '/home/taehwan/Dataset/CDT/MCI-multiple-drawings-main/' # server

class Conv_Att_MCI_Dataset_v2(Dataset):
    def __init__(self, img_type, label_type):
        assert type(img_type) == list
        assert type(label_type) == str and label_type in ['hard', 'soft'], f"label_type is only valid with 'hard' or 'soft', but given {label_type}"
        self.img_type = img_type
        self.label_type = label_type
        self.load_dataset()
        self.img_normalization = {'mean': [0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        
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
            score_soft = 1 - sigmoid(moca_score - 24.5)
            
            dataset[int(patient_id)] = {os.path.basename(_path).split('.')[0]:_path for _path in image_paths}
            dataset[int(patient_id)]['score_hard'] = moca_score
            dataset[int(patient_id)]['label_hard'] = int(moca_score < 25) # MIC
            dataset[int(patient_id)]['score_soft'] = score_soft.astype('float32')
            dataset[int(patient_id)]['label_soft'] = int(score_soft >= 0.5) # MCI

        self.dataset_raw = pd.DataFrame(dataset).T        
        
    def split_trn_val_test(self):
        X = self.dataset_raw.loc[:, self.img_type]
        y = self.dataset_raw.loc[:, [f'score_{self.label_type}', f'label_{self.label_type}']]
        X_trn_val, X_test, y_trn_val, y_test = train_test_split(X, y, stratify=y.loc[:, f'label_{self.label_type}'], test_size=0.15)
        X_trn, X_val, y_trn, y_val = train_test_split(X_trn_val, y_trn_val, stratify=y_trn_val.loc[:, f'label_{self.label_type}'], test_size=0.15)
        dataset_trn = pd.concat([X_trn, y_trn], axis=1)
        dataset_val = pd.concat([X_val, y_val], axis=1)
        dataset_test = pd.concat([X_test, y_test], axis=1)
        
        dataset_trn = self.load_images(dataset_trn, augmentation=True)
        dataset_val = self.load_images(dataset_val)
        dataset_test = self.load_images(dataset_test)
        
        return self._split(dataset_trn, 'trn'), self._split(dataset_val, 'val'), self._split(dataset_test, 'test')
    
    def get_augmentation(self, x):
        transform = transforms.Compose([
            transforms.Pad([12,12,12,12], fill=255), 
            # left, top, right, bottom, fill=255 to match the background color of original images
            transforms.RandomCrop([256, 256]),
        ])
        return transform(x)
    
    def inverse_transform(self, img_torch):
        assert img_torch.shape[0] == 3, f"Image shape should be (channel, H, W), given {img_torch.shape}"
        
        img = img_torch.cpu().numpy()
        mean = np.array(self.img_normalization['mean'])[:,None,None]
        std = np.array(self.img_normalization['std'])[:,None,None]

        img_inv = img * std + mean
        img_inv = np.clip(img_inv, 0, 1)
        img_inv = img_inv.transpose(1,2,0)
        return img_inv
    
    def load_images(self, dataset_raw, augmentation=False):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.img_normalization['mean'], 
                                         std=self.img_normalization['std'])
                ])
        
        dataset = {}
        for patient_id, row in dataset_raw.iterrows():
            dataset[patient_id] = {'image':{}}
            if augmentation:
                dataset[f'{patient_id}_aug'] = {'image':{}}
                
            for _type in self.img_type:
                _img_path = row[_type]
                _score, _label = row[f'score_{self.label_type}'], row[f'label_{self.label_type}']
                
                _img = Image.open(_img_path)
                assert _img.size == (256,256)
                img = transform(_img)
                dataset[patient_id]['image'].update({_type:img})
                dataset[patient_id]['label'] = _label
                dataset[patient_id]['score'] = _score
                
                if augmentation:
                    _img_aug = _img.copy()
                    img_aug = transform(self.get_augmentation(_img_aug))
                    dataset[f'{patient_id}_aug']['image'].update({_type:img_aug})
                    dataset[f'{patient_id}_aug']['label'] = _label
                    dataset[f'{patient_id}_aug']['score'] = _score
        return dataset
    
    def _split(self, dataset, dataset_type):
        self.dataset = dataset
        self.dataset_type = dataset_type
        return deepcopy(self)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient_id = list(self.dataset.keys())[idx]
        images = []
        for _type in self.img_type:
            images.append(self.dataset[patient_id]['image'][_type])
        label = self.dataset[patient_id]['label']
        score = self.dataset[patient_id]['score']
        info = {}#{'patient_id':patient_id}#, 'img_type':[_type], }
            
        return images, label, score, info

class Conv_Att_MCI_Dataset(Dataset):
    def __init__(self, img_type, is_soft_label=False):
        self.img_type = img_type
        self.is_soft_label = is_soft_label
        self.dataset_raw = self.load_dataset()
        self.load_images()

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

    def load_images(self):
        transform_aug = transforms.Compose([
            transforms.Pad([12,12,12,12], fill=255), # left, top, right, bottom, fill=255 to match the background color of original images
            transforms.RandomCrop([256,256]),
        ])

        transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        for patient_id in self.dataset_raw:
            for _type in self.img_type:
                _img_path = self.dataset_raw[patient_id][_type]
                _img = Image.open(_img_path)
                assert _img.size == (256,256)
                _img_aug = _img.copy()

                img = transform(_img)
                img_aug = transform(transform_aug(_img_aug))
                self.dataset_raw[patient_id][_type] = {'original': img, 'augmented':img_aug}
    
    def make_dataset(self):
        images = {_type:[] for _type in self.img_type}
        images_aug = {_type:[] for _type in self.img_type}
        labels, scores, patient_ids = [],[],[]
        for patient_id in self.dataset_raw:
            for _type in self.img_type:
                _img = self.dataset_raw[patient_id][_type]['original']
                _img_aug = self.dataset_raw[patient_id][_type]['augmented']
                images[_type].append(_img)
                images_aug[_type].append(_img_aug)
            
            _score = self.dataset_raw[patient_id]['score']
            scores.append(_score)
            labels.append(_score < 25)
            patient_ids.append(patient_id)
        dataset = {'scores':torch.Tensor(scores),
                   'labels':torch.Tensor(labels),
                   'patient_ids':torch.LongTensor(patient_ids)
                  }
        
        for _type in self.img_type:
            dataset.update({
                _type:torch.stack(images[_type]), 
                _type+'_aug':torch.stack(images_aug[_type]),
            })
        
        n_tot = len(dataset['labels'])
        assert n_tot == len(dataset['scores']), "%d %s"%(n_tot, dataset['scores'].shape)
        for _type in self.img_type:
            assert n_tot == len(dataset[_type]) == len(dataset[_type+'_aug']), "%d %s"%(n_tot, dataset[_type].shape, dataset[_type+'_aug'].shape)
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
            'labels':torch.cat([_dataset['labels'], _dataset['labels']]),
            'scores':torch.cat([soft_label, soft_label]),
            'patient_ids':torch.cat([_dataset['patient_ids'], _dataset['patient_ids']])
        }
        for _type in self.img_type:
            self.dataset.update({
                _type:torch.cat([_dataset[_type], _dataset[_type+'_aug']]),
            })
        return deepcopy(self)
    
    def __len__(self):
        return len(self.dataset['labels'])
    
    def get_soft_label(self, scores):
        return 1 - torch.sigmoid(scores - 24.5)

    def __getitem__(self, idx):
        images = []
        for _type in self.img_type:
            images.append(self.dataset[_type][idx])
        scores = self.dataset['scores'][idx]
        labels = self.dataset['labels'][idx]
        patient_ids = self.dataset['patient_ids'][idx]
        info = {'labels':labels, 'scores':scores, 'patient_ids':patient_ids}
        if self.is_soft_label:
            return images, scores, info
        else:
            return images, labels, info