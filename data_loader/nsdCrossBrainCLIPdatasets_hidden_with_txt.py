import torch
from torch.utils.data import Dataset
# from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from typing import Tuple
import sys
from torchvision import transforms as T
import PIL
import os
from random import randint
import numpy as np
class nsdCLIPDataset(Dataset):
    def __init__(self, sub, split='train',data_folder=''):
        # if split=="train":
        train_path = os.path.join(data_folder,'processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub))
        train_fmri = np.load(train_path)
        test_path =  os.path.join(data_folder,'processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub))
        test_fmri = np.load(test_path)

        ## Preprocessing fMRI

        train_fmri = train_fmri / 300
        test_fmri = test_fmri / 300

        norm_mean_train = np.mean(train_fmri, axis=0)
        norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
        self.train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
        self.test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

        print(np.mean(self.train_fmri), np.std(self.train_fmri))
        print(np.mean(self.test_fmri), np.std(self.test_fmri))

        print(np.max(self.train_fmri), np.min(self.train_fmri))
        print(np.max(self.test_fmri), np.min(self.test_fmri))

        self.num_voxels, self.num_train, self.num_test = self.train_fmri.shape[1], len(self.train_fmri), len(self.test_fmri)
        if split=='train':
            self.train_clipvision = torch.load(
            os.path.join(data_folder,'extracted_features/subj{:02d}/nsd_clipvision_L14_emb_hidden_train.pt'.format(sub)))
            self.train_cliptext = torch.load(
                os.path.join(data_folder, 'extracted_features/subj{:02d}/nsd_cliptext_L14_train.pt'.format(sub)))
            self.test_fmri=None
            # self.train_filtered_fmri=torch.load( os.path.join(data_folder,'extracted_features/subj{:02d}/nsd_encoder_filtered_fMRI_betas.pt'.format(sub)))
        else:
            self.test_clipvision = torch.load(
            os.path.join(data_folder,'extracted_features/subj{:02d}/nsd_clipvision_L14_emb_hidden_test.pt'.format(sub)))
            self.test_cliptext = torch.load(
            os.path.join(data_folder,'extracted_features/subj{:02d}/nsd_cliptext_L14_test.pt'.format(sub)))
            self.train_fmri=None
        self.split=split


    def __len__(self) -> int:
        if self.split=='train':
            return len(self.train_clipvision)
        else:
            return len(self.test_clipvision)
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        if self.split=="train":
            txt_embs=self.train_cliptext[item]
            n=len(txt_embs)
            txt_emb = txt_embs[randint(0,n-1)].to(torch.float16)#(768,)
            # img_emb=self.train_clipvision[item]['emb'][0]#(768)
            hidden=self.train_clipvision[item]["hidden"][0]#(257,768)
            fmri = torch.from_numpy(self.train_fmri[item]).to(torch.float16)
            fmri=fmri+0.01*torch.randn(fmri.shape).to(fmri)
            return fmri, hidden,txt_emb#img_emb, txt_emb,hidden
        else:
            txt_embs = self.test_cliptext[item]
            n = len(txt_embs)
            txt_emb = txt_embs[randint(0, n - 1)].to(torch.float16)
            # img_emb = self.test_clipvision[item]['emb'][0]
            hidden = self.test_clipvision[item]["hidden"][0]
            fmri = torch.from_numpy(self.test_fmri[item]).to(torch.float16)
            # fmri+=0.01*torch.randn(fmri.shape)
            return item,fmri, hidden,txt_emb#img_emb, txt_emb,hidden,item
