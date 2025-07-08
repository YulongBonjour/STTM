import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from clip.model import build_model
from pathlib import Path
from random import randint
import json
# json.dump()
import PIL
from torch.utils.data import DataLoader
import torch
import json
from torch.utils.data import Dataset
import argparse
import clip.clip as CL
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)

assert sub in [1,2,5,7]

clip_st = torch.load('clip/ViT-L-14.pt', map_location='cpu')
clip=build_model(clip_st).eval()
clip.cuda()
   
train_caps = np.load('processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub))
test_caps = np.load('processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))

# num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

train_clip = {}#np.zeros((num_train,num_embed, num_features))
test_clip ={} #np.zeros((num_test,num_embed, num_features))
with torch.no_grad():
    for i,annots in enumerate(test_caps):
        cin = list(annots[annots!=''])
        cin=CL.tokenize(cin).cuda()
        print(i)
        c = clip.encode_text(cin)
        print(c.shape)
        test_clip[i] = c.to('cpu')#N 512
    if not os.path.exists('extracted_features/subj{:02d}'.format(sub)):
        os.makedirs('extracted_features/subj{:02d}'.format(sub), mode=0o777, exist_ok=True)
    torch.save(test_clip,'extracted_features/subj{:02d}/nsd_cliptext_L14_test.pt'.format(sub))
        
    for i,annots in enumerate(train_caps):
        cin = list(annots[annots!=''])
        cin = CL.tokenize(cin).cuda()
        print(i)
        c = clip.encode_text(cin)
        train_clip[i] = c.to('cpu')
    torch.save(train_clip,'extracted_features/subj{:02d}/nsd_cliptext_L14_train.pt'.format(sub))


