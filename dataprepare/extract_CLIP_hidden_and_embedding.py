import os
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import clip

from PIL import Image

# import torchvision.transforms as T

import sys
import numpy as np

# from clip.model import build_model

from torch.utils.data import DataLoader
import torch

from torch.utils.data import Dataset
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1, 2, 5, 7]


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=True, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)

        if clip_variant == "ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("./clip-vit-large-patch14",use_auth_token=None).eval()
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            # from transformers import CLIPVisionModelWithProjection
            # sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.cuda()
            for param in image_encoder.parameters():
                param.requires_grad = False  # dont need to calculate gradients
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")

        # clip_model, preprocess = clip.load(clip_variant, device=device)
        # clip_model.eval()  # dont want to train model
        # for param in clip_model.parameters():
        #     param.requires_grad = False  # dont need to calculate gradients
        #
        # self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448, 448)
        else:
            self.clip_size = (224, 224)

        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device = device

        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings


    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        # if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
        clip_emb = self.preprocess((image).to(self.device))
        out = self.image_encoder(clip_emb)
        clip_emb=out.image_embeds
        clip_hidden = self.versatile_normalize_embeddings(out)
        if self.norm_embs:
                # normalize all tokens by cls token's norm
                clip_hidden = clip_hidden / torch.norm(clip_hidden[:, 0], dim=-1).reshape(-1, 1, 1)
                #print(torch.norm(clip_hidden[:, 0], dim=-1).reshape(-1, 1, 1))
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb,clip_hidden

    def embed_text(self, text_samples):
        clip_text = clip.tokenize(text_samples).to(self.device)
        clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text

    def embed_curated_annotations(self, annots):
        for i, b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5, (1, 1))[0][0]
                t = b[0, rand]
            if i == 0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt, t))
        txt = txt.flatten()
        return self.embed_text(txt)
clip_extractor = Clipper( clip_variant="ViT-L/14",norm_embs=True, device=torch.device('cuda:0'), hidden_state=True)
#clip_extractor=clip_extractor.cuda()



class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)
        self.image_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        # img = T.functional.resize(img, (224, 224))
        img = self.image_transform(img)
        #print(img.shape)
        return img,idx

    def __len__(self):
        return len(self.im)


batch_size = 1
image_path = 'processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub, sub)
train_images = batch_generator_external_images(data_path=image_path)

image_path = 'processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub, sub)
test_images = batch_generator_external_images(data_path=image_path)

trainloader = DataLoader(train_images, batch_size, shuffle=False)
testloader = DataLoader(test_images, batch_size, shuffle=False)

# num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)

train_clip = {}  # np.zeros((num_train,num_embed,num_features))
test_clip = {}  # np.zeros((num_test,num_embed,num_features))

with torch.no_grad():
    '''for cin,idx in testloader:
        idx=idx[0].item()
        print(idx)
        # ctemp = cin*2 - 1
        emb,hidden = clip_extractor.embed_image(cin.cuda())
        print('emb:', emb.shape,"hidden:",hidden.shape)
        test_clip[idx]={}
        test_clip[idx]["emb"] = emb.cpu().half()
        #test_clip[idx]["hidden"]=hidden.cpu().half()
    torch.save(test_clip, 'extracted_features/subj{:02d}/nsd_clipvision_L14_emb_test.pt'.format(sub))
    del test_clip'''
    for cin, idx in trainloader:
        idx = idx[0].item()
        print(idx)
        # ctemp = cin*2 - 1
        emb,hidden = clip_extractor.embed_image(cin.cuda())
        train_clip[idx] = {}
        print('emb:', emb.shape,"hidden:",hidden.shape)
        train_clip[idx]["emb"] = emb.cpu().half()
        train_clip[idx]["hidden"] = hidden.cpu().half()
    torch.save(train_clip, 'extracted_features/subj{:02d}/nsd_clipvision_L14_emb_hidden_train.pt'.format(sub))







