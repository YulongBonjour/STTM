import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import clip
class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=True, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)

        if clip_variant == "ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("/public/share/huchen1/yulong/fmri2language/fmriCLIP/nsd/clip-vit-large-patch14").eval()
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            # from transformers import CLIPVisionModelWithProjection
            # sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.to(device)
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
                # print(torch.norm(clip_hidden[:, 0], dim=-1).reshape(-1, 1, 1))
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
