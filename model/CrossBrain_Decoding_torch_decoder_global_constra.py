import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from typing import Optional
import torch
from einops import rearrange
from torch import nn
from functools import partial
import model.utils as utils
# for prior
from model.dalle2_pytorch import DiffusionPrior
from model.dalle2_pytorch import l2norm, default, exists
from tqdm.auto import tqdm
import random
import json
from model.attention import MultiHeadAttention
# from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
from torch.nn import TransformerDecoderLayer,TransformerDecoder
# vd prior
from einops import rearrange
from model.dalle2_pytorch import RotaryEmbedding, SinusoidalPosEmb, MLP, Rearrange, repeat, \
    rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward

def decoder(d_model=768, nhead=12, depth=8,dim_feedforward=2048, dropout=0.1, activation="gelu"):
    decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward,
                                            dropout=dropout, activation=activation)
    transformer_decoder =TransformerDecoder(decoder_layer, num_layers=depth)
    return transformer_decoder


def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def random_select(tensor_list, kept_size):
    '''
    :param x: B D x
    :return:
    '''

    b = tensor_list[0].shape[0]
    num_keep = kept_size
    shuffle_indices = torch.rand(b).argsort()
    keep_ind = shuffle_indices[:num_keep]
    out = []
    for x in tensor_list:
        out.append(x[keep_ind, :])
    return out

class SelfAttention_with_MLP(nn.Module):
    """modified Self-attention module."""
    def __init__(
        self,
        *,
        hidden_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        output_dim=768,
        num_heads: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """Constructor.

        Args:
            hidden_dim: Dimension of input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(
            kv_dim=hidden_dim,
            q_dim=hidden_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim,output_dim*2),
                                      nn.LayerNorm(output_dim*2),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(output_dim*2,output_dim))#FeedForward(hidden_dim, widening_factor, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: Input tensor of shape (B, M, C).
            attention_mask: Input mask tensor of shape (B, M, M).
                Mask values selected in [0, 1]. Defaults to None.
        """
        x_norm = self.layer_norm(x)
        attention = self.attention(
            inputs_kv=x_norm,
            inputs_q=x_norm,
            attention_mask=attention_mask
        )
        attention = self.dropout(attention)
        x = x + attention
        return self.out_proj(self.qkv_layer_norm(x))


class CrossBrainNetwork(nn.Module):
    def __init__(self,voxel_num_dict= {1: 15724, 2: 14278, 5: 13039, 7: 12682}, clip_size=768, token_dim=128,h=4096, n_blocks=3, norm_type='ln', act_first=False,
                     use_token_attention=True,input_n_blocks=2):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm,
                                                                                              normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        self.input_proj={}
        self.input_res_block={}
        for k, v in voxel_num_dict.items():
            self.input_proj['head{}'.format(k)] = nn.Sequential(
                nn.Linear(v,h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.5),
                )
            self.input_res_block['head{}'.format(k)]=nn.ModuleList([nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
                ) for _ in range(input_n_blocks)
            ])
        self.input_proj=nn.ModuleDict(self.input_proj)
        self.input_res_block=nn.ModuleDict(self.input_res_block)
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])

        self.hidd2token=nn.Sequential(nn.LayerNorm(h),
                                      nn.GELU(),
                                      nn.Linear(h,257* token_dim))
        if use_token_attention:
            self.token_attention=SelfAttention_with_MLP(hidden_dim= token_dim,
                                           qk_out_dim=768,
                                           v_out_dim=768,
                                           num_heads=8,
                                           output_dim=clip_size,
                                           dropout=0.15,
                                           attention_dropout=0.1
                                           )
        else:
            self.token_attention=nn.Sequential(nn.Linear(token_dim,clip_size*2),
                                      nn.LayerNorm(clip_size*2),
                                      nn.GELU(),
                                      nn.Dropout(0.15),
                                      nn.Linear(clip_size*2,clip_size))
        self.projector = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(),
            nn.Linear(clip_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            #nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
           # nn.Dropout(0.2),
            nn.Linear(2048, clip_size)
        )
        self.n_blocks = n_blocks
        self.input_n_blocks=input_n_blocks
        self.clip_size=clip_size
        self.token_dim=token_dim
        self.use_token_attention=use_token_attention
    def get_subject_proj(self,x,sub_id):
        x = self.input_proj['head{}'.format(sub_id)](x)
        residual = x
        for i in range(self.input_n_blocks):
             x=self.input_res_block['head{}'.format(sub_id)][i](x)
             x += residual
             residual = x
        return x
    def get_adapter_out(self,x,sub_id):
        x = self.input_proj['head{}'.format(sub_id)](x)
        residual = x
        for i in range(self.input_n_blocks):
            x=self.input_res_block['head{}'.format(sub_id)][i](x)
            x += residual
            residual = x
        return x
    def get_backbone_out(self,x,sub_id):
        x=self.get_subject_proj(x, sub_id=sub_id)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        return x
    def run_shared_module(self,x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = self.hidd2token(x)
        x = x.reshape(len(x), -1, self.token_dim)
        x = self.token_attention(x)
        return x, self.projector(x)
    def forward(self, x,sub_id):
        x=self.get_subject_proj(x,sub_id=sub_id)
        return self.run_shared_module(x)

class BrainDiffusionPrior(DiffusionPrior):
    """
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """

    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.,
                 generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, text_cond=text_cond,
                                                                          self_cond=self_cond,
                                                                          clip_denoised=clip_denoised,
                                                                          cond_scale=cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            # noise = torch.randn_like(x)
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float().to(x)).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond,
                                                 cond_scale=cond_scale,
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None,is_train=True):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            is_train=is_train,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
            self,
            text=None,
            image=None,
            voxel=None,
            text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
            image_embed=None,
            text_encodings=None,  # as well as CLIP text encodings
            # global_image_emb=None,
            global_text_emb=None,
            mix_up=True, #return output of subject-specific module
            subject_id=None,
            use_cross_sub_loss=True,
            inference_mode=True,
            epoch_temp=0.005,
            only_loss_prior=False,
            is_train=False,
            *args,
            **kwargs
    ):
        if inference_mode:
            assert exists(text) ^ exists(text_embed) ^ exists(
                voxel), 'either text, text embedding, or voxel must be supplied'
            assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
            assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(
                text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

            if exists(voxel):
                assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
                assert not exists(text_embed), 'cannot pass in both text and voxels'
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse

            if exists(image):
                image_embed, _ = self.clip.embed_image(image)

            # calculate text conditionings, based on what is passed in

            if exists(text):
                text_embed, text_encodings = self.clip.embed_text(text)

            text_cond = dict(text_embed=text_embed)

            if self.condition_on_text_encodings:
                assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
                text_cond = {**text_cond, 'text_encodings': text_encodings}

            # timestep conditioning from ddpm

            batch, device = image_embed.shape[0], image_embed.device
            times = self.noise_scheduler.sample_random_times(batch)

            # PS: I dont think we need this? also if uncommented this does in-place global variable change
            # scale image embed (Katherine)
            # image_embed *= self.image_embed_scale

            # calculate forward loss

            loss, pred = self.p_losses(image_embed * self.image_embed_scale, times, text_cond=text_cond,is_train=is_train, *args,
                                       **kwargs)

            # undo the scaling so we can directly use it for real mse loss and reconstruction
            return loss, pred
        else:#for training
            assert exists(voxel)
            assert exists(subject_id)
            if only_loss_prior:
                clip_voxels, clip_voxels_proj = self.voxel2clip(voxel, subject_id)
                del clip_voxels_proj
                torch.cuda.empty_cache()
                loss_prior = self.loss_prior(clip_voxels, image_embed,is_train=is_train)
                return loss_prior
            if mix_up:
                loss_nce,loss_nce_global_img,loss_nce_global_txt,clip_voxels,_=self.BiMixCo_fmri(voxel,image_embed,global_text_emb,subject_id)
                loss_prior = self.loss_prior(clip_voxels, image_embed,is_train=is_train)
                return loss_nce,loss_nce_global_img,loss_nce_global_txt,loss_prior

            else:#softclip+loss_prior
                clip_voxels, clip_voxels_proj = self.voxel2clip(voxel, subject_id)
                #loss_prior=self.loss_prior(clip_voxels,image_embed)
                #del clip_voxels
                #torch.cuda.empty_cache()
                global_emb_pred=clip_voxels_proj[:,0]#B 768
                global_emb_tgt=image_embed[:,0]
                global_emb_pred=global_emb_pred/global_emb_pred.norm(dim=-1,keepdim=True)
                global_emb_tgt=global_emb_tgt/global_emb_tgt.norm(dim=-1,keepdim=True)
                global_text_emb= global_text_emb/ global_text_emb.norm(dim=-1,keepdim=True)
                #print(global_emb_pred.shape,global_emb_tgt.shape,global_text_emb.shape)

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(image_embed.flatten(1), dim=-1)
                del clip_voxels_proj#, image_embed
                torch.cuda.empty_cache()
                #torch.cuda.empty_cache()
                clip_voxels_norm, clip_target_norm= utils.all_gather_batch_with_grad([clip_voxels_norm, clip_target_norm])
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)
                del clip_voxels_norm,clip_target_norm
                torch.cuda.empty_cache()

                global_emb_pred, global_emb_tgt,global_text_emb = utils.all_gather_batch_with_grad(
                    [global_emb_pred, global_emb_tgt,global_text_emb])
                #print(global_emb_pred.shape,global_emb_tgt.shape,global_text_emb.shape)
                loss_nce_global_img = utils.soft_clip_loss(
                    global_emb_pred, global_emb_tgt,
                    temp=epoch_temp)
                loss_nce_global_txt = utils.soft_clip_loss(
                    global_emb_pred, global_text_emb,
                    temp=epoch_temp)

                del global_emb_pred, global_emb_tgt,global_text_emb
                torch.cuda.empty_cache()
                loss_prior=self.loss_prior(clip_voxels,image_embed,is_train=is_train)
                del clip_voxels

                return loss_prior, loss_nce, loss_nce_global_img,loss_nce_global_txt

    def BiMixCo_fmri(self,fmri,clip_hiddens,global_text_emb,subject_id=None):
        voxel, perm, betas, select = utils.mixco(fmri)
        clip_voxels, clip_voxels_proj = self.voxel2clip(voxel, subject_id)
        global_emb_pred = clip_voxels_proj[:, 0]  # B 768
        global_emb_tgt = clip_hiddens[:, 0]
        global_emb_pred = global_emb_pred / global_emb_pred.norm(dim=-1, keepdim=True)
        global_emb_tgt = global_emb_tgt / global_emb_tgt.norm(dim=-1, keepdim=True)
        global_text_emb = global_text_emb / global_text_emb.norm(dim=-1, keepdim=True)
       # print(global_emb_pred.shape, global_emb_tgt.shape, global_text_emb.shape)

        clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(clip_hiddens.flatten(1), dim=-1)
        loss_nce = utils.mixco_nce(
            clip_voxels_norm,
            clip_target_norm,
            temp=.006,
            perm=perm, betas=betas, select=select)
        loss_nce_global_img=utils.mixco_nce(
            global_emb_pred,
            global_emb_tgt,
            temp=.006,
            perm=perm, betas=betas, select=select)
        loss_nce_global_txt=utils.mixco_nce(
            global_emb_pred,
            global_text_emb,
            temp=.006,
            perm=perm, betas=betas, select=select)
        return loss_nce,loss_nce_global_img,loss_nce_global_txt,clip_voxels,clip_voxels_proj
    def BiMixCo_z(self,z,clip_hiddens):
        z, perm, betas, select = utils.mixco(z)
        clip_voxels, clip_voxels_proj = self.voxel2clip.run_shared_module(z)
        clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(clip_hiddens.flatten(1), dim=-1)
        loss_nce = utils.mixco_nce(
            clip_voxels_norm,
            clip_target_norm,
            temp=.006,
            perm=perm, betas=betas, select=select)
        return loss_nce
    def loss_prior(self,clip_voxels,img_emb,is_train=True):
        loss_prior, aligned_clip_voxels = self.forward_ori(text_embed=clip_voxels,
                                                                 image_embed=img_emb,is_train=is_train)
        return loss_prior
    def forward_ori(
            self,
            text=None,
            image=None,
            voxel=None,
            text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
            image_embed=None,
            text_encodings=None,  # as well as CLIP text encodings
            is_train=False,
            *args,
            **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(
            voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(
            text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
            text_embed = clip_voxels_mse

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed * self.image_embed_scale, times, text_cond=text_cond, is_train=is_train,*args, **kwargs)

        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

class VersatileDiffusionPriorNetwork(nn.Module):
    def __init__(
            self,
            dim,
            num_timesteps=None,
            num_time_embeds=1,
            # num_image_embeds = 1,
            # num_brain_embeds = 1,
            num_tokens=257,
            causal=True,
            learned_query_mode='none',
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
            self.learned_query_brain = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        # self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)
        self.causal_transformer = decoder(d_model=768, nhead=12, depth=6,dim_feedforward=768*4, dropout=0., activation="gelu")
        # self.register_buffer('mask',generate_square_subsequent_mask(num_tokens+1))
        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False
    def random_mask(self,x, mask_ratio):
        '''
        :param x: B N D
        :param mask_ratio:
        :return:
        '''
        b,n=x.shape[0],x.shape[1]
        num_keep = int((1-mask_ratio) * n)
        shuffle_indices = torch.rand(b, n).argsort()
        #shuffle_indices[:,0]=-10000
        keep_ind = shuffle_indices[:, :num_keep]
        keep_ind,_=keep_ind.sort(dim=-1)
        batch_ind = torch.arange(b).unsqueeze(-1)
        x=x[batch_ind, keep_ind]
        return x
    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob=1., image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            brain_embed=None,
            text_embed=None,
            brain_cond_drop_prob=0.,
            text_cond_drop_prob=None,
            image_cond_drop_prob=0.,
            is_train=False
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        image_embed = image_embed.view(len(image_embed), -1, 768)

        brain_embed = brain_embed.view(len(brain_embed), -1, 768)

        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds

        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)
        pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
        pos_embs_brain = repeat(self.learned_query_brain, 'n d -> b n d', b=batch)
        image_embed = image_embed + pos_embs
        brain_embed = brain_embed + pos_embs_brain
        # print('pso_emb shape',pos_embs.shape)
        if is_train:
            up_=int(257*0.65)
            low_=int(257*0.6)
            mask_ratio=random.randint(low_,up_)
            mask_ratio=mask_ratio/257
            brain_embed=self.random_mask(brain_embed,mask_ratio=mask_ratio)#b n d
        #print(brain_embed.shape)
        tokens = torch.cat((
            # brain_embed,  # 257*kept_ratio
            time_embed,  # 1
            image_embed,  # 257
            # learned_queries  # 257
        ), dim=-2)
        brain_embed=rearrange(brain_embed,'b n d -> n b d')
        tokens=rearrange(tokens,'b n d -> n b d')

        # attend
        tokens = self.causal_transformer(tgt=tokens,memory=brain_embed,tgt_mask=None)
        tokens = tokens[-self.num_tokens:]
        tokens=rearrange(tokens,'n b d -> b n d')
        return tokens


class FlaggedCausalTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            norm_in=False,
            norm_out=True,
            attn_dropout=0.,
            ff_dropout=0.,
            final_proj=True,
            normformer=False,
            rotary_emb=True,
            causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=causal, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                          rotary_emb=rotary_emb),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))

        self.norm = LayerNorm(dim,
                              stable=True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        x=x.to(self.project_out.weight)
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


