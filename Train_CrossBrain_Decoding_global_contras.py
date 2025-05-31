import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from data_loader.nsdCrossBrainCLIPdatasets_hidden_with_txt import nsdCLIPDataset
# from data_loader.vision_self_supervision_dataset import ImageNet_Val_CLIPDataset
from model.CrossBrain_Decoding_torch_decoder_global_constra import CrossBrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork
# import utils
import torch
import torch.nn.functional as F
import model.utils as utils
import torch.nn as nn
from shutil import copyfile
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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


def soft_clip_loss(preds, targs, temp=0.07):
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2) / 2
    return loss


def contrast_loss(emb, tgt_emb=None, temp=0.05):
    batch_size = emb.shape[0]
    logits = emb @ tgt_emb.t() / temp
    labels = torch.arange(batch_size).cuda()  # +batch_size*rank
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    clip_loss = (loss2 + loss1) / 2
    return clip_loss


def exists(val):
    return val is not None


def cosine_anneal(start, end, steps):
    return end + (start - end) / 2 * (1 + torch.cos(torch.pi * torch.arange(steps) / (steps - 1)))


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def save_model(epoch, model, opt, dir, lr_scheduler, fname=None):
    save_obj = {
        'epoch': epoch,
        'weights': model.module.state_dict(),
        'opt_state': opt.state_dict(),
        'lr_scheduler_state': lr_scheduler.state_dict()

    }
    if fname is None:
        path = os.path.join(dir, "CrossBrain_Decoding" + str(epoch) + '.pt')
    else:
        path = os.path.join(dir, fname)
    torch.save(save_obj, path)


def num_training_steps(args, dataset_len):
    """Total training steps inferred from datamodule and devices."""
    dataset_size = dataset_len
    num_devices = args.world_size
    effective_batch_size = args.batch_size * num_devices
    return dataset_size // effective_batch_size


def configure_optimizers(args, diffusion_prior, max_lr=1e-4, train_num_each_sub=8859):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
    total_steps = int(args.num_epochs * (train_num_each_sub // args.batch_size + 1))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2 / args.num_epochs
    )
    return optimizer, lr_scheduler


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main(args):
    #############################进程组#################
    rank = args.rank  # int(os.environ['RANK'])  #获取当前进程号
    # world_size=int(os.environ['WORLD_SIZE'])
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )  # 初始化
    assert dist.is_initialized()
    synchronize()
    print('进程组初始化完成')
    set_seed(42+rank)
    torch.cuda.set_device(args.local_rank)
    start_epoch = 0
    ################################# Resume ################
    RESUME = exists(args.resume)
    if RESUME:
        assert os.path.exists(args.resume), 'model file does not exist'
        loaded_obj = torch.load(args.resume, map_location='cpu')
        start_epoch, weights = loaded_obj['epoch'], loaded_obj['weights']
        opt_state = loaded_obj.get('opt_state')
        scheduler_state = loaded_obj.get("lr_scheduler_state")
        start_epoch+=1
    ############################dataset#####################
    if args.train_single_subject == 0:
        rank2sub_id = {0: 1, 1: 2, 2: 5, 3: 7}
        subject_id = rank2sub_id[rank%4]
        print('rank%4:',rank%4)
        voxel_num_dict = {1: 15724, 2: 14278, 5: 13039, 7: 12682}
    else:
        subject_id = args.subject_id
        voxel_num_dict_ = {1: 15724, 2: 14278, 5: 13039, 7: 12682}
        voxel_num_dict = {subject_id: voxel_num_dict_[subject_id]}

    train_dataset = nsdCLIPDataset(
        sub=subject_id,
        split='train',
        data_folder=args.fmri_data_folder
    )
    # SSP_data=ImageNet_Val_CLIPDataset(world_size=args.world_size,rank=rank,data_folder=args.fmri_data_folder)
    print(f"word size:{args.world_size}--rank:{rank}")
    print('loading dataset is complete!')
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=args.world_size,
    #     rank=args.rank
    # )
    print('dataloader 初始化')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,  # args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=None,  # train_sampler,
                                               drop_last=False)
    print(f'train with {len(voxel_num_dict)} subjects,subject ids:{list(voxel_num_dict.keys())}')
    voxel2clip = CrossBrainNetwork(voxel_num_dict=voxel_num_dict, clip_size=768, token_dim=128, h=4096,
                                   n_blocks=4, norm_type='ln', act_first=False,
                                   use_token_attention=args.use_token_attention == 1, input_n_blocks=1)
    # voxel2clip=voxel2clip.half()
    out_dim = clip_size = 768
    depth = 6
    dim_head = 64
    heads = clip_size // 64
    guidance_scale = 3.5
    timesteps = 100
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=257,
        learned_query_mode="pos_emb"
    )
    print("prior_network loaded")
    # prior_network =prior_network.half()
    # custom version that can fix seeds
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    )
    # diffusion_prior=diffusion_prior.half()
    print("params of diffusion prior:")
    if rank == 0:
        utils.count_params(diffusion_prior)
    first_stage_pct = 0.35
    soft_loss_temps = utils.cosine_anneal(0.004, 0.006, args.num_epochs - int(first_stage_pct* args.num_epochs))
    prior_mult = 30
    if RESUME:
        res=diffusion_prior.load_state_dict(weights)
        print(res)
    diffusion_prior = diffusion_prior.cuda(args.local_rank)
    print('模型初始化完成')
    optimizer, lr_scheduler = configure_optimizers(args=args, diffusion_prior=diffusion_prior,
                                                   max_lr=args.max_lr)
    diffusion_prior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(diffusion_prior)
    # print('BN同步完成')
    diffusion_prior = torch.nn.parallel.DistributedDataParallel(diffusion_prior, device_ids=[args.local_rank],
                                                                output_device=args.local_rank,
                                                                find_unused_parameters=True)
    print('DDP model')
    if RESUME:
        pass
        #optimizer.load_state_dict(opt_state)
        #lr_scheduler.load_state_dict(scheduler_state)
    diffusion_prior.train()
    print('正在同步')
    synchronize()
    print('同步完成')
    scaler = GradScaler()
    use_cross_sub_loss = args.use_cross_sub_loss == 1
    for epoch in range(start_epoch, args.num_epochs):
        #if epoch==275:
        #    break
        diffusion_prior.train()
        for i, (fmri, img_emb,txt_emb) in enumerate(train_loader):
            torch.cuda.empty_cache()
            #if rank==0:print("1:{}".format(torch.cuda.memory_allocated(0)))
            optimizer.zero_grad()
            fmri = fmri.cuda()
            img_emb = img_emb.cuda()
            txt_emb=txt_emb.cuda()
            if epoch < int(first_stage_pct * args.num_epochs):
                with autocast():
                    batch_weight = F.softmax(torch.randn(2), dim=-1).cuda()
                    loss_nce,loss_nce_global_img,loss_nce_global_txt,loss_prior = diffusion_prior(
                        voxel=fmri,
                        image_embed=img_emb,
                        mix_up=True,
                        subject_id=subject_id,
                        use_cross_sub_loss=False,
                        global_text_emb=txt_emb,
                        inference_mode=False,
                        # epoch_temp=epoch_temp,
                        only_loss_prior=False,
                        is_train=True)
                    loss = batch_weight[0] * (loss_nce+0.5*loss_nce_global_img+0.5*loss_nce_global_txt) + batch_weight[1] * 30 * loss_prior
                    message = f'Epoch:{epoch}--Iter:{i}\n--loss_nce:{loss_nce.item()}--loss_prior:{loss_prior.item()}'
            else:
                with autocast():
                    batch_weight = F.softmax(torch.randn(2), dim=-1).cuda()
                    epoch_temp = soft_loss_temps[epoch - int(first_stage_pct * args.num_epochs)]
                    if epoch<=int(first_stage_pct * args.num_epochs)+2:#using small batch size the first epoch for  smoothing
                        fmri=fmri[:12]
                        img_emb=img_emb[:12]
                        txt_emb=txt_emb[:12]
                    loss_prior, loss_nce,loss_nce_global_img,loss_nce_global_txt=diffusion_prior(
                        voxel=fmri,
                        image_embed=img_emb,
                        mix_up=False,  # return output of subject-specific module
                        subject_id=subject_id,
                        global_text_emb=txt_emb,
                        inference_mode=False,
                        epoch_temp=epoch_temp,
                        only_loss_prior=False,
                        is_train=True)
                    loss = batch_weight[0] * (loss_nce+0.2*loss_nce_global_img+0.2*loss_nce_global_txt) + batch_weight[1] * 30 * loss_prior
                    message = f'Epoch:{epoch}--Iter:{i}\n--loss_nce:{loss_nce.item()}--loss_prior:{loss_prior.item()}'

            scaler.scale(loss).backward()
           # if rank==0:print("4:{}".format(torch.cuda.memory_allocated(0)))
            #torch.nn.utils.clip_grad_norm_(parameters=diffusion_prior.parameters(), max_norm=1., norm_type=2)
            scaler.step(optimizer)
            #if rank==0:print("5:{}".format(torch.cuda.memory_allocated(0)))
            scaler.update()
            #if rank==0:print("6:{}".format(torch.cuda.memory_allocated(0)))
            # loss.backward()
            lr_scheduler.step()
            #if rank==0:print("7:{}".format(torch.cuda.memory_allocated(0)))
            del loss
            # optimizer.step()
            # lr_scheduler.step()
            if rank == 0 and i % 10 == 0:
                print(message)
        if rank==0:
            save_model(epoch, diffusion_prior, optimizer, args.checkpoint_dir, lr_scheduler, 'CrossBrain_Embedding_last.pt')
        if (epoch + 1) % 5 == 0 and rank == 0:
            save_model(epoch, diffusion_prior, optimizer, args.checkpoint_dir, lr_scheduler,
                       'CrossBrain_Embedding_backup.pt')
        # if plot_umap:
        #     print('umap plotting...')
        #     combined = np.concatenate((clip_target.flatten(1).detach().cpu().numpy(),
        #                                aligned_clip_voxels.flatten(1).detach().cpu().numpy()), axis=0)
        #     reducer = umap.UMAP(random_state=42)
        #     embedding = reducer.fit_transform(combined)
        #
        #     colors = np.array([[0, 0, 1, .5] for i in range(len(clip_target))])
        #     colors = np.concatenate((colors, np.array([[0, 1, 0, .5] for i in range(len(aligned_clip_voxels))])))
        #
        #     fig = plt.figure(figsize=(5, 5))
        #     plt.scatter(
        #         embedding[:, 0],
        #         embedding[:, 1],
        #         c=colors)
        #     if wandb_log:
        #         logs[f"val/umap"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
        #         plt.close()
        #     else:
        #         plt.savefig(os.path.join(outdir, f'umap-val-epoch{epoch:03d}.png'))
        #         plt.show()
    dist.destroy_process_group()  # 销毁进程组


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, help='distributed backend init_method')
    parser.add_argument('--num_workers', type=int, default=0, help='num of wokers for dataloader')
    parser.add_argument('--num_epochs', type=int, default=50, help='how many epochs to train')
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--shuffle', type=bool, default=False, help='whether permute the order of samples')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='path to your save checkpoint')
    parser.add_argument('--use_cross_sub_loss', type=int, default=1, help='')
    parser.add_argument('--fmri_data_folder', type=str, default='', help='path to your preprocessed fmri data')
    parser.add_argument('--resume', type=str, help='path to your partially trained model')
    parser.add_argument('--save_every_n_steps', default=200, type=int, help='Save a checkpoint every n steps')
    parser.add_argument('--subject_id', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--use_token_attention', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=3.)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--use_patched_mask', type=int, default=1)
    parser.add_argument('--train_single_subject', type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint_dir) and args.rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
