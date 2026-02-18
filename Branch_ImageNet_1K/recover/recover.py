'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
import torchvision.datasets
from PIL import Image

import torch.multiprocessing as mp
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import torch.distributed as dist
mp.set_sharing_strategy('file_system')
from utils import *
from torchvision.transforms import functional as F
from torch.amp import autocast, GradScaler
import torch.cuda

    
def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main_worker(gpu, ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range,K,loss_threshold,AMP):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    model_teacher = [_model_teacher.cuda(gpu).eval() for _model_teacher in model_teacher]
    scaler = torch.amp.GradScaler()
    for _model_teacher in model_teacher:
        for p in _model_teacher.parameters():
            p.requires_grad = False

    model_verifier = model_verifier.cuda(gpu)
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    save_every = 20
    batch_size = args.batch_size
    best_cost = 1e4
    load_tag_dict = [True for i in range(len(model_teacher))]
    loss_r_feature_layers = [[] for _ in range(len(model_teacher))]
    load_tag = True

    for i, (_model_teacher) in enumerate(model_teacher):
        for name, module in _model_teacher.named_modules():
            if args.aux_teacher[i] in ["wide_resnet50_2", "regnet_y_400mf", 
            "regnet_x_400mf"]:
                full_name = str(_model_teacher.__class__.__name__) + "_" + str(args.aux_teacher[i]) + "=" + name
            else:
                full_name = str(_model_teacher.__class__.__name__) + "=" + name
            if isinstance(module, nn.BatchNorm2d):
                _hook_module = BNFeatureHook(module,save_path=args.statistic_path,
                                            name=full_name,
                                            gpu=gpu,training_momentum=args.training_momentum,
                                            flatness_weight=args.flatness_weight,
                                            category_aware=args.category_aware)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

            elif isinstance(module, nn.Conv2d):
                _hook_module = ConvFeatureHook(module, save_path=args.statistic_path,
                                               name=full_name,
                                               gpu=gpu, training_momentum=args.training_momentum,
                                               drop_rate=args.drop_rate,
                                               flatness_weight=args.flatness_weight,
                                               category_aware=args.category_aware)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

    sub_batch_size = int(batch_size // ngpus_per_node)

    if args.initial_img_dir != "None":
        initial_img_cache = PreImgPathCache(args.initial_img_dir,transforms=transforms.Compose([
                                                                transforms.Resize((224,224)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),
                                                                ShufflePatches(2)],
                                                                ))
    if args.category_aware == "local":
        original_img_cache = PreImgPathCache(args.train_data_path,transforms=transforms.Compose([
                                                                transforms.Resize((224,224)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),]
                                                                ))
    if not load_tag:
        train_dataset = torchvision.datasets.ImageFolder(root=args.train_data_path,
                                                         transform=transforms.Compose([
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])]))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   num_workers=4,
                                                   batch_size=256,
                                                   drop_last=False,
                                                   shuffle=True)

        with torch.no_grad():
            for j, _model_teacher in enumerate(model_teacher):
                if not load_tag_dict[j]:
                    print(f"conduct backbone {args.aux_teacher[j]} statistics")
                    for i, (data, targets) in tqdm(enumerate(train_loader)):
                        data = data.cuda(gpu)
                        targets = targets.cuda(gpu)
                        for _loss_t_feature_layer in loss_r_feature_layers[j]:
                            _loss_t_feature_layer.set_label(targets)
                        _ = _model_teacher(data)
                    
                    for _loss_t_feature_layer in loss_r_feature_layers[j]:
                        _loss_t_feature_layer.save()

        print("Training Statistic Information Is Successfully Saved")
    else:
        print("Training Statistic Information Is Successfully Load")

    for j in range(len(loss_r_feature_layers)):
        for _loss_t_feature_layer in loss_r_feature_layers[j]:
            _loss_t_feature_layer.set_hook(pre=False)

    targets_all_all = torch.LongTensor(np.arange(1000))[None, ...].expand(len(ipc_id_range), 1000).contiguous().view(-1)
    ipc_id_all = torch.LongTensor(ipc_id_range)[..., None].expand(len(ipc_id_range), 1000).contiguous().view(-1)

    total_number = 1000 * (ipc_id_range[-1] + 1 - ipc_id_range[0])
    turn_index = torch.LongTensor(np.arange(total_number)).view(len(ipc_id_range), 1000) \
        .transpose(1, 0).contiguous().view(-1)

    counter = 0
    saved_iterations = 0
    print(f"Starting loop on GPU {gpu}, total_number: {total_number}, batch_size: {batch_size}")
    for zz in range(0, total_number, batch_size): 
        sub_turn_index = turn_index[zz + gpu * sub_batch_size:min(zz + (gpu + 1) * sub_batch_size, total_number)]

        
        targets = targets_all_all[sub_turn_index].cuda(gpu)
        ipc_ids = ipc_id_all[sub_turn_index].cuda(gpu)

        if targets.numel() == 0:
            continue
                    
        data_type = torch.float
        sub_batch_size = min(zz + (gpu + 1) * sub_batch_size, total_number) - (zz + gpu * sub_batch_size)
        if sub_batch_size < 0:
            continue
        print(f"In GPU {gpu}, targets is set as: \n{targets}\n, ipc_ids is set as: \n{ipc_ids}")
        
        if args.initial_img_dir != "None":

            sampled_images = [initial_img_cache.random_img_sample(_target) for _target in targets.tolist()]
            
            inputs = torch.stack(sampled_images, 0).to(f'cuda:{gpu}').to(data_type)
            inputs.requires_grad_(True)
        else:
            inputs = torch.randn((sub_batch_size, 3, 224, 224), requires_grad=True, device=f'cuda:{gpu}', dtype=data_type)
        
        if args.category_aware == "local":
            expand_ratio = int(50000 / (args.ipc_number*1000))
            tea_images = torch.stack([original_img_cache.random_img_sample(_target) for _target in (targets.tolist() * expand_ratio)],0).to(f'cuda:{gpu}').to(data_type)
            with torch.no_grad():
                for id in range(len(args.aux_teacher)):
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.set_tea()
                    sub_outputs = model_teacher[id](tea_images)
        
        iterations_per_layer = args.iteration
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        #inputs_ema = EMA(alpha=args.ema_alpha, initial_value=inputs)


        high_loss_crops = [[] for _ in range(sub_batch_size)]  
        high_loss_values = [[] for _ in range(sub_batch_size)]  
        
        class ExplorationExploitationAug:
            def __init__(self, batch_size):
                self.cropper = transforms.RandomResizedCrop(224, scale=(0.5, 1))
                self.flipper = transforms.RandomHorizontalFlip()
                self.last_crops = [None] * batch_size  
                self.selected_indices = [None] * batch_size  

            def __call__(self, imgs, iteration, high_loss_crops, high_loss_values):
                batch_size = imgs.shape[0]
                cropped_imgs = []

                # cropping per image
                for img_idx in range(batch_size):
                    if iteration > K and high_loss_crops[img_idx]:

                        # Exploitation Phase
                        loss_weights = torch.tensor(high_loss_values[img_idx], device=imgs.device)
                        selection_probs = torch.nn.functional.softmax(loss_weights, dim=0)

                        selected_idx = torch.multinomial(selection_probs, 1).item()
                        i, j, h, w = high_loss_crops[img_idx][selected_idx]
                        self.selected_indices[img_idx] = selected_idx  
                    else:
                        # Exploration Phase
                        self.selected_indices[img_idx] = None
                        i, j, h, w = self.cropper.get_params(imgs[img_idx], self.cropper.scale, self.cropper.ratio)
                    
                    self.last_crops[img_idx] = (i, j, h, w)  
                    cropped_img = F.resized_crop(imgs[img_idx], i, j, h, w, self.cropper.size)
                    cropped_img = self.flipper(cropped_img)  
                    cropped_imgs.append(cropped_img)

                return torch.stack(cropped_imgs)  

       
        aug_function = ExplorationExploitationAug(sub_batch_size)

        for iteration in range(iterations_per_layer):
            if iteration > K and all(len(crops) == 0 for crops in high_loss_crops):
                # Early stopping if no high-loss crops remain
                print(f"No more high-loss crops remaining. Early stopping at iteration {iteration}")
                saved_iterations += iterations_per_layer - iteration
                break  

            lr_scheduler(optimizer, iteration, iteration)
        
            inputs_jit = aug_function(inputs, iteration, high_loss_crops, high_loss_values)
            selected_indices = aug_function.selected_indices  
        
            id = counter % len(model_teacher)
            for mod in loss_r_feature_layers[id]:
                mod.set_label(targets)
            
            counter += 1
            optimizer.zero_grad()
            for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                mod.set_ori()

            if AMP:
                if id == 2:
                    with autocast('cuda'):  
                        sub_outputs = model_teacher[id](inputs_jit)
                else:
                    sub_outputs = model_teacher[id](inputs_jit)
            else:
                sub_outputs = model_teacher[id](inputs_jit)



                
            rescale = [args.first_multiplier] + [1. for _ in range(len(loss_r_feature_layers[id]) - 1)]
            loss_r_feature = sum(
                    [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers[id])])
            loss_aux = args.r_loss * loss_r_feature

            criterion_ce = nn.CrossEntropyLoss(reduction='none').cuda()
            loss_ce_all = criterion_ce(sub_outputs.float(), targets)
            loss_ce = loss_ce_all.mean()

            loss_ema_ce = torch.tensor(0., device=inputs_jit.device)
            loss = loss_ce + loss_aux + loss_ema_ce * args.flatness_weight
            
            crop_loss = loss_ce_all
            
            loss_vals = crop_loss.detach().cpu().numpy()
            

            # updating high-loss crops per image
            
            for img_idx, (i, j, h, w) in enumerate(aug_function.last_crops):
                if loss_vals[img_idx] > loss_threshold and iteration <= K:
                    crop = (i, j, h, w)
                    if crop in high_loss_crops[img_idx]:  
                        crop_idx = high_loss_crops[img_idx].index(crop)
                        high_loss_values[img_idx][crop_idx] = loss_vals[img_idx]  
                    else:
                        high_loss_crops[img_idx].append(crop)  
                        high_loss_values[img_idx].append(loss_vals[img_idx])

                if selected_indices[img_idx] is not None:
                    new_loss = loss_vals[img_idx]
                    sel_idx = selected_indices[img_idx]

                    if new_loss > loss_threshold:
                        high_loss_values[img_idx][sel_idx] = new_loss
                    else:
                        del high_loss_crops[img_idx][sel_idx]
                        del high_loss_values[img_idx][sel_idx]


            
            if iteration % save_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_feature", loss_r_feature.item())
                print("loss_ema_ce", loss_ema_ce.item())
                print("main criterion",
                      loss_ce.item())

                best_inputs = inputs.data.clone()  
                
            if AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                (loss).backward()
                optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)
            if gpu == 0 and (best_cost > loss.item() or iteration == 1):
                best_inputs = inputs.data.clone()
        
                
        #del inputs_ema
        
        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_ids)
        
        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()        



def save_images(args, images, targets, ipc_ids,iter=None):
    ipc_id_range = ipc_ids
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        if iter is None:
            place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id_range[id])
        else:
            place_to_store = dir_path + '/class{:03d}_id{:03d}_iter{:04d}.jpg'.format(class_id, ipc_id_range[id],iter)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())
    return prec1.item()

def main_syn():
    parser = argparse.ArgumentParser(
        "G-VBSM: applying generalized matching for data condensation")
    """Data save flags"""
    parser.add_argument('--flatness', action='store_true', default=False,
                        help='encourage the flatness or not')
    parser.add_argument('--flatness-weight', type=float, default=1.,
                        help='the weight of flatness weight')
    parser.add_argument('--ema_alpha', type=float, default=0.9,
                        help='the weight of EMA learning rate')
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--ipc-number', type=int, default=50, help='the number of each ipc')
    parser.add_argument('--initial-img-dir', type=str, default="None", help="imgs used for initialization")
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--gpu-id', type=str, default='0,1')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--category-aware', default="global", type=str, help='category-aware matching (local or global)')
    parser.add_argument('--r-loss', type=float, default=0.05,
                        help='coefficient for BN and Conv feature distribution regularization')
    parser.add_argument('--first-multiplier', type=float, default=10.,
                        help='additional multiplier on first layer of L_bn or L_conv')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--training-momentum', type=float, default=0.4,
                        help="controls the form of score distillation sampling")
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help="controls the efficiency of GSM")
    parser.add_argument('--nuc-norm', type=float, default=0.00001,
                        help='coefficient for total variation Nuclear loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--tau', type=float, default=4.0, help='the temperature of nuc norm')
    parser.add_argument('--average_grad_ratio', default=0., type=float)
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--train-data-path', type=str, default='./imagenet/train',
                        help="the path of the ImageNet-1k's training set")
    parser.add_argument('--statistic-path', type=str, default='./statistic',
                        help="the path of the statistic file"),
    parser.add_argument('--K', type=int, default=700,
                        help="the number of iterations for exploration")
    parser.add_argument('--seed', type=int, default=None, 
                        help="random seed for reproducibility") 
    parser.add_argument('--loss-threshold', type=float, default=0.5,
                        help="epsilon, loss threshold for high-loss crops")
    parser.add_argument('--AMP', type=int, default=1,
                        help="whether to use automatic mixed precision for reducing memory usage")
    args = parser.parse_args()

    print(args)
    if args.seed is not None:
        set_seed(args.seed)

    K = args.K

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    aux_teacher = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5", "alexnet"] #  "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5", "alexnet" "densenet121", "convnext_tiny"
    args.aux_teacher = aux_teacher
    model_teacher = []
    for name in aux_teacher:
        model_teacher.append(models.__dict__[name](pretrained=True))
    
    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    ipc_id_range = list(range(0, args.ipc_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node,
             args=(ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range,K,args.loss_threshold,bool(args.AMP)))


if __name__ == '__main__':
    main_syn()
