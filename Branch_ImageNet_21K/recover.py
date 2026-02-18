import argparse
import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

from utils import BNFeatureHook, load_model_weights, lr_cosine_policy, PreImgPathCache, ShufflePatches, FastPreImgPathCache


def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 25
    batch_size = args.batch_size

    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    targets_all = torch.LongTensor(np.arange(10450))


    if args.initial_img_dir != "None":
        initial_img_cache = FastPreImgPathCache(args.initial_img_dir,transforms=transforms.Compose([
                                                                transforms.Resize((224,224)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                ShufflePatches(2)],
                                                               ))
    
    print("Image Cache is initialized with", args.initial_img_dir)                                                       
    saved_iterations = 0
    K = args.K
    loss_threshold = args.loss_threshold
    for kk in range(0, 10450, batch_size):
        targets = targets_all[kk : min(kk + batch_size, 10450)].to("cuda")
        print("Targets list: ", targets)

        data_type = torch.float
        if args.initial_img_dir != "None":
            # Sample images for each target
            sampled_images = [initial_img_cache.random_img_sample(_target) for _target in targets.tolist()]
            
            # Stack the valid images into a tensor
            inputs = torch.stack(sampled_images, 0).cuda().to(data_type)            
            inputs.requires_grad_(True)
        else:
            inputs = torch.randn((targets.shape[0], 3, 224, 224), requires_grad=True, device="cuda", dtype=data_type)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter, args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        #criterion = nn.CrossEntropyLoss()
        #criterion = criterion.cuda()

        high_loss_crops = [[] for _ in range(batch_size)]  # Stores crops per image
        high_loss_values = [[] for _ in range(batch_size)]  # Stores loss values per image

        class AugWithTracking:
            def __init__(self, batch_size):
                self.cropper = transforms.RandomResizedCrop(224, scale=(0.5, 1))
                self.flipper = transforms.RandomHorizontalFlip()
                self.last_crops = [None] * batch_size  # Track crops per image
                self.selected_indices = [None] * batch_size  # Track selected crop index per image

            def __call__(self, imgs, iteration, high_loss_crops, high_loss_values):
                batch_size = imgs.shape[0]
                cropped_imgs = []
                
                for img_idx in range(batch_size):
                    if iteration > K and high_loss_crops[img_idx]:  # Use stored high-loss crops
                        loss_weights = torch.tensor(high_loss_values[img_idx], device=imgs.device)
                        selection_probs = torch.nn.functional.softmax(loss_weights, dim=0)

                        selected_idx = torch.multinomial(selection_probs, 1).item()
                        i, j, h, w = high_loss_crops[img_idx][selected_idx]
                        self.selected_indices[img_idx] = selected_idx  # Track selected crop
                    else:  # Random cropping in early iterations
                        self.selected_indices[img_idx] = None
                        i, j, h, w = self.cropper.get_params(imgs[img_idx], self.cropper.scale, self.cropper.ratio)
                    
                    self.last_crops[img_idx] = (i, j, h, w)  
                    cropped_img = F.resized_crop(imgs[img_idx], i, j, h, w, self.cropper.size)
                    cropped_img = self.flipper(cropped_img) 
                    cropped_imgs.append(cropped_img)

                return torch.stack(cropped_imgs)  

        aug_function = AugWithTracking(batch_size)

        for iteration in range(iterations_per_layer):
            
            if iteration > K and all(len(crops) == 0 for crops in high_loss_crops):
                print(f"No more high-loss crops remaining. Early stopping at iteration {iteration}")
                saved_iterations += iterations_per_layer - iteration
                break  

            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            inputs_jit = aug_function(inputs, iteration, high_loss_crops, high_loss_values)
            selected_indices = aug_function.selected_indices  # Track per-image crop selection

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            criterion_ce = nn.CrossEntropyLoss(reduction='none').cuda()
            loss_ce_all = criterion_ce(outputs.float(), targets)
            loss_ce = loss_ce_all.mean()

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # combining losses
            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            crop_loss = loss_ce_all
            loss_vals = crop_loss.detach().cpu().numpy()
            
            for img_idx, crop in enumerate(aug_function.last_crops[:targets.shape[0]]):
                if crop is None:
                    continue
                i, j, h, w = crop
                if loss_vals[img_idx] > loss_threshold and iteration <= K:
                    crop_tuple = (i, j, h, w)
                    if crop_tuple in high_loss_crops[img_idx]:
                        crop_idx = high_loss_crops[img_idx].index(crop_tuple)
                        high_loss_values[img_idx][crop_idx] = loss_vals[img_idx]
                    else:
                        high_loss_crops[img_idx].append(crop_tuple)
                        high_loss_values[img_idx].append(loss_vals[img_idx])
            
                if selected_indices[img_idx] is not None:
                    new_loss = loss_vals[img_idx]
                    sel_idx = selected_indices[img_idx]
                    if new_loss > loss_threshold:
                        high_loss_values[img_idx][sel_idx] = new_loss
                    else:
                        del high_loss_crops[img_idx][sel_idx]
                        del high_loss_values[img_idx][sel_idx]

            if (iteration % save_every == 0 or iteration == iterations_per_layer - 1): # and hook_for_display is not None:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())
                '''if hook_for_display is not None:
                    acc_jit, _ = hook_for_display(inputs_jit, targets)
                    acc_image, loss_image = hook_for_display(inputs, targets)

                    metrics = {
                        "crop/acc_crop": acc_jit,
                        "image/acc_image": acc_image,
                        "image/loss_image": loss_image,
                    }
                    wandb_metrics.update(metrics)

                metrics = {
                    "crop/loss_ce": loss_ce.item(),
                    "crop/loss_r_bn_feature": loss_r_bn_feature.item(),
                    "crop/loss_total": loss.item(),
                }
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics)'''

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            # inputs.data = clip(inputs.data) # do not clip since no normalization in training 21k squeezed model
            inputs.data = inputs.data.clamp(0, 1)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            # best_inputs = denormalize(best_inputs) # donot denormalize since no normalization in training 21k squeezed model
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        if args.verifier:
            exit()
    torch.cuda.empty_cache()


def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        os.makedirs(args.syn_data_path, exist_ok=True)

        # save into separate folders
        dir_path = "{}/new{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:03d}.jpg".format(class_id, ipc_id)
        try:
            os.makedirs(dir_path, exist_ok=True)
        except FileExistsError:
            pass 
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
        loss = nn.CrossEntropyLoss()(output, target)

    print("Verifier accuracy: ", prec1.item())
    return prec1.item(), loss.item()


def main_syn(args, ipc_id):
    os.makedirs(args.syn_data_path, exist_ok=True)

    import timm

    model_teacher = timm.create_model(args.arch_name, pretrained=False, num_classes=10450)
    model_teacher = load_model_weights(
        model_teacher,
        args.arch_path,
    )

    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher = model_teacher.cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if args.verifier:
        # mobilenet_v3 as verifier
        model_verifier = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=10450)
        model_verifier = load_model_weights(
            model_verifier,
            args.verifier_path,
        )

        model_verifier = model_verifier.cuda()
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False

        hook_for_display = lambda x, y: validate(x, y, model_verifier)
    else:
        hook_for_display = None
    get_images(args, model_teacher, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser("CDA for ImageNet-21K")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--syn-data-path", type=str, default="./syn-data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=32, type=int, help="random shift on the synthetic data")
    parser.add_argument("--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn")
    """Model related flags"""
    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")
    parser.add_argument("--arch-path", type=str, default="", help="path to the pretrained model")
    parser.add_argument("--verifier", action="store_true", help="whether to evaluate synthetic data with another model")
    parser.add_argument("--verifier-path", type=str, default="", help="path to the verifier model")
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument("--G", default="-1", type=str)
    parser.add_argument("--ipc-start", default=0, type=int)
    parser.add_argument("--ipc-end", default=1, type=int)
    parser.add_argument("--wandb-key", default="", type=str)
    parser.add_argument('--initial-img-dir', type=str, default="None", help="imgs used for initialization")
    parser.add_argument('--K', type=int, default=700,
                        help="the iteration threshold for selective cropping")
    parser.add_argument('--loss-threshold', type=float, default=0.7,
                        help="loss threshold for selective cropping")

    args = parser.parse_args()

    assert args.milestone >= 0 and args.milestone <= 1

    if args.G != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.G
        print("set CUDA_VISIBLE_DEVICES to ", args.G)

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == "__main__":
    args = parse_args()

    if not wandb.api.api_key:
        wandb.login(key=args.wandb_key)
    else:
        wandb.init(project="cda-gen-21k", name=args.exp_name)
    global wandb_metrics
    wandb_metrics = {}
    # for ipc_id in range(0,50):
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        #wandb.log({"ipc_id": ipc_id})
        main_syn(args, ipc_id)
    #main_syn(args, args.ipc_start)

    wandb.finish()
    print("Done.")
