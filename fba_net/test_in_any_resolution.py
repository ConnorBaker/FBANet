import argparse
import os

import torch
import utils

# from skimage.metrics import structural_similarity as ssim_loss
from ManualDataset import ManualDatasets_test
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from utils.dataset_utils import tensor_divide_burst, tensor_merge

parser = argparse.ArgumentParser(description="RGB super-resolution test")
parser.add_argument("--input_dir", default="...", type=str, help="Directory of validation images")
# /userhome/aimia/sunyj/Dataset/Manual/Manual_dataset-v2
# /userhome/aimia/sunyj/Dataset/DRealBSR_RGB/
parser.add_argument("--result_dir", default="./results/Motion_MFSR_0.5/", type=str, help="Directory for results")
parser.add_argument("--weights", default="...", type=str, help="Path to weights")
parser.add_argument("--gpus", default="0", type=str, help="CUDA_VISIBLE_DEVICES")
parser.add_argument("--arch", default="BaseModel", type=str, help="arch")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size for dataloader")
parser.add_argument("--save_images", action="store_true", help="Save denoised images in result directory")
parser.add_argument("--embed_dim", type=int, default=64, help="number of data loading workers")
parser.add_argument("--win_size", type=int, default=10, help="number of data loading workers")
parser.add_argument("--token_projection", type=str, default="linear", help="linear/conv token projection")
parser.add_argument("--token_mlp", type=str, default="leff", help="ffn/leff token mlp")
# args for vit
parser.add_argument("--vit_dim", type=int, default=256, help="vit hidden_dim")
parser.add_argument("--vit_depth", type=int, default=12, help="vit depth")
parser.add_argument("--vit_nheads", type=int, default=8, help="vit hidden_dim")
parser.add_argument("--vit_mlp_dim", type=int, default=512, help="vit mlp_dim")
parser.add_argument("--vit_patch_size", type=int, default=16, help="vit patch_size")
parser.add_argument("--global_skip", action="store_true", default=False, help="global skip connection")
parser.add_argument("--local_skip", action="store_true", default=False, help="local skip connection")
parser.add_argument("--vit_share", action="store_true", default=False, help="share vit module")

parser.add_argument("--train_ps", type=int, default=160, help="patch size of training sample")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = ManualDatasets_test(root=args.input_dir, burst_size=14, split="val")
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

psize = 80
overlap = 40

model_restoration.cuda()
model_restoration.eval()
with torch.no_grad():
    lpips_val = []
    psnr_val = []
    ssim_val = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test["LR"].cuda()
        rgb_gt = torch.zeros(rgb_noisy.shape[0], rgb_noisy.shape[2], 4 * rgb_noisy.shape[3], 4 * rgb_noisy.shape[4])
        filenames = data_test["burst_name"]

        with torch.cuda.amp.autocast():
            tensor = rgb_noisy  # [B,5,3,H,W]
            hr_size = rgb_gt  # [B,3,4*H,4*W]

            blocks = tensor_divide_burst(tensor, psize, overlap)  # blocks -> List of N * [1,T,C,h,w]
            blocks = torch.cat(blocks, dim=0)  # blocks -> [N,T,C,h,w]
            results = []

            iters = blocks.shape[0]

            for idx in range(iters):
                if idx + 1 == iters:
                    input = blocks[idx:]
                else:
                    input = blocks[idx : (idx + 1)]

                restored_output = model_restoration(input)

                results.append(restored_output)  # results -> List of [1,3,4*h,4*w]
                print("Processing Image: %d Part: %d / %d" % (ii, idx + 1, iters))

            results = torch.cat(results, dim=0)  # results -> [N,3,4*h,4*w]
            rgb_restored = tensor_merge(results, hr_size, psize=psize * 4, overlap=overlap * 4)

        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        if args.save_images:
            transform = transforms.Compose([transforms.ToPILImage()])
            for restored_index in range(len(rgb_restored)):
                if rgb_restored[restored_index].dim() == 3:
                    sr_img_saved = transform(rgb_restored[restored_index])
                    sr_img_saved.save("{}/{}.png".format(args.result_dir, filenames[restored_index]))

# lpips_val = sum(lpips_val)/len(test_loader)
# ssim_val = sum(ssim_val)/len(test_loader)
# psnr_val = sum(psnr_val) / len(test_dataset)
# print("PSNR_v2: %f SSIM: %f LPIPS: %f" %(psnr_val, ssim_val, lpips_val))
