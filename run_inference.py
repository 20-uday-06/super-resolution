# Run MRUNet super‑resolution on ASTER 1 km GeoTIFFs
# ----------------------------------------------------------
import os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from model import MRUNet
from tiff_process_aster import tiff_process           # ← NEW helper
from utils import upsampling, normalization           # reuse your utils

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='MRUNet inference on ASTER 1 km TIFFs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--datapath',
                    default='ASTER/ASTER_2021_AST_07XT',
                    help='folder with ASTER GeoTIFFs')
parser.add_argument('--pretrained',
                    default='MRUNet_ASTER_1k_250m.pt',
                    help='checkpoint (.pt) from training script')
parser.add_argument('--savepath',
                    default='results_aster',
                    help='folder where PNG figures are saved')
parser.add_argument('--max_val',
                    default=333.32, type=float,
                    help='normalization factor (max pixel value of training set)')
parser.add_argument('--band',        default=2,  type=int, help='ASTER band')
parser.add_argument('--native_res',  default=15, type=int, choices=[15,30],
                    help='native ASTER resolution (m)')
args = parser.parse_args()

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(args.savepath, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load model
    model = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)
    ckpt  = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 2) Build LR/HR stacks from ASTER files
    print("► Building LR/HR arrays …")
    LR_stack, HR_stack = tiff_process(
        args.datapath,
        band=args.band,
        native_res=args.native_res
    )

    max_val = args.max_val
    scale   = 4                 # 1 km ➜ 250 m

    # 3) Inference loop
    for idx, (lr, hr_gt) in enumerate(zip(LR_stack, HR_stack)):
        # Bicubic upsampling (baseline)
        bicubic = cv2.resize(lr, (hr_gt.shape[1], hr_gt.shape[0]), cv2.INTER_CUBIC)

        # Model input (normalize + channel/batch dims)
        inp = normalization(bicubic, max_val)
        inp = torch.tensor(inp[None, None, :, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            sr = model(inp).cpu().numpy()[0, 0] * max_val

        # 4) Save comparison figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        cmap = 'jet'; fs = 14

        ax[0].imshow(hr_gt,   cmap=cmap); ax[0].set_title("GT 250 m", fontsize=fs); ax[0].axis('off')
        ax[1].imshow(bicubic, cmap=cmap); ax[1].set_title("Bicubic",  fontsize=fs); ax[1].axis('off')
        ax[2].imshow(sr,      cmap=cmap); ax[2].set_title("SR 250 m", fontsize=fs); ax[2].axis('off')

        plt.tight_layout()
        out_png = os.path.join(args.savepath, f"sr_{idx:04d}.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved {out_png}")

if __name__ == '__main__':
    main()
