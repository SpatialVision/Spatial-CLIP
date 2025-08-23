#!/usr/bin/env python3
"""
BLINK Benchmark inference with Spatial-CLIP & DepthAnythingV2
Date: 2024-06
Description:
    This script evaluates a Spatial-CLIP model on the BLINK benchmark
    using RGB + depth features from DepthAnythingV2.
"""

import os
import sys
import json
from collections import defaultdict
from typing import List, Tuple, Optional
from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from PIL import Image
from tqdm import tqdm

# Custom imports (ensure repo paths are correct)
sys.path.append(os.path.abspath(".."))
sys.path.append('../Depth-Anything-V2')
import spat_clip as sclip
from depth_anything_v2.dpt import DepthAnythingV2


# ==============================
# Configuration
# ==============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

def to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB format."""
    return image.convert("RGB")

clip_transform = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(336),
    to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    ),
])

# depth_transform is not used in the original code, so it's kept as is.
depth_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])


def load_depth_model(encoder: str = "vitl") -> DepthAnythingV2:
    """Load pretrained DepthAnythingV2."""
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt_path = f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    model.to(torch.float32).to(DEVICE)
    return model

class BLINK:
    """Data structure for BLINK benchmark annotations."""

    def __init__(self, annotation_file: Optional[str] = None):
        self.anns = defaultdict(list)
        self.imgs = {}
        self.answers = {}
        self.task = {}
        if annotation_file:
            with open(annotation_file, 'r') as f:
                datas = json.load(f)
                for data in datas:
                    self.anns[data['idx']] = data['captions']
                    self.imgs[data['idx']] = data['image_1']
                    self.answers[data['idx']] = data['answer']
                    self.task[data['idx']] = data['sub_task']


class BLINK_Benchmark(Dataset):
    """BLINK Benchmark Dataset for RGB data loading."""

    # ### MODIFICATION START ###
    # Removed depth_model from the constructor
    def __init__(self, root_dir: str):
    # ### MODIFICATION END ###
        self.root_dir = root_dir
        self.dataset = BLINK(os.path.join(root_dir, 'test.json'))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = list(self.dataset.anns.values())
        self.transform = clip_transform
        # ### MODIFICATION START ###
        # self.depth_model is removed, it's no longer a property of the dataset
        # ### MODIFICATION END ###

    def __len__(self) -> int:
        return len(self.image_ids)

    # ### MODIFICATION START ###
    # Modified __getitem__ to return only image and answer
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'output_images', self.dataset.imgs[img_id])
        image = Image.open(image_path).convert('RGB')
        
        image_tensor = self.transform(image)
        
        # The following lines are removed as they perform GPU computation in the worker process:
        # depth_map = self.depth_model(image_tensor.unsqueeze(0).to(DEVICE)).squeeze(0)

        answer = torch.tensor(self.dataset.answers[img_id], dtype=torch.long)
        
        # Return only the image tensor and the answer
        return image_tensor, answer
    # ### MODIFICATION END ###


@torch.no_grad()
def zeroshot_classifier(captions: List[List[str]], model, device: torch.device) -> torch.Tensor:
    """Compute zeroshot text embeddings."""
    zeroshot_weights = []
    for caption_set in tqdm(captions, desc="Encoding text"):
        texts = sclip.tokenize(caption_set).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights.append(class_embeddings)
    return torch.stack(zeroshot_weights, dim=0).to(device)


# ### MODIFICATION START ###
# Modified test_epoch to accept depth_model and perform depth calculation inside the loop
@torch.no_grad()
def test_epoch(model, dataloader: DataLoader, text_embeddings: torch.Tensor, depth_model: DepthAnythingV2) -> float:
# ### MODIFICATION END ###
    """Test one epoch on BLINK."""
    corr_pred, total_num = 0, 0
    # ### MODIFICATION START ###
    # The loop now unpacks images and answers, as returned by the modified __getitem__
    for images, answer in tqdm(dataloader, desc="Testing"):
    # ### MODIFICATION END ###
        images = images.to(DEVICE)
        answer = answer.to(DEVICE)
        
        # ### MODIFICATION START ###
        # Calculate depth map inside the main process loop, after moving data to GPU
        depth = depth_model(images)
        # ### MODIFICATION END ###
        
        image_features = model.visual(images, depth, pos_embed="")
        image_features /= image_features.norm(dim=-1, keepdim=True)

        score = (image_features.unsqueeze(1).repeat(1, text_embeddings.size(1), 1)
                 * text_embeddings)
        pred = score.sum(dim=-1).topk(1, dim=1)[1].squeeze(dim=1)
        corr_pred += (pred == answer).sum().item()
        total_num += images.size(0)
    return corr_pred / total_num


def run_blink_eval(model, data_root: str):
    """Full BLINK Evaluation pipeline."""
    depth_model = load_depth_model("vitl")
    
    # ### MODIFICATION START ###
    # The dataset constructor no longer needs the depth_model
    testset = BLINK_Benchmark(data_root)
    # ### MODIFICATION END ###
    
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    text_embeddings = zeroshot_classifier(testset.captions, model, DEVICE)
    
    # ### MODIFICATION START ###
    # Pass the depth_model to the test_epoch function
    acc = test_epoch(model, testloader, text_embeddings, depth_model)
    # ### MODIFICATION END ###
    
    print(f"[RESULT] BLINK Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # ### MODIFICATION START ###
    # Best practice: Set the start method to 'spawn' for CUDA compatibility with multiprocessing.
    # This should be at the very top of the main block.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # It's already been set, so we can ignore this.
    # ### MODIFICATION END ###
        
    import argparse
    parser = argparse.ArgumentParser(description="BLINK Benchmark Evaluation")
    parser.add_argument("--data_root", type=str, default="../dataset/BLINK/Spatial_Relation",
                        help="Path to BLINK dataset root")
    parser.add_argument("--ckpt", type=str, default="../checkpoint/iter_5000.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="vitl", choices=MODEL_CONFIGS.keys(),
                        help="Depth model encoder type")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference")
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    model, _ = sclip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    ckpt_path = hf_hub_download(
        repo_id="demo911/spatialclip",
        repo_type="dataset", 
        filename="iter_5000.pth"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt, strict=True)
    model.to(DEVICE)

    # Run evaluation
    run_blink_eval(model, args.data_root)

