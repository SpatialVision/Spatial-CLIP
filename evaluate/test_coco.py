import argparse
import os
import sys
from collections import OrderedDict
from typing import Callable, Dict, Tuple
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
sys.path.append('../Depth-Anything-V2')
import spat_clip as sclip
from depth_anything_v2.dpt import DepthAnythingV2


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Spatial-CLIP Retrieval Evaluation on COCO")
    
    parser.add_argument('--data-root', type=str, default='/path/to/your/coco/val2017',
                        help='Root directory of the COCO validation dataset.')
    parser.add_argument('--ann-file', type=str, default='/path/to/your/coco/annotations/captions_val2017.json',
                        help='Path to the COCO validation annotation file.')
    
    parser.add_argument('--clip-repo-id', type=str, default='demo911/spatialclip',
                        help='Hugging Face repository ID for the Spatial-CLIP checkpoint.')
    parser.add_argument('--clip-filename', type=str, default='iter_5000.pth',
                        help='Filename of the Spatial-CLIP checkpoint in the repository.')
                        
    parser.add_argument('--depth-model-encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Encoder type for the DepthAnythingV2 model.')
    parser.add_argument('--depth-ckpt-path', type=str, default='../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
                        help='Local path to the DepthAnythingV2 checkpoint.')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation (e.g., "cuda:0" or "cpu").')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='Batch size for evaluation.')
    parser.add_argument('-w', '--num-workers', type=int, default=4,
                        help='Number of data loader workers.')
                        
    return parser.parse_args()


class CocoRetrievalDataset(Dataset):
    """
    COCO Dataset for image-text retrieval tasks.
    Returns pre-processed images and tokenized captions.
    """
    def __init__(self, root: str, ann_file: str, transform: Callable, tokenize: Callable):
        self.coco = CocoCaptions(root=root, annFile=ann_file)
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, captions = self.coco[idx]
        
        image_tensor = self.transform(image)
        
        # Use first 5 captions for consistency in retrieval task
        tokenized_captions = self.tokenize(captions[:5])
        
        return image_tensor, tokenized_captions


def load_depth_model(encoder: str, ckpt_path: str, device: torch.device) -> DepthAnythingV2:
    """Loads the DepthAnythingV2 model."""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    print(f"Loading DepthAnythingV2 model with {encoder} encoder...")
    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    depth_model.to(torch.float32)
    return depth_model


def load_spatial_clip_model(repo_id: str, filename: str, device: torch.device) -> nn.Module:
    """Loads the Spatial-CLIP model from Hugging Face Hub."""
    print("Loading Spatial-CLIP model...")
    model, _ = sclip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    
    ckpt_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_ckpt[k.replace("module.", "")] = v
        
    model.load_state_dict(new_ckpt, strict=True)
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def run_retrieval_evaluation(
    clip_model: nn.Module, 
    depth_model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extracts features and computes retrieval metrics."""
    image_features_list = []
    text_features_list = []

    for images, captions_batch in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        depth_maps = depth_model(images)
        
        image_features = clip_model.visual(images, depth_maps, pos_embed="")
        image_features_list.append(image_features.cpu())
        
        captions_batch = captions_batch.to(device)
        b, n, l = captions_batch.shape
        text_features = clip_model.encode_text(captions_batch.view(-1, l))
        text_features_list.append(text_features.cpu())

    all_image_features = torch.cat(image_features_list)
    all_image_features /= all_image_features.norm(dim=-1, keepdim=True)

    all_text_features = torch.cat(text_features_list)
    all_text_features /= all_text_features.norm(dim=-1, keepdim=True)
    
    similarity = all_image_features @ all_text_features.T
    num_images = similarity.shape[0]
    num_captions = similarity.shape[1]

    # Image-to-Text Retrieval
    i2t_accuracies = {}
    for k in [1, 5, 10]:
        correct_preds = 0
        for i in range(num_images):
            topk_preds = similarity[i].topk(k).indices
            if any((i * 5 <= pred < (i + 1) * 5) for pred in topk_preds):
                correct_preds += 1
        i2t_accuracies[f"R@{k}"] = correct_preds / num_images

    # Text-to-Image Retrieval
    t2i_accuracies = {}
    similarity_t = similarity.T
    for k in [1, 5, 10]:
        correct_preds = 0
        for i in range(num_captions):
            topk_preds = similarity_t[i].topk(k).indices
            if i // 5 in topk_preds:
                correct_preds += 1
        t2i_accuracies[f"R@{k}"] = correct_preds / num_captions
        
    return i2t_accuracies, t2i_accuracies


def main():
    """Main execution function."""
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    clip_model = load_spatial_clip_model(args.clip_repo_id, args.clip_filename, device)
    depth_model = load_depth_model(args.depth_model_encoder, args.depth_ckpt_path, device)

    # Prepare data
    clip_preprocess = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    dataset = CocoRetrievalDataset(root=args.data_root, ann_file=args.ann_file, 
                                   transform=clip_preprocess, tokenize=sclip.tokenize)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers, pin_memory=True)

    # Run evaluation
    i2t_acc, t2i_acc = run_retrieval_evaluation(clip_model, depth_model, dataloader, device)

    # Print results
    print("\n--- Retrieval Results ---")
    print("Image-to-Text:")
    for key, val in i2t_acc.items():
        print(f"  {key}: {val:.4f}")
    
    print("Text-to-Image:")
    for key, val in t2i_acc.items():
        print(f"  {key}: {val:.4f}")
    print("-------------------------")


if __name__ == "__main__":
    main()
