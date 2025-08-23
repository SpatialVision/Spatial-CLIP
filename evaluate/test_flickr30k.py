import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add local package paths
sys.path.append(os.path.abspath(".."))
sys.path.append('../Depth-Anything-V2')
import spat_clip as sclip
from depth_anything_v2.dpt import DepthAnythingV2


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Spatial-CLIP Retrieval Evaluation on Flickr30k")
    
    parser.add_argument('--flickr-root', type=str, default='../dataset/flickr30k/flickr30k-images',
                        help='Root directory of the Flickr30k image dataset.')
    parser.add_argument('--token-file', type=str, default='../dataset/flickr30k/results_20130124.token',
                        help='Path to the Flickr30k token file.')

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
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Batch size for evaluation.')
    parser.add_argument('-w', '--num-workers', type=int, default=4,
                        help='Number of data loader workers.')
                        
    return parser.parse_args()


class Flickr30kDataset(Dataset):
    """
    Flickr30k Dataset for image-text retrieval.
    Parses the standard token file and pairs images with their 5 captions.
    """
    def __init__(self, root_dir: str, token_file: str, transform, tokenize):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenize = tokenize
        self.image_paths, self.captions = self._parse_token_file(token_file)

    def _parse_token_file(self, token_file: str) -> Tuple[List[str], List[List[str]]]:
        """Reads the token file and organizes data into images and their captions."""
        with open(token_file, 'r') as f:
            lines = f.readlines()
        
        image_paths = []
        captions_per_image = []
        
        # Group by 5, as each image has 5 captions
        for i in range(0, len(lines), 5):
            image_filename = lines[i].split('\t')[0][:-2] # Remove #0, #1, etc.
            image_paths.append(os.path.join(self.root_dir, image_filename))
            
            image_captions = [line.split('\t')[1].strip() for line in lines[i:i+5]]
            captions_per_image.append(image_captions)
            
        return image_paths, captions_per_image

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        tokenized_captions = self.tokenize(self.captions[idx], truncate=True)
        
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
    
    new_ckpt = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
        
    model.load_state_dict(new_ckpt, strict=True)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters.")
    
    return model


@torch.no_grad()
def extract_features(
    clip_model: nn.Module, 
    depth_model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts and normalizes image and text features from the dataloader."""
    image_features_list = []
    text_features_list = []

    for images, captions_batch in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        captions_batch = captions_batch.to(device)
        
        # Image feature extraction
        depth_maps = depth_model(images)
        image_features = clip_model.visual(images, depth_maps, pos_embed="")
        image_features_list.append(image_features.cpu())
        
        # Text feature extraction (captions_batch is [B, 5, 77])
        b, n_captions, seq_len = captions_batch.shape
        flat_captions = captions_batch.view(-1, seq_len)
        text_features = clip_model.encode_text(flat_captions)
        text_features_list.append(text_features.cpu())
    
    # Concatenate and normalize
    all_image_features = torch.cat(image_features_list)
    all_image_features /= all_image_features.norm(dim=-1, keepdim=True)

    all_text_features = torch.cat(text_features_list)
    all_text_features /= all_text_features.norm(dim=-1, keepdim=True)
    
    return all_image_features, all_text_features


def calculate_retrieval_metrics(
    image_features: torch.Tensor, 
    text_features: torch.Tensor
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculates and returns I2T and T2I retrieval accuracies."""
    metrics = {}
    
    similarity = image_features.cuda() @ text_features.cuda().T
    num_images = similarity.shape[0]
    
    # Image-to-Text Retrieval
    i2t_accuracies = {}
    for k in [1, 5, 10]:
        correct_preds = 0
        topk_preds = similarity.topk(k, dim=1).indices
        for i in range(num_images):
            ground_truth_range = torch.arange(i * 5, (i + 1) * 5, device=topk_preds.device)
            if any(pred in ground_truth_range for pred in topk_preds[i]):
                correct_preds += 1
        i2t_accuracies[f"R@{k}"] = correct_preds / num_images
    metrics["I2T"] = i2t_accuracies

    # Text-to-Image Retrieval
    t2i_accuracies = {}
    similarity_t = similarity.T
    num_captions = similarity_t.shape[0]
    for k in [1, 5, 10]:
        correct_preds = 0
        topk_preds = similarity_t.topk(k, dim=1).indices
        for i in range(num_captions):
            ground_truth = i // 5
            if ground_truth in topk_preds[i]:
                correct_preds += 1
        t2i_accuracies[f"R@{k}"] = correct_preds / num_captions
    metrics["T2I"] = t2i_accuracies
    
    return metrics


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
    
    dataset = Flickr30kDataset(root_dir=args.flickr_root, token_file=args.token_file,
                              transform=clip_preprocess, tokenize=sclip.tokenize)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers, pin_memory=True)

    # Run evaluation
    image_features, text_features = extract_features(clip_model, depth_model, dataloader, device)
    metrics = calculate_retrieval_metrics(image_features, text_features)

    # Print results
    print("\n--- Flickr30k Retrieval Results ---")
    for task, accuracies in metrics.items():
        print(f"{task}:")
        for key, val in accuracies.items():
            print(f"  {key}: {val:.4f}")
    print("---------------------------------")

if __name__ == "__main__":
    main()
