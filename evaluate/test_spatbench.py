import argparse
import json
import os
import sys
from collections import defaultdict, OrderedDict
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
sys.path.append('../Depth-Anything-V2')
import spat_clip as sclip
from depth_anything_v2.dpt import DepthAnythingV2


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Spatial-CLIP Zero-Shot Evaluation on RGB-D Benchmarks")
    
    parser.add_argument('--outdoor-root', type=str, default='../dataset/spatial-benchmark/outdoors',
                        help='Root directory of the outdoor benchmark dataset.')
    parser.add_argument('--indoor-root', type=str, default='../dataset/spatial-benchmark/indoors',
                        help='Root directory of the indoor benchmark dataset.')

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


class SpatialBenchmarkDataset(Dataset):
    """
    A flexible dataset for spatial reasoning benchmarks (indoor/outdoor).
    Handles different annotation formats and path constructions.
    """
    def __init__(self, root_dir: str, ann_file: str, transform: Callable, path_builder: Callable):
        self.root_dir = root_dir
        self.transform = transform
        self.path_builder = path_builder
        
        self.annotations = self._load_annotations(os.path.join(root_dir, ann_file))
        self.image_ids = list(self.annotations.keys())
        # Map original image_ids to a continuous index for DataLoader
        self.idx_to_id = {i: img_id for i, img_id in enumerate(self.image_ids)}

    def _load_annotations(self, file_path: str) -> Dict:
        """Loads and parses the JSON annotation file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {item['id']: item for item in data}

    def get_captions(self) -> List[List[str]]:
        """Returns all caption sets for zeroshot classifier construction."""
        return [self.annotations[img_id]['captions'] for img_id in self.image_ids]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_id = self.idx_to_id[idx]
        item_data = self.annotations[img_id]
        
        image_path = self.path_builder(self.root_dir, item_data)
        image = Image.open(image_path).convert('RGB')
        
        image_tensor = self.transform(image)
        answer = item_data['answer']
        
        # Return the index to map image features to text features
        return image_tensor, answer, idx


def build_outdoor_path(root_dir: str, item_data: Dict) -> str:
    """Constructs the file path for an outdoor dataset item."""
    return os.path.join(root_dir, 'pic_all', item_data['image'])


def build_indoor_path(root_dir: str, item_data: Dict) -> str:
    """Constructs the file path for an indoor dataset item."""
    return os.path.join(root_dir, 'pic_all', item_data['image'])


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
def build_zeroshot_classifier(captions: List[List[str]], model: nn.Module, device: torch.device) -> torch.Tensor:
    """Encodes text captions to create zeroshot classifier weights."""
    all_embeddings = []
    for caption_set in tqdm(captions, desc="Encoding text prompts"):
        tokens = sclip.tokenize(caption_set).to(device)
        class_embeddings = model.encode_text(tokens)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(class_embeddings)
    return torch.stack(all_embeddings)


@torch.no_grad()
def run_zeroshot_epoch(
    clip_model: nn.Module, 
    depth_model: nn.Module, 
    dataloader: DataLoader, 
    text_embeddings: torch.Tensor, 
    device: torch.device
) -> float:
    """Runs one epoch of zeroshot evaluation and returns accuracy."""
    correct_preds, total_samples = 0, 0
    for images, answers, indices in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        answers = answers.to(device)
        indices = indices.to(device)
        
        depth_maps = depth_model(images)
        image_features = clip_model.visual(images, depth_maps, pos_embed="")
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Select the text embeddings corresponding to the batch items
        relevant_text_embeddings = text_embeddings[indices]
        
        # Calculate scores by element-wise multiplication and sum
        scores = (image_features.unsqueeze(1) * relevant_text_embeddings).sum(dim=-1)
        
        preds = scores.argmax(dim=1)
        correct_preds += (preds == answers).sum().item()
        total_samples += images.size(0)
        
    return correct_preds / total_samples if total_samples > 0 else 0.0


def main():
    """Main execution function."""
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    clip_model = load_spatial_clip_model(args.clip_repo_id, args.clip_filename, device)
    depth_model = load_depth_model(args.depth_model_encoder, args.depth_ckpt_path, device)

    # Define image transform
    clip_preprocess = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((336, 336), interpolation=Image.BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Evaluate Outdoor Benchmark
    print("\n--- Evaluating Outdoor Benchmark ---")
    outdoor_dataset = SpatialBenchmarkDataset(
        root_dir=args.outdoor_root, 
        ann_file='all_data.json',
        transform=clip_preprocess,
        path_builder=build_outdoor_path
    )
    outdoor_text_embeddings = build_zeroshot_classifier(outdoor_dataset.get_captions(), clip_model, device)
    outdoor_loader = DataLoader(outdoor_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    outdoor_accuracy = run_zeroshot_epoch(clip_model, depth_model, outdoor_loader, outdoor_text_embeddings, device)
    print(f"Outdoor Benchmark Accuracy: {outdoor_accuracy:.4f}")

    # Evaluate Indoor Benchmark
    print("\n--- Evaluating Indoor Benchmark ---")
    indoor_dataset = SpatialBenchmarkDataset(
        root_dir=args.indoor_root,
        ann_file='all_data.json',
        transform=clip_preprocess,
        path_builder=build_indoor_path
    )
    indoor_text_embeddings = build_zeroshot_classifier(indoor_dataset.get_captions(), clip_model, device)
    indoor_loader = DataLoader(indoor_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    indoor_accuracy = run_zeroshot_epoch(clip_model, depth_model, indoor_loader, indoor_text_embeddings, device)
    print(f"Indoor Benchmark Accuracy: {indoor_accuracy:.4f}")
    print("------------------------------------")


if __name__ == "__main__":
    main()
