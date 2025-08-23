"""
CLIP: Connecting Text and Images

This module provides functions to load CLIP models, tokenize text, and preprocess images.
It is based on the original implementation from OpenAI.

For more details, please refer to the paper: https://arxiv.org/abs/2103.00020
"""

import hashlib
import os
import urllib.request
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable

import torch
from PIL import Image
from pkg_resources import packaging
from torch.nn import Module
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

Device = Union[str, torch.device]
PreprocessFn = Callable[[Image.Image], torch.Tensor]

_IMAGE_NET_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGE_NET_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended for optimal performance.")

__all__ = ["available_models", "load", "tokenize"]

_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": ("https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt", "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"),
    "RN101": ("https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt", "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599"),
    "RN50x4": ("https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt", "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd"),
    "RN50x16": ("https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt", "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa"),
    "RN50x64": ("https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt", "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c"),
    "ViT-B/32": ("https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt", "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"),
    "ViT-B/16": ("https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt", "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"),
    "ViT-L/14": ("https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
    "ViT-L/14@336px": ("https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt", "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"),
}

def _sha256(path: Path) -> str:
    """
    Computes the SHA256 checksum of a file in a memory-efficient way.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The hex digest of the SHA256 checksum.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, root: Path, expected_sha256: str) -> Path:
    """
    Downloads a file from a URL to a specified directory, verifying its checksum.

    Args:
        url (str): The URL to download from.
        root (Path): The directory to save the file in.
        expected_sha256 (str): The expected SHA256 checksum of the file.

    Returns:
        Path: The path to the downloaded file.
    
    Raises:
        RuntimeError: If the downloaded file's checksum does not match.
    """
    root.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    download_target = root / filename

    if download_target.exists():
        if _sha256(download_target) == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file.")
            download_target.unlink() 

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length", 0)), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if _sha256(download_target) != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not match. Expected {expected_sha256}.")

    return download_target


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    """Converts a PIL image to RGB format."""
    return image.convert("RGB")


def _transform(n_px: int) -> PreprocessFn:
    """
    Creates a torchvision transform pipeline for a CLIP model.

    Args:
        n_px (int): The input resolution of the model.

    Returns:
        Callable[[Image.Image], torch.Tensor]: A function that preprocesses a PIL image.
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(_IMAGE_NET_MEAN, _IMAGE_NET_STD),
    ])


def available_models() -> List[str]:
    """
    Returns the names of available CLIP models.
    
    Example:
        >>> clip.available_models()
        ['RN50', 'RN101', 'RN50x4', ...]
    """
    return list(_MODELS.keys())


def load(
    name: str,
    device: Device = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: Optional[Union[str, Path]] = None,
    lora_adapt: bool = False,
    rank: int = 16,
    vision_ckpt_pth: Optional[Union[str, Path]] = None,
) -> Tuple[Module, PreprocessFn]:
    """
    Loads a CLIP model and its corresponding image preprocessing function.

    Args:
        name (str): A model name from `clip.available_models()` or a path to a model checkpoint file.
        device (Union[str, torch.device]): The device to load the model onto.
        jit (bool): If True, loads the TorchScript JIT model for potential optimization. Defaults to False.
        download_root (Optional[Union[str, Path]]): Path to download the model files. Defaults to "~/.cache/clip".
        lora_adapt (bool): If True, adapts the model with LoRA layers (for non-JIT models). Defaults to False.
        rank (int): The rank for LoRA adaptation. Defaults to 16.
        vision_ckpt_pth (Optional[Union[str, Path]]): Path to a checkpoint for the vision part of the model,
            typically used for inference with fine-tuned models.

    Returns:
        Tuple[torch.nn.Module, Callable[[PIL.Image], torch.Tensor]]: A tuple containing:
            - The loaded CLIP model.
            - A torchvision transform for preprocessing images.

    Raises:
        RuntimeError: If the specified model name is not available and not a valid file path.
        FileNotFoundError: If a specified checkpoint path does not exist.
    """
    if name in _MODELS:
        url, sha256 = _MODELS[name]
        if download_root is None:
            download_root = Path(os.path.expanduser("~/.cache/clip"))
        else:
            download_root = Path(download_root)
        
        model_path = _download(url, download_root, sha256)
    elif Path(name).is_file():
        model_path = Path(name)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # --- 模型加载 ---
    try:
        with open(model_path, 'rb') as f:
            model = torch.jit.load(f, map_location=device if jit else "cpu").eval()
            state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead.")
            jit = False
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location="cpu")
    
    if not jit:
        model_state = state_dict or model.state_dict()
        model = build_model(model_state, lora_adapt=lora_adapt, rank=rank).to(device)
        
        if str(device) == "cpu":
            model.float()

        if vision_ckpt_pth is not None:
            vision_ckpt_path = Path(vision_ckpt_pth)
            if not vision_ckpt_path.is_file():
                raise FileNotFoundError(f"Vision checkpoint not found at: {vision_ckpt_path}")
            
            model.visual.load_state_dict(torch.load(vision_ckpt_path, map_location=device))
            model.eval()  

        preprocess = _transform(model.visual.input_resolution)
        return model, preprocess

    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a JIT node which is polymorphic over return type."""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for g in graphs:
            for n in g.findAllNodes("prim::Constant"):
                if "value" in n.attributeNames() and str(_node_get(n, "value")).startswith("cuda"):
                    n.copyAttributes(device_node)
    
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for g in graphs:
                for n in g.findAllNodes("aten::to"):
                    inputs = list(n.inputs())
                    # dtype can be the second or third argument
                    if len(inputs) > 1 and _node_get(inputs[1].node(), "value") == 5: # 5 is fp16
                        inputs[1].node().copyAttributes(float_node)
        
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    
    input_resolution = model.input_resolution.item()
    preprocess = _transform(input_resolution)
    return model, preprocess


def tokenize(
    texts: Union[str, List[str]], 
    context_length: int = 77, 
    truncate: bool = True
) -> torch.Tensor:
    """
    Returns the tokenized representation of given input string(s).

    Args:
        texts (Union[str, List[str]]): An input string or a list of strings to tokenize.
        context_length (int): The context length to use. All CLIP models use 77.
        truncate (bool): If True, truncates the text if its encoding is longer than the
            context length. If False, an error will be raised.

    Returns:
        torch.Tensor: A 2D tensor containing the token IDs, shape = [num_texts, context_length].
            The dtype is torch.long for compatibility with older PyTorch versions.
    
    Raises:
        RuntimeError: If a text is too long and `truncate` is set to False.
        
    Example:
        >>> tokens = clip.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> tokens.shape
        torch.Size([2, 77])
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    
    all_tokens = []
    for text in texts:
        tokens = [sot_token] + _tokenizer.encode(text) + [eot_token]
        all_tokens.append(tokens)

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token  #确保最后一个是 EOT token
            else:
                raise RuntimeError(
                    f"Input text '{texts[i]}' is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

