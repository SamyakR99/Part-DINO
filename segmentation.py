"""
segmentation.py — Modular foreground segmentation backends.

Supports:
  - SAM3 (facebook/sam3) with text prompts  (default for CUB: "bird")
  - TokenCut (unsupervised, no external model needed)

Usage:
    from segmentation import SAM3Segmenter

    segmenter = SAM3Segmenter(device="cuda", text_prompt="bird")
    mask = segmenter.segment(pil_image)   # returns binary np.array (H, W)
"""

import numpy as np
import torch
from PIL import Image


# ─── Expanded SAM3 prompts for PartImageNet OOD supercategories ───────────
# Each supercategory may contain diverse ImageNet synsets (e.g. "Car" includes
# trolleybuses, jeeps, sports cars).  Listing all plausible object names
# ensures SAM3's text-grounded segmentation does not miss the main subject.
SUPERCAT_PROMPTS = {
    'Aeroplane':  'airplane',
    'Bicycle':    'bicycle',
    'Biped':      'monkey',
    'Bird':       'bird',
    'Boat':       'boat',
    'Bottle':     'bottle',
    'Car':        'vehicle',
    'Fish':       'fish',
    'Quadruped':  'animal',
    'Reptile':    'reptile',
    'Snake':      'snake',
}


def get_sam3_prompt(supercategory):
    """
    Return an expanded SAM3 text prompt for a PartImageNet OOD supercategory.
    Falls back to the supercategory name itself if not in the mapping.
    """
    return SUPERCAT_PROMPTS.get(supercategory, supercategory.lower())


class SAM3Segmenter:
    """
    Text-prompted foreground segmentation using SAM3 (facebook/sam3).
    Requires: pip install git+https://github.com/huggingface/transformers.git accelerate
    Requires: huggingface login with a token that has access to facebook/sam3
    """

    def __init__(self, device="cuda", text_prompt="bird", model_id="facebook/sam3",
                 score_threshold=0.5, hf_token=None):
        """
        Args:
            device: "cuda" or "cpu"
            text_prompt: text prompt for segmentation (e.g., "bird" for CUB)
            model_id: HuggingFace model ID
            score_threshold: minimum confidence to accept a mask
            hf_token: HuggingFace token for gated model access
        """
        from transformers import Sam3Processor, Sam3Model

        self.device = device
        self.text_prompt = text_prompt
        self.score_threshold = score_threshold

        # Login if token provided
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        print(f"Loading SAM3 ({model_id}) on {device}...")
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model.eval()
        print("SAM3 loaded!")

    @torch.no_grad()
    def segment(self, pil_image):
        """
        Segment the foreground object from a PIL image using text prompt.

        Args:
            pil_image: PIL.Image.Image in RGB

        Returns:
            mask: binary numpy array (H, W), True = foreground
        """
        # Preprocess
        inputs = self.processor(
            images=pil_image,
            text=self.text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            target_sizes=inputs["original_sizes"].tolist()
        )[0]

        if len(results["masks"]) == 0:
            # print(f"  [SAM3] No '{self.text_prompt}' detected, falling back to full image")
            h, w = pil_image.size[1], pil_image.size[0]
            return np.ones((h, w), dtype=bool)

        # Take the largest-area mask (most likely to be the full foreground object)
        masks = [m.cpu().numpy().astype(bool) for m in results["masks"]]
        areas = [m.sum() for m in masks]
        best_idx = int(np.argmax(areas))
        mask = masks[best_idx]

        return mask

    def segment_tensor(self, img_tensor):
        """
        Segment from a normalized PyTorch tensor (C, H, W).
        Converts to PIL, runs SAM3, returns binary mask at tensor resolution.

        Args:
            img_tensor: (C, H, W) normalized tensor

        Returns:
            mask: binary numpy array (H, W)
        """
        return self.segment_tensor_with_prompt(img_tensor, self.text_prompt)

    def segment_tensor_with_prompt(self, img_tensor, prompt):
        """
        Like segment_tensor() but with a dynamically specified text prompt.
        Useful when the prompt changes per image (e.g. per-supercategory SAM).

        Args:
            img_tensor: (C, H, W) normalized tensor
            prompt: text prompt string (e.g. 'Bird', 'Car')

        Returns:
            mask: binary numpy array (H, W)
        """
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        pil_image = Image.fromarray((img_np * 255).astype(np.uint8))

        # Temporarily swap the prompt
        old_prompt = self.text_prompt
        self.text_prompt = prompt
        mask = self.segment(pil_image)
        self.text_prompt = old_prompt

        # Resize mask to match tensor resolution if different
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        if mask.shape != (H, W):
            from skimage.transform import resize
            mask = resize(mask.astype(float), (H, W), order=0) > 0.5

        return mask



class TokenCutSegmenter:
    """
    Unsupervised foreground segmentation using TokenCut (from main.py).
    This is the original method — no external model needed.
    """

    def __init__(self):
        # TokenCut is computed inside the pipeline itself, no setup needed
        pass

    def segment(self, vec_2d_up, smoothed_ramp):
        """
        Wrapper around refine_mask from main.py.
        Called internally by the pipeline.
        """
        from main import refine_mask
        return refine_mask(vec_2d_up, smoothed_ramp)
