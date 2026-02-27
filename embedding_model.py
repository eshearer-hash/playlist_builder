import torch
from PIL import Image
from transformers import SiglipProcessor, AutoModel
import io

MODEL_ID = "google/siglip2-base-patch16-224"

# Load once, reuse across calls
model = AutoModel.from_pretrained(MODEL_ID).eval()
processor = SiglipProcessor.from_pretrained(MODEL_ID)


def create_text_embeddings(texts: list[str], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate normalized text embeddings from an array of text using SigLIP.

    Args:
        texts: List of strings to embed.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        Tensor of shape (len(texts), embedding_dim) with L2-normalized embeddings.
    """
    model.to(device)

    inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.get_text_features(**inputs)
        text_embeddings = output.pooler_output if hasattr(output, 'pooler_output') else output

    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


def create_images_embeddings(images: list[bytes | Image.Image], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate normalized image embeddings from a list of images using SigLIP.

    Args:
        images: List of raw image bytes or PIL Images.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        Tensor of shape (len(images), embedding_dim) with L2-normalized embeddings.
    """
    model.to(device)

    if not isinstance(images, list):
        images = [images]

    # Convert bytes to PIL Images if needed
    pil_images = [
        Image.open(io.BytesIO(img)).convert("RGB") if isinstance(img, bytes) else img.convert("RGB")
        for img in images
    ]

    inputs = processor(
        images=pil_images,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.get_image_features(**inputs)
        image_embeddings = output.pooler_output if hasattr(output, 'pooler_output') else output

    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings