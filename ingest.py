import os

# MUST come before any ML library import so weights land on D:, not C:
os.environ['HF_HOME'] = 'D:/AI_Models_Cache'
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
import sys
from contextlib import contextmanager
from unittest.mock import patch

import chromadb
import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

# ----------------------------- Configuration -----------------------------
VIDEO_PATH = "Sample_video.mp4"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "video_frames"
MODEL_ID = "microsoft/Florence-2-base"
TASK_PROMPT = "<DETAILED_CAPTION>"
FRAME_INTERVAL_SEC = 2
# -------------------------------------------------------------------------


def _get_imports_no_flash_attn(filename):
    """Florence-2 remote code lists flash_attn as a hard import; it's optional
    and not installable on most Windows setups, so strip it."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_model():
    """Load Florence-2-base in FP16 on GPU (fits comfortably in 4GB VRAM)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: CUDA not available, falling back to CPU (slow).")

    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    with patch("transformers.dynamic_module_utils.get_imports", _get_imports_no_flash_attn):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor, device, dtype


@torch.inference_mode()
def caption_image(image, model, processor, device, dtype):
    """Generate a detailed caption for a single PIL image."""
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    inputs = {
        k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
        do_sample=False,
    )
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_text, task=TASK_PROMPT, image_size=(image.width, image.height)
    )
    return parsed[TASK_PROMPT].strip()


def extract_frames(video_path, interval_sec):
    """Yield (timestamp_seconds, PIL.Image) every `interval_sec` seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * interval_sec)))
    print(f"Video: {fps:.2f} FPS, {total_frames} frames, "
          f"~{total_frames / fps:.0f}s. Sampling every {step} frames.")

    try:
        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield int(round(frame_idx / fps)), Image.fromarray(rgb)
    finally:
        cap.release()


def main():
    if not os.path.isfile(VIDEO_PATH):
        sys.exit(f"ERROR: '{VIDEO_PATH}' not found. Place it in the project folder.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Allow resuming: skip timestamps that were already captioned
    existing_ids = set(collection.get(include=[])["ids"])

    try:
        model, processor, device, dtype = load_model()
    except Exception as e:
        sys.exit(f"ERROR: failed to load model: {e}")

    processed = skipped = 0
    try:
        for timestamp, image in extract_frames(VIDEO_PATH, FRAME_INTERVAL_SEC):
            doc_id = f"{os.path.basename(VIDEO_PATH)}_{timestamp}"
            if doc_id in existing_ids:
                skipped += 1
                continue

            try:
                caption = caption_image(image, model, processor, device, dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{timestamp:>5}s] CUDA OOM, skipping frame.")
                continue

            collection.upsert(
                ids=[doc_id],
                documents=[caption],
                metadatas=[{"timestamp": timestamp}],
            )
            processed += 1
            print(f"[{timestamp:>5}s] {caption}")
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run to resume.")

    print(f"\nDone. New: {processed}, skipped: {skipped}, "
          f"total in DB: {collection.count()}")


if __name__ == "__main__":
    main()
