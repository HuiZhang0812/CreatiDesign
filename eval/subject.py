import os, sys, json, math, argparse, glob
from pathlib import Path
from typing import List
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoProcessor, CLIPModel,
    AutoImageProcessor, AutoModel
)
from datasets import load_dataset

def scale_bbox(bbox, ori_size, target_size):
    x_min, y_min, x_max, y_max = bbox
    ori_width, ori_height = ori_size
    target_width, target_height = target_size
    
    width_ratio = target_width / ori_width
    height_ratio = target_height / ori_height
    
    scaled_x_min = int(x_min * width_ratio)
    scaled_y_min = int(y_min * height_ratio)
    scaled_x_max = int(x_max * width_ratio)
    scaled_y_max = int(y_max * height_ratio)
    
    scaled_x_min = max(0, scaled_x_min)
    scaled_y_min = max(0, scaled_y_min)
    scaled_x_max = min(target_width, scaled_x_max)
    scaled_y_max = min(target_height, scaled_y_max)
    
    return [scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max]

@torch.no_grad()
def encode_clip(imgs: List[Image.Image]) -> torch.Tensor:
    features_list = []
    for img in imgs:
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**inputs)
         
        normalized_features = image_features / image_features.norm(dim=1, keepdim=True)
        features_list.append(normalized_features.squeeze().cpu())
    return torch.stack(features_list)

@torch.no_grad()
def encode_dino(imgs: List[Image.Image]) -> torch.Tensor:
    features_list = []
    for img in imgs:
        inputs = dino_processor(images=img, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
        image_features = outputs.last_hidden_state.mean(dim=1)
        normalized_features = image_features / image_features.norm(dim=1, keepdim=True)
        features_list.append(normalized_features.squeeze().cpu())
    return torch.stack(features_list)

@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a @ b.T).squeeze()

# ------------- Command line arguments -----------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--benchmark_repo", type=str, default="HuiZhang0812/CreatiDesign_benchmark",
                    help="Root directory for one thousand cases")
parser.add_argument("--gen_root", type=str, default="outputs/CreatiDesign_benchmark",
                    help="Root directory for generated images (should have images/<case_id>.jpg underneath)")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--outfile", type=str,
                    help="Path for result CSV; by default written to gen_root")
args = parser.parse_args()

print("handling:", args.gen_root)
if args.outfile is None:
    args.outfile = os.path.join(args.gen_root,"scores.csv")

# Convert outfile to Path object
outfile_path = Path(args.outfile)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ------------- Loading models -------------------
print("[INFO] loading CLIP...")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

print("[INFO] loading DINOv2...")
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dino_model.eval()

benchmark = load_dataset(args.benchmark_repo, split="test")

DEBUG = True
if DEBUG:
    subject_save_roor = os.path.join(args.gen_root,"subject-eval-visual")
    os.makedirs(subject_save_roor,exist_ok=True)
records = []
for case in tqdm(benchmark):
    json_data = json.loads(case["metadata"])
    case_info = json_data["img_info"]
    case_id = case_info["img_id"]

    # ---------- Read reference subjects ----------
    ref_imgs = case['condition_white_variants']
    if len(ref_imgs) == 0:
        print(f"[WARN] {case_id} has no reference subject, skipping")
        continue

    # ---------- Read generated image ----------
    gen_path = os.path.join(args.gen_root, "images", f"{case_id}.jpg")
    gen_img = Image.open(gen_path).convert("RGB")
    # Get width and height of generated image
    gen_width, gen_height = gen_img.size
    reg_bbox_id = [item["bbox_idx"] for item in sorted(json_data["subject_annotations"], key=lambda x: x["bbox_idx"])]
    ref_bbox = [item["bbox"] for item in sorted(json_data["subject_annotations"], key=lambda x: x["bbox_idx"])]
    ori_width,ori_height = json_data["img_info"]["img_width"],json_data["img_info"]["img_height"]
    # Extract corresponding images from the generated image
    gen_imgs = []
    for bbox in ref_bbox:
        # Scale the bounding box
        scaled_bbox = scale_bbox(
            bbox, 
            (ori_width, ori_height), 
            (gen_width, gen_height)
        )
        
        # Crop the image area
        x_min, y_min, x_max, y_max = scaled_bbox
        cropped_img = gen_img.crop((x_min, y_min, x_max, y_max))
        gen_imgs.append(cropped_img)
    if DEBUG:
        folder_root = os.path.join(subject_save_roor,case_id)
        os.makedirs(folder_root,exist_ok=True)
        # Save cropped images
        for i, (img, img_id) in enumerate(zip(gen_imgs, reg_bbox_id)):
            img.save(os.path.join(folder_root, f"{img_id}.png"))
            
            
    # ---------- Features ----------
    ref_clip = encode_clip(ref_imgs)      # (n,dim)
    gen_clip = encode_clip(gen_imgs)      # (n,dim)

    ref_dino = encode_dino(ref_imgs)      # (n,dim)
    gen_dino = encode_dino(gen_imgs)      # (n,dim)

    # ---------- Similarity ----------
    clip_sims = torch.nn.functional.cosine_similarity(ref_clip, gen_clip)      
    dino_sims = torch.nn.functional.cosine_similarity(ref_dino, gen_dino)       

    clip_i   = clip_sims.mean().item()
    dino_avg = dino_sims.mean().item()
    m_dino   = dino_sims.prod().item()            

    records.append(dict(
        case_id=case_id,
        num_subject=len(ref_imgs),
        clip_i=clip_i,
        dino=dino_avg,
        m_dino=m_dino
    ))

# ---------------- Result statistics -----------------
df = pd.DataFrame(records).sort_values("case_id")
overall = df[["clip_i","dino","m_dino"]].mean().to_dict()

print("\n========== Overall Average ==========")
for k,v in overall.items():
    print(f"{k:>8}: {v:.6f}")
print("=====================================\n")

# Group by number of subjects
df_by_subjects = {}
avg_by_subjects = {}

# Create subset for each subject count (1-5)
for i in range(1, 6):
    # Filter records with subject count = i
    subset = df[df["num_subject"] == i]
    
    if len(subset) > 0:
        # Calculate average for this group
        subset_avg = subset[["clip_i", "dino", "m_dino"]].mean().to_dict()
        avg_by_subjects[i] = subset_avg
        
        # Create subset with average row
        avg_row = {"case_id": f"average_subject_{i}", "num_subject": i}
        avg_row.update(subset_avg)
        
        # Add average row to subset
        subset_with_avg = pd.concat([subset, pd.DataFrame([avg_row])], ignore_index=True)
        df_by_subjects[i] = subset_with_avg
        
        # Print average for this group
        print(f"\n=== Subject {i} Average (n={len(subset)}) ===")
        for k, v in subset_avg.items():
            print(f"{k:>8}: {v:.6f}")
        
        # Save subset - fixed path handling
        subject_path = outfile_path.parent / f"{outfile_path.stem}_subject{i}_location_prior{outfile_path.suffix}"
        subset_with_avg.to_csv(subject_path, index=False, float_format="%.6f")
        print(f"[INFO] Subject {i} results written to {subject_path}")

# Save overall average to CSV - fixed path handling
overall_df = pd.DataFrame([overall], index=["overall"])
overall_path = outfile_path.parent / f"{outfile_path.stem}_overall_location_prior{outfile_path.suffix}"
overall_df.to_csv(overall_path, float_format="%.6f")
print(f"[INFO] Overall results written to {overall_path}")

# Write CSV
df.to_csv(args.outfile, index=False, float_format="%.6f")
print(f"[INFO] Written to {args.outfile}")

# Create statistics table with averages for all groups
if avg_by_subjects:
    # Merge averages for each group into one table
    stats_rows = []
    for num_subject, avg_dict in avg_by_subjects.items():
        row = {"num_subject": num_subject}
        row.update(avg_dict)
        stats_rows.append(row)
    
    # Add overall average
    overall_row = {"num_subject": "all"}
    overall_row.update(overall)
    stats_rows.append(overall_row)
    
    # Create summary statistics table
    stats_df = pd.DataFrame(stats_rows)
    # Fixed path handling
    stats_path = outfile_path.parent / f"{outfile_path.stem}_stats_location_prior{outfile_path.suffix}"
    stats_df.to_csv(stats_path, index=False, float_format="%.6f")
    print(f"[INFO] All group statistics written to {stats_path}")
