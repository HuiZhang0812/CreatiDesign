import os, json, csv, re, cv2, numpy as np, torch
from tqdm import tqdm
from editdistance import eval as edit_distance
from paddleocr import PaddleOCR
from datasets import load_dataset
# -------------------------------------------------------------------
# Paths
benchmark_repo = 'HuiZhang0812/CreatiDesign_benchmark' #  huggingface repo of benchmark
benchmark = load_dataset(benchmark_repo, split="test")
root_gen = "outputs/CreatiDesign_benchmark/images"

save_root = root_gen.replace("images", "text_eval")  # Output directory
os.makedirs(save_root, exist_ok=True)
DEBUG = True
# -------------------------------------------------------------------
# 1. OCR initialization (must be det=True)
ocr = PaddleOCR(det=True, rec=True, cls=False, use_angle_cls=False, lang='en')

# -------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# 3. Utility functions

def spatial_match_iou(det_res, gt_box, gt_text_fmt, iou_thr=0.5):
    best_iou = 0.0
    if det_res is None or len(det_res) == 0:
        return best_iou
        
    for item in det_res:
        poly = item[0]  # Detection box coordinates
        txt_info = item[1]  # Text information tuple
        txt = txt_info[0]   # Text content

        if min_ned_substring(normalize_text(txt), gt_text_fmt) <= 0.7: # When calculating spatial, allow some degree of text error
            iou_val = iou(quad2bbox(poly), gt_box)
            best_iou = max(best_iou, iou_val)
    return best_iou

# ① New tool: Minimum NED substring
def min_ned_substring(pred_fmt: str, tgt_fmt: str) -> float:
    """
    Find a substring in pred_fmt with the same length as tgt_fmt, to minimize normalized edit distance
    Return the minimum value (0 ~ 1)
    """
    Lp, Lg = len(pred_fmt), len(tgt_fmt)
    if Lg == 0:
        return 0.0
    if Lp < Lg:           # If prediction string is shorter than target, calculate directly
        return normalized_edit_distance(pred_fmt, tgt_fmt)

    best = Lg            # Maximum possible distance
    for i in range(Lp - Lg + 1):
        sub = pred_fmt[i:i+Lg]
        d   = edit_distance(sub, tgt_fmt)
        if d < best:
            best = d
            if best == 0:                 # Early exit
                break
    return best / Lg                      # Normalize

def normalize_text(txt: str) -> str:
    txt = txt.lower().replace(" ", "")
    return re.sub(r"[^\w\s]", "", txt)

def normalized_edit_distance(pred: str, gt: str) -> float:
    if not gt and not pred:
        return 0.0
    return edit_distance(pred, gt) / max(len(gt), len(pred))

def iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

def quad2bbox(quad):
    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]

def crop(img, box):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,1,3), np.uint8)
    return img[y1:y2, x1:x2]


# -------------------------------------------------------------------
# 4. Main loop
per_img_rows, all_sen_acc, all_ned, all_spatial, text_pairs = [], [], [], [], []

for case in tqdm(benchmark):
    json_data = json.loads(case["metadata"])
    case_info = json_data["img_info"]
    case_id = case_info["img_id"]

    gt_list = json_data["text_list"]          # [{'text':..., 'bbox':[x1,y1,x2,y2]}, ...]
    ori_w, ori_h = json_data["img_info"]["img_width"], json_data["img_info"]["img_height"]

    img_path = os.path.join(root_gen, f"{case_id}.jpg")
    
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    wr, hr = W / ori_w, H / ori_h        # GT → Generated image scaling ratio

    # ---------- 1) Full image OCR ----------
    pred_lines = []        # Save OCR line text
    ocr_res = ocr.ocr(img, cls=False)
    if ocr_res and ocr_res[0]:
        for quad, (txt, conf) in ocr_res[0]:
            pred_lines.append(txt.strip())

    # Concatenate into full text and normalize
    pred_full_fmt = normalize_text(" ".join(pred_lines))

    # ==========================================================
    # ③ For each GT sentence, do "substring minimum NED" ---- no longer using IoU
    img_sen_hits, img_neds, img_spatials = [], [], []

    for t_idx, gt in enumerate(gt_list):
        gt_text_orig = gt["text"].replace("\n", " ").strip()
        gt_text_fmt  = normalize_text(gt_text_orig)

        # ---- Pure text matching ----
        ned  = min_ned_substring(pred_full_fmt, gt_text_fmt)
        acc  = 1.0 if ned == 0 else 0.0
        img_sen_hits.append(acc)
        img_neds.append(ned)

        # ---------- Spatial consistency, using IOU ----------
        gt_box = [v*wr if i%2==0 else v*hr for i,v in enumerate(gt["bbox"])]
        det_res = ocr_res[0] if ocr_res else [] 
        spatial_score = spatial_match_iou(det_res, gt_box, gt_text_fmt)
        img_spatials.append(spatial_score)   # Can be used directly or binarized
        crop_box_int = list(map(int, gt_box))
        img_crop = crop(img, crop_box_int)
        if DEBUG:
            # Save cropped image
            img_crop_for_ocr_save_root = os.path.join(save_root, case_id)
            os.makedirs(img_crop_for_ocr_save_root, exist_ok=True)
            safe_text = gt_text_orig.replace('/', '_').replace('\\', '_')
            safe_filename = f"{t_idx}_{safe_text}.jpg"
            cv2.imwrite(os.path.join(img_crop_for_ocr_save_root, safe_filename), img_crop)

        # --------- Record text pairs ----------
        text_pairs.append({
            "image_id"       : case_id,
            "text_id"        : t_idx,
            "gt_original"    : gt_text_orig,
            "gt_formatted"   : gt_text_fmt
        })

    # ---------- 3) Summarize to image level ----------
    sen_acc  = float(np.mean(img_sen_hits))
    ned      = float(np.mean(img_neds))
    spatial  = float(np.mean(img_spatials))

    per_img_rows.append([case_id, sen_acc, ned, spatial])
    all_sen_acc.append(sen_acc)
    all_ned.append(ned)
    all_spatial.append(spatial)

# -------------------------------------------------------------------
# 5. Write results
result_root = root_gen.replace("images","")
csv_perimg = os.path.join(result_root, "text_results_per_image.csv")
with open(csv_perimg, "w", newline='', encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["image_id","sen_acc","ned","score_spatial"]); w.writerows(per_img_rows)


with open(os.path.join(result_root, "text_overall.txt"), "w", encoding="utf-8") as f:
    f.write(f"Images evaluated : {len(per_img_rows)}\n")
    f.write(f"Global Sen ACC   : {np.mean(all_sen_acc):.4f}\n")
    f.write(f"Global NED       : {np.mean(all_ned):.4f}\n")
    f.write(f"Global Spatial   : {np.mean(all_spatial):.4f}\n")

print("✓ Done! Results saved to", result_root)
