import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset
if __name__ == "__main__":
    model_id ="openbmb/MiniCPM-V-2_6"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # evaluation
    benchmark_repo = 'HuiZhang0812/CreatiDesign_benchmark' #  huggingface repo of benchmark
    benchmark = load_dataset(benchmark_repo, split="test")
    gen_root =  "outputs/CreatiDesign_benchmark/images"
    print("processing:",gen_root)
    save_json_path = gen_root.replace("images", "minicpm-vqa.json")
    temp_root = gen_root.replace("images", "images-perarea")
    os.makedirs(temp_root, exist_ok=True)

    skipped_files_log = gen_root.replace("images", "skipped_files.log")
    skipped_files = []
    image_stats = {}
    
    for case in tqdm(benchmark):
        json_data = json.loads(case["metadata"])
        case_info = json_data["img_info"]
        case_id = case_info["img_id"]
        file_name = f"{case_id}.jpg"
        generated_img_path = os.path.join(gen_root, file_name)
        global_caption = json_data["global_caption"]
        object_annotations = json_data["object_annotations"]
        detial_region_caption_list =  [item["bbox_detail_description"] for item in object_annotations]
        region_caption_list = [item["class_name"] for item in object_annotations]
        region_bboxes_list = [item["bbox"] for item in object_annotations]
        
        img = Image.open(generated_img_path).convert("RGB")
        width, height = img.size

        orignal_img_width = json_data["img_info"]["img_width"]
        orignal_img_height = json_data["img_info"]["img_height"]

        temp_save_root = os.path.join(temp_root, file_name.split('.')[0])
        os.makedirs(temp_save_root, exist_ok=True)

        bbox_count = len(region_caption_list)

        # Initialize scores
        img_score_spatial = 0
        img_score_color = 0
        img_score_texture = 0
        img_score_shape = 0
        for i, (bbox,detial_region_caption,region_caption) in enumerate(zip(region_bboxes_list,detial_region_caption_list,region_caption_list)):
            x1, y1, x2, y2= bbox
            x1 = int(x1 / orignal_img_width*width)
            y1 = int(y1 / orignal_img_height*height)    
            x2 = int(x2 / orignal_img_width*width)
            y2 = int(y2 / orignal_img_height*height)

    
            cropped_img = img.crop((x1, y1, x2, y2))

            # save crop img
            description = region_caption.replace('/', '')
            detail_description = detial_region_caption.replace('/', '')
            cropped_img_path = os.path.join(temp_save_root, f'{description}.jpg')
            cropped_img.save(cropped_img_path)

            # spatial
            question = f'Is the subject "{description}" present in the image? Strictly answer with "Yes" or "No", without any irrelevant words.'
            
            msgs = [{'role': 'user', 'content': [cropped_img, question]}]

            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                seed=42
                )
            
            if "Yes" in res or "yes" in res:
                score_spatial = 1.0
            else:
                score_spatial = 0.0

            score_color, score_texture,score_shape = 0.0, 0.0, 0.0
            # attribute
            if score_spatial==1.0:
                #color
                question_color = f'Is the subject in "{description}" in the image consistent with the color described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the color is not mentioned in the detailed description, the answer is "Yes".'
                msgs_color = [{'role': 'user', 'content': [cropped_img, question_color]}]

                color_attribute = model.chat(
                image=None,
                msgs=msgs_color,
                tokenizer=tokenizer,
                seed=42
                )
                
                if "Yes" in color_attribute or "yes" in color_attribute:
                    score_color = 1.0
            # texture
            if score_spatial==1.0:
                question_texture = f'Is the subject in "{description}" in the image consistent with the texture described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the texture is not mentioned in the detailed description, the answer is "Yes".'
                msgs_texture = [{'role': 'user', 'content': [cropped_img, question_texture]}]

                texture_attribute = model.chat(
                image=None,
                msgs=msgs_texture,
                tokenizer=tokenizer,
                seed=42
                )
                if "Yes" in texture_attribute or "yes" in texture_attribute:
                    score_texture = 1.0
            #shape
            if score_spatial==1.0:
                question_shape = f'Is the subject in "{description}" in the image consistent with the shape described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the shape is not mentioned in the detailed description, the answer is "Yes".'
                msgs_shape = [{'role': 'user', 'content': [cropped_img, question_shape]}]

                shape_attribute = model.chat(
                image=None,
                msgs=msgs_shape,
                tokenizer=tokenizer,
                seed=42
                )
                
                if "Yes" in shape_attribute or "yes" in shape_attribute:
                    score_shape = 1.0
  
            # Update total scores
            img_score_spatial += score_spatial
            img_score_color += score_color
            img_score_texture += score_texture
            img_score_shape += score_shape
            
            
        # Store image stats
        image_stats[os.path.basename(file_name)] = {
            "bbox_count": bbox_count,
            "score_spatial": img_score_spatial,
            "score_color": img_score_color,
            "score_texture": img_score_texture,
            "score_shape": img_score_shape,
        }

        if len(image_stats) % 50 == 0:
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(image_stats, json_file, indent=4)
    
    # Save the image_stats dictionary to a JSON file
    with open(save_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(image_stats, json_file, indent=4)

    print(f"Image statistics saved to {save_json_path}")

    
    score_save_path = save_json_path.replace('minicpm-vqa.json', 'minicpm-vqa-score.txt')

    # Read the JSON file containing image statistics
    with open(save_json_path, "r") as f:
        json_data = json.load(f)

    total_num = 0
    total_bbox_num = 0
    total_score_spatial = 0
    total_score_color = 0
    total_score_texture = 0
    total_score_shape = 0

    miss_match =0
    # Iterate over the JSON data
    for key, value in json_data.items():
        
        total_num += value["bbox_count"]
        total_score_spatial +=value["score_spatial"] 
        total_score_color +=value["score_color"]
        total_score_texture +=value["score_texture"]
        total_score_shape +=value["score_shape"]

        if value["bbox_count"]!=value["score_spatial"] or value["bbox_count"]!=value["score_color"] or value["bbox_count"]!=value["score_texture"] or value["bbox_count"]!=value["score_shape"]:
            print(key,value["bbox_count"],value["score_spatial"],value["score_color"],value["score_texture"],value["score_shape"])
            miss_match+=1

    print(miss_match)
    #save total_score_spatial,total_score_color,total_score_texture,total_score_shape
    with open(score_save_path, "w") as f:
        f.write(f"Total number of bbox: {total_num}\n")
        f.write(f"Total score of spatial: {total_score_spatial}; Average score of spatial: {round(total_score_spatial/total_num,4)}\n")
        f.write(f"Total score of color: {total_score_color}; Average score of color: {round(total_score_color/total_num,4)}\n")
        f.write(f"Total score of texture: {total_score_texture}; Average score of texture: {round(total_score_texture/total_num,4)}\n")
        f.write(f"Total score of shape: {total_score_shape}; Average score of shape: {round(total_score_shape/total_num,4)}\n")
