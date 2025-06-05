from random import uniform
import torch
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from IPython.core.debugger import set_trace 
from dataloader.creatidesign_dataset_benchmark import DesignDataset,visualize_bbox,collate_fn,tensor_to_pil,make_image_grid_RGB
import numpy as np
from PIL import Image
from safetensors.torch import save_file, load_file
from accelerate import load_checkpoint_and_dispatch
from modules.flux.transformer_flux_creatidesign import FluxTransformer2DModel
from pipeline.pipeline_flux_creatidesign import FluxPipeline
import json
from huggingface_hub import snapshot_download
from datasets import load_dataset

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    resolution = 1024
    condition_resolution = 512
    neg_condition_image = 'same'
    background_color = 'gray'
    use_bucket = True
    condition_resolution_scale_ratio=0.5
    
    benchmark_repo = 'HuiZhang0812/CreatiDesign_benchmark' #  huggingface repo of benchmark
    
    datasets = DesignDataset(dataset_name=benchmark_repo,
                             resolution=resolution,
                             condition_resolution=condition_resolution,
                             neg_condition_image =neg_condition_image,
                             background_color=background_color,
                             use_bucket=use_bucket,
                             condition_resolution_scale_ratio=condition_resolution_scale_ratio
                             )
    test_dataloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4,collate_fn=collate_fn)

    
    model_path = "black-forest-labs/FLUX.1-dev"

    ckpt_repo = "HuiZhang0812/CreatiDesign" # huggingface repo of ckpt

    ckpt_path = snapshot_download(
        repo_id=ckpt_repo,
        repo_type="model",
        local_dir="./CreatiDesign_checkpoint",
        local_dir_use_symlinks=False
    )

    # Load transformer config from checkpoint
    with open(os.path.join(ckpt_path, "transformer", "config.json"), 'r') as f:
        config = json.load(f)
    
    transformer = FluxTransformer2DModel(**config)
    transformer = load_checkpoint_and_dispatch(transformer, checkpoint=os.path.join(model_path,"transformer"), device_map=None)

    # Load lora parameters using safetensors
    state_dict = load_file(os.path.join(ckpt_path, "transformer","model.safetensors"))

    # Load parameters, allow partial loading
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded parameters: {len(state_dict)}",state_dict.keys())
    print(f"Missing keys: {len(missing_keys)}",missing_keys)
    print(f"Unexpected keys: {len(unexpected_keys)}",unexpected_keys)

    transformer = transformer.to(dtype=torch.bfloat16)

    pipe = FluxPipeline.from_pretrained(model_path, transformer=transformer,torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    
    seed=42
    num_samples = 1
    true_cfg_scale=3.5
    guidance_scale=1.0
    if resolution == 512:
        position_delta=[0,-32]
    else:
        position_delta=[0,-64]
    if use_bucket:
        scale_h = 1/condition_resolution_scale_ratio
        scale_w = 1/condition_resolution_scale_ratio
    else:
        scale_h = resolution/condition_resolution
        scale_w = resolution/condition_resolution

    num_inference_steps = 28

    # Create save directory based on benchmark directory name
    save_root =os.path.join("outputs",benchmark_repo.split("/")[-1])
    os.makedirs(save_root,exist_ok=True)

    img_save_root = os.path.join(save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)

    img_withgt_save_root = os.path.join(save_root,"images_with_gt")
    os.makedirs(img_withgt_save_root,exist_ok=True)

    total_time = 0
    for i, batch in enumerate(tqdm(test_dataloader)):
        prompts = batch["caption"] 
        imgs_id = batch['id']
        objects_boxes = batch["objects_boxes"]
        objects_caption = batch['objects_caption'] 
        objects_masks = batch['objects_masks']
        condition_img = batch['condition_img']
        neg_condtion_img = batch['neg_condtion_img']
        objects_masks_maps= batch['objects_masks_maps']
        subject_masks_maps = batch['condition_img_masks_maps']
        target_width=batch['target_width'][0]
        target_height=batch['target_height'][0]

        img_info = batch["img_info"][0] 
        filename = img_info["img_id"]+'.jpg'
        start_time = time.time()
        with torch.no_grad():
            images = pipe(prompt=prompts*num_samples,
                          generator=torch.Generator(device="cuda").manual_seed(seed),
                          num_inference_steps = num_inference_steps,
                          objects_boxes=objects_boxes,
                          objects_caption=objects_caption,
                          objects_masks = objects_masks,
                          objects_masks_maps=objects_masks_maps,
                          condition_img = condition_img,
                          subject_masks_maps = subject_masks_maps,
                          neg_condtion_img = neg_condtion_img,
                          height= target_height,
                          width = target_width,
                          true_cfg_scale = true_cfg_scale,
                          position_delta=position_delta,
                          guidance_scale=guidance_scale,
                          scale_h = scale_h,
                          scale_w = scale_w,
                          use_bucket=use_bucket
                    )   
        images=images.images
        use_time = time.time() - start_time
        total_time +=use_time

        make_image_grid_RGB(images, rows=1, cols=num_samples).save(os.path.join(img_save_root,filename))
        use_time = time.time() - start_time
        total_time +=use_time

        # Process original image and bounding boxes
        ori_image = tensor_to_pil(batch['img'][0])
        orig_width, orig_height = ori_image.size
        normalized_boxes = batch['objects_boxes'][0].cpu().numpy()
        denormalized_boxes = []
        for box in normalized_boxes:
            x1, y1, x2, y2 = box
            denorm_box = [
                x1 * orig_width,  # x1
                y1 * orig_height, # y1
                x2 * orig_width,  # x2
                y2 * orig_height  # y2
            ]
            denormalized_boxes.append(denorm_box)
        
        objects_result = {
            "boxes": denormalized_boxes,
            "labels": batch['objects_caption'][0],
            "masks": []
        }
        
        # Only keep boxes and captions where mask is 1
        valid_boxes = []
        valid_labels = []
        for box, label, mask in zip(objects_result['boxes'], 
                                objects_result['labels'], 
                                batch['objects_masks'][0]):
            if mask:
                valid_boxes.append(box)
                valid_labels.append(label)
        
        objects_result['boxes'] = valid_boxes
        objects_result['labels'] = valid_labels

        ori_image_with_bbox = visualize_bbox(ori_image ,objects_result)

        # Concatenate images
        total_width = ori_image.width + ori_image.width+ num_samples*ori_image.width
        max_height = ori_image.height

        # Create a new blank image to hold the concatenated images
        new_image = Image.new('RGB', (total_width, max_height))
        
        new_image.paste(ori_image_with_bbox, (0, 0))

        # Process condition image
        condition_img = tensor_to_pil(batch['original_size_condition_img'][0])
        subject_canvas_with_bbox = visualize_bbox(condition_img ,objects_result)

        new_image.paste(subject_canvas_with_bbox, (ori_image.width, 0))  
    
        # Paste generated images
        for j, image in enumerate(images):   

            save_name=os.path.join(img_withgt_save_root,filename)
            
            image_with_bbox = visualize_bbox(image ,objects_result)
           
            new_image.paste(image_with_bbox, (ori_image.width*(j+2), 0))
            
        new_image.save(save_name)

    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(test_dataloader):.2f} seconds")