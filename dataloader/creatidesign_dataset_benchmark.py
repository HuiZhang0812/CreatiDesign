import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
def find_nearest_bucket_size(input_width, input_height, mode="x64", ratio=1):
    buckets = [
            (512, 2048),
            (512, 1984),
            (512, 1920),
            (512, 1856),
            (576, 1792),
            (576, 1728),
            (576, 1664),
            (640, 1600),
            (640, 1536),
            (704, 1472),
            (704, 1408),
            (704, 1344),
            (768, 1344),
            (768, 1280),
            (832, 1216),
            (832, 1152),
            (896, 1152),
            (896, 1088),
            (960, 1088),
            (960, 1024),
            (1024, 1024),
            (1024, 960),
            (1088, 960),
            (1088, 896),
            (1152, 896),
            (1152, 832),
            (1216, 832),
            (1280, 768),
            (1344, 768),
            (1408, 704),
            (1472, 704),
            (1536, 640),
            (1600, 640),
            (1664, 576),
            (1728, 576),
            (1792, 576),
            (1856, 512),
            (1920, 512),
            (1984, 512),
            (2048, 512)
        ]
    aspect_ratios = [w / h for (w, h) in buckets]

    assert mode in ["x64", "x8"]
    if mode == "x64":
        asp = input_width / input_height
        diff = [abs(ar - asp) for ar in aspect_ratios]
        bucket_id = int(np.argmin(diff))
        gen_width, gen_height = buckets[bucket_id]
    elif mode == "x8":
        max_pixels = 1024 * 1024
        ratio = (max_pixels / (input_width * input_height)) ** (0.5)
        gen_width, gen_height = round(input_width * ratio), round(input_height * ratio)
        gen_width = gen_width - gen_width % 8
        gen_height = gen_height - gen_height % 8
    else:
        raise NotImplementedError

    return (int(gen_width * ratio), int(gen_height * ratio))

def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    # Adjust and normalize bbox
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width,2)   
        y1_norm = round(y1 / orig_height,2)
        x2_norm = round(x2 / orig_width,2)
        y2_norm = round(y2 / orig_height,2)
       
        
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_bboxes

def img_transforms(image, height=512, width=512):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (height, width), interpolation=transforms.InterpolationMode.BILINEAR  
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_transformed = transform(image)
    return image_transformed

def mask_transforms(mask, height=512, width=512):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (height, width), 
                interpolation=transforms.InterpolationMode.NEAREST   
            ),
            transforms.ToTensor(),
        ]
    )
    mask_transformed = transform(mask)
    return mask_transformed


class DesignDataset(Dataset):

    def __init__(
        self,
        dataset_name,
        resolution=512,
        condition_resolution=512,
        condition_resolution_scale_ratio=0.5,
        max_boxes_per_image=10,
        neg_condition_image = 'same',
        background_color = 'gray',
        use_bucket=True,
        box_confidence_th = 0.0
    ):


        print(f"Loading dataset from Hugging Face: {dataset_name}")
        
        self.dataset = load_dataset(dataset_name, split="test")
        print(f"Loaded {len(self.dataset)} samples") 
        from IPython.core.debugger import set_trace 
        set_trace()
        self.max_boxes_per_image = max_boxes_per_image
        self.resolution = resolution
        self.condition_resolution=condition_resolution
        self.neg_condition_image = neg_condition_image
        self.use_bucket = use_bucket
        self.condition_resolution_scale_ratio=condition_resolution_scale_ratio
        self.box_confidence_th = box_confidence_th
       
        if background_color == 'white':
            self.background_color = (255, 255, 255)
        elif background_color == 'black':
            self.background_color = (0, 0, 0)
        elif background_color == 'gray':
            self.background_color = (128, 128, 128)
        else:
            raise ValueError("Invalid background color. Use 'white' or 'black'.")


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_source = sample['original_image']   
        subject_image = sample['condition_gray_background']
        subject_mask = sample['subject_mask']
        json_data = json.loads(sample['metadata'])  

        #img info 
        img_info = json_data['img_info']
        img_id = img_info['img_id']
        orig_width, orig_height = int(img_info["img_width"]),int(img_info["img_height"])

        if self.use_bucket:
            target_width, target_height = find_nearest_bucket_size(orig_width,orig_height)
            condition_width = int(target_width * self.condition_resolution_scale_ratio)
            condition_height = int(target_height * self.condition_resolution_scale_ratio)
        else:
            target_width = target_height = self.resolution
            condition_width = condition_height = self.condition_resolution


        img_tensor = img_transforms(image_source,height=target_height,width=target_width)


        # global caption
        global_caption = json_data['global_caption']


        # object_annotations
        object_annotations = json_data['object_annotations']

        # object bbox list
        objects_bbox = [item['bbox'] for item in object_annotations]

        # object bbox caption
        objects_caption = [item['bbox_detail_description'] for item in object_annotations]  
    
        # object bbox score
        objects_bbox_score = [item['score'][0] for item in object_annotations]

        # text
        text_list = json_data["text_list"]
        txt_bboxs = [item['bbox'] for item in text_list]
        txt_captions = ["text:"+item['text'] for item in text_list]
        
        txt_scores = [1.0 for _ in txt_bboxs]
        # combine bbox 和 description
        objects_bbox.extend(txt_bboxs)
        objects_caption.extend(txt_captions)
        objects_bbox_score.extend(txt_scores)

        objects_bbox =torch.tensor(adjust_and_normalize_bboxes(objects_bbox,orig_width,orig_height))
        
        objects_bbox_score = torch.tensor(objects_bbox_score)

        boxes_mask = objects_bbox_score > self.box_confidence_th
        objects_bbox_raw = objects_bbox[boxes_mask]
        objects_caption = [object_caption for object_caption, box_mask in zip(objects_caption, boxes_mask) if box_mask]


        num_boxes = objects_bbox_raw.shape[0]
        objects_boxes_padded = torch.zeros((self.max_boxes_per_image, 4)) 
        objects_masks_padded = torch.zeros(self.max_boxes_per_image)

        objects_caption = objects_caption[:self.max_boxes_per_image]
        objects_boxes_padded[:num_boxes] = objects_bbox_raw[:self.max_boxes_per_image]
        objects_masks_padded[:num_boxes] = 1.

        # objects_masks_maps
        objects_masks_maps_padded = torch.zeros((self.max_boxes_per_image, target_height, target_width))
        for idx in range(num_boxes):
            x1, y1, x2, y2 = objects_boxes_padded[idx]
            
            x1_pixel = int(x1 * target_width)
            y1_pixel = int(y1 * target_height)
            x2_pixel = int(x2 * target_width)
            y2_pixel = int(y2 * target_height)

           
            x1_pixel = max(0, min(x1_pixel, target_width-1))
            y1_pixel = max(0, min(y1_pixel, target_height-1))
            x2_pixel = max(0, min(x2_pixel, target_width-1))
            y2_pixel = max(0, min(y2_pixel, target_height-1))
        
            objects_masks_maps_padded[idx, y1_pixel:y2_pixel+1, x1_pixel:x2_pixel+1] = 1.0



        # subject
        original_size_subject_tensor = img_transforms(subject_image,height=target_height,width=target_width)
        subject_tensor = img_transforms(subject_image,height=condition_height,width=condition_width)
        subject_mask_tensor = mask_transforms(subject_mask, height=condition_height,width=condition_width)


        if self.neg_condition_image=='black':
            subject_image_black = Image.new('RGB', (orig_width, orig_height), (0, 0, 0))
            subject_image_neg_tensor = img_transforms(subject_image_black,height=condition_height,width=condition_width)
        elif self.neg_condition_image=='white':
            subject_image_white = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
            subject_image_neg_tensor = img_transforms(subject_image_white,height=condition_height,width=condition_width)
        elif self.neg_condition_image=='gray':
            subject_image_gray = Image.new('RGB', (orig_width, orig_height), (128, 128, 128))
            subject_image_neg_tensor = img_transforms(subject_image_gray,height=condition_height,width=condition_width)
        elif self.neg_condition_image=='same':
            subject_image_neg_tensor = subject_tensor


        output = dict(
            id=img_id,
            caption=global_caption,
            objects_boxes=objects_boxes_padded,
            objects_caption=objects_caption,
            objects_masks=objects_masks_padded,
            objects_masks_maps=objects_masks_maps_padded,
            img=img_tensor,
            condition_img_masks_maps = subject_mask_tensor,  
            condition_img = subject_tensor,
            original_size_condition_img = original_size_subject_tensor,
            neg_condtion_img = subject_image_neg_tensor,
            img_info = img_info,
            target_width=target_width,
            target_height=target_height,
        )

        return output


def collate_fn(examples):

    collated_examples = {}

    for key in ['id', 'objects_caption', 'caption','img_info','target_width','target_height']:
        collated_examples[key] = [example[key] for example in examples]
    
    for key in ['img', 'objects_boxes',  'objects_masks','condition_img','neg_condtion_img','objects_masks_maps','condition_img_masks_maps','original_size_condition_img']:
        collated_examples[key] = torch.stack([example[key] for example in examples]).float()
    
    return collated_examples




from typing import Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
def draw_mask(mask, draw, random_color=True):
    """Draws a mask with a specified color on an image.

    Args:
        mask (np.array): Binary mask as a NumPy array.
        draw (ImageDraw.Draw): ImageDraw object to draw on the image.
        random_color (bool): Whether to use a random color for the mask.
    """
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))
    
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
def visualize_bbox(image_pil: Image,
              result: Dict,
              draw_width: float = 6.0,
              return_mask=True) -> Image:
    """Plot bounding boxes and labels on an image with text wrapping for long descriptions.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        result (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): The target dictionary containing
            the bounding boxes and labels. The keys are:
                - boxes (List[int]): A list of bounding boxes in shape (N, 4), [x1, y1, x2, y2] format.
                - labels (List[str]): A list of labels for each object
                - masks (List[PIL.Image], optional): A list of masks in the format of PIL.Image

    Returns:
        PIL.Image: The input image with plotted bounding boxes, labels, and masks.
    """
    # Get the bounding boxes and labels from the target dictionary
    boxes = result["boxes"]
    categorys = result["labels"]
    masks = result.get("masks", [])

    color_list = [(255, 162, 76), (177, 214, 144),
                 (13, 146, 244), (249, 84, 84), (54, 186, 152),
                 (74, 36, 157), (0, 159, 189),
                 (80, 118, 135), (188, 90, 148), (119, 205, 255)]
    
    # Use smaller font size to allow more text to be displayed
    font_size = 30  # Reduce font size
    font = ImageFont.truetype("dataloader/arial.ttf", font_size)
    
    # Get image dimensions
    img_width, img_height = image_pil.size
    
    # Find all unique categories and build a cate2color dictionary
    cate2color = {}
    unique_categorys = sorted(set(categorys))
    for idx, cate in enumerate(unique_categorys):
        cate2color[cate] = color_list[idx % len(color_list)]

    # Create a PIL ImageDraw object to draw on the input image
    if isinstance(image_pil, np.ndarray):
        image_pil = Image.fromarray(image_pil)
    draw = ImageDraw.Draw(image_pil)
    
    # Create a new binary mask image with the same size as the input image
    mask = Image.new("L", image_pil.size, 0)
    # Create a PIL ImageDraw object to draw on the mask image
    mask_draw = ImageDraw.Draw(mask)

    # Draw boxes, labels, and masks for each box and label in the target dictionary
    for box, category in zip(boxes, categorys):
        # Extract the box coordinates
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        box_width = x1 - x0
        box_height = y1 - y0
        color = cate2color.get(category, color_list[0])  # Default color

        # Draw the box outline on the input image
        draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width))
        
        # Allow text box to be maximum 2 times the bounding box width, but not exceed image boundaries
        max_text_width = min(box_width * 2, img_width - x0)
        
        # Determine the maximum height for text background area
        max_text_height = min(box_height * 2, 200)  # Also allow more text display, but limit height
        
        # Handle long text based on bounding box width, split text into lines
        lines = []
        words = category.split()
        current_line = words[0]
        
        for word in words[1:]:
            # Try to add the next word
            test_line = current_line + " " + word
            # Use textbbox or textlength to check if width fits the maximum text width
            if hasattr(draw, "textbbox"):
                # Use textbbox method
                bbox = draw.textbbox((0, 0), test_line, font=font)
                w = bbox[2] - bbox[0]
            elif hasattr(draw, "textlength"):
                # Use textlength method
                w = draw.textlength(test_line, font=font)
            else:
                # Fallback - estimate width
                w = len(test_line) * (font_size * 0.6)  # Estimate average character width
                
            if w <= max_text_width - 20:  # Leave some margin
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        
        # Limit number of lines to prevent overflow
        max_lines = max_text_height // (font_size + 2)  # Line height (font size + spacing)
        if len(lines) > max_lines:
            lines = lines[:max_lines-1]
            lines.append("...")  # Add ellipsis
        
        # Calculate actual required width for each line
        line_widths = []
        for line in lines:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
            elif hasattr(draw, "textlength"):
                line_width = draw.textlength(line, font=font)
            else:
                line_width = len(line) * (font_size * 0.6)  # Estimate width
            line_widths.append(line_width)
        
        # Determine actual required width for text box
        if line_widths:
            needed_text_width = max(line_widths) + 10  # Add small margin
        else:
            needed_text_width = 0
            
        # Use bounding box width as minimum, only expand when needed
        text_bg_width = max(box_width, min(needed_text_width, max_text_width))
        
        # Ensure it doesn't exceed image boundaries
        text_bg_width = min(text_bg_width, img_width - x0)
        
        # Calculate text background height
        text_bg_height = len(lines) * (font_size + 2)
        
        # Ensure text background doesn't exceed image bottom
        if y0 + text_bg_height > img_height:
            # If it would exceed bottom, adjust text position to above the bounding box bottom
            text_y0 = max(0, y1 - text_bg_height)
        else:
            text_y0 = y0
        
        # Draw text background - note RGBA color handling
        if image_pil.mode == "RGBA":
            # For RGBA mode, we can directly use alpha color
            bg_color = (*color, 180)  # Semi-transparent background
        else:
            # For RGB mode, we cannot use alpha
            bg_color = color
            
        draw.rectangle([x0, text_y0, x0 + text_bg_width, text_y0 + text_bg_height], fill=bg_color)
        
        # Draw text
        for i, line in enumerate(lines):
            y_pos = text_y0 + i * (font_size + 2)
            draw.text((x0 + 5, y_pos), line, fill="white", font=font)

    # Draw the mask on the input image if masks are provided
    if len(masks) > 0 and return_mask:
        size = image_pil.size
        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            mask = np.array(mask)[:, :, -1]
            draw_mask(mask, mask_draw)

        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), mask_image).convert("RGB")
    
    return image_pil

import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageChops

def tensor_to_pil(img_tensor):
    """将tensor转换为PIL图像"""
    img_tensor = img_tensor.cpu()
    # 反归一化 ([0.5], [0.5])
    img_tensor = img_tensor * 0.5 + 0.5
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return T.ToPILImage()(img_tensor)

def make_image_grid_RGB(images, rows, cols, resize=None):
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), box=(i % cols * w, i // cols * h))
    return grid

if __name__ == "__main__":  
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
    test_dataloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=1,collate_fn=collate_fn)

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

