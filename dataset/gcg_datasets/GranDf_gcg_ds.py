import os
import cv2
import json
import random
import warnings
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import (
    GCG_QUESTIONS, 
    DUAL_STREAM_GCG_QUESTIONS,
    CONSISTENCY_SUFFIX_TEMPLATES,
    CONSISTENCY_INTRO_TEMPLATES,
)
import pdb

class GCGBaseDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True,
                 image_dir='', json_path='', use_dual_stream=False):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.validation = validation
        # random_sampling is deprecated: shuffle should be handled by DataLoader/Sampler
        if random_sampling is not True:
            warnings.warn(
                "random_sampling parameter is deprecated and ignored. "
                "Shuffle is now handled by the DataLoader's Sampler.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.random_sampling = random_sampling  # kept for backward compat, but ignored
        self.use_dual_stream = use_dual_stream

        # Use dual-stream prompts if enabled
        if use_dual_stream:
            self.question_templates = DUAL_STREAM_GCG_QUESTIONS
        else:
            self.question_templates = GCG_QUESTIONS
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation

        # Defining paths
        self.base_dir = dataset_dir
        self.image_folder = os.path.join(image_dir)
        if self.validation:
            self.ann_file = os.path.join(self.base_dir, "test", "annotations", json_path)
        else:
            self.ann_file = os.path.join(self.base_dir, "train", "annotations", json_path)

        with open(self.ann_file, "r") as file:
            datas = json.load(file)
        self.epoch_samples = len(datas)
        self.data_infos = self._load_annotations(self.ann_file)

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['file_name']}
        width, height = Image.open(image_path).size
        annotations['caption'] = ann_info['caption'].strip('"').strip()

        for word, grounding in ann_info["groundings"].items():
            annotations['labels'].append(word)
            annotations['tokens_positive'].append(grounding["token_positives"])

            # Convert segmentation to binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for rle in grounding["rle_masks"]:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            annotations['masks'].append(binary_mask)

        return annotations

    def __getitem__(self, index):
        # Standard PyTorch contract: return exactly the sample at `index`.
        # Shuffle is the responsibility of the DataLoader / Sampler.
        for _ in range(len(self.data_infos)):
            ann_info = self.data_infos[index]
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['labels']) > 0:
                break
            # Deterministic fallback: advance to the next sample
            index = (index + 1) % len(self.data_infos)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
            "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}
        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, caption, tokens_positive):
        question = random.choice(self.question_templates).strip()

        # Prepare caption with tags, optionally injecting consistency confirmations
        def tag_caption(caption, tokens, inject_consistency=False):
            """
            Tag caption with segmentation markers and optionally inject consistency phrases.
            
            Args:
                caption: The original caption text
                tokens: List of (start, end) tuples for phrases to tag
                inject_consistency: Whether to add consistency confirmation phrases
            
            Note: Consistency phrases are placed AFTER [SEG] to avoid polluting 
            the hidden state that drives the SAM decoder. The [SEG] token's 
            embedding should be driven purely by the target phrase semantics.
            """
            # Process tokens in reverse order to preserve indices
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                phrase = caption[start:end]
                
                # if inject_consistency and random.random() < 0.7:
                #     # 70% chance to inject consistency confirmation for each phrase
                #     # Place consistency info AFTER [SEG] to keep <p>...</p> clean
                #     suffix = random.choice(CONSISTENCY_SUFFIX_TEMPLATES)
                #     tagged = f"<p> {phrase} </p> [SEG]{suffix}"
                # else:
                #     # Original tagging without consistency injection
                #     tagged = f"<p> {phrase} </p> [SEG]"
                
                tagged = f"<p> {phrase} </p> [SEG]"
                
                caption = f"{caption[:start]}{tagged}{caption[end:]}"
            
            return caption

        # Decide whether to inject consistency based on dual-stream mode
        inject_consistency = self.use_dual_stream
        detailed_answer = tag_caption(caption, tokens_positive, inject_consistency)
        
        # Optionally add an intro phrase for dual-stream mode
        if self.use_dual_stream and random.random() < 0.5:
            intro = random.choice(CONSISTENCY_INTRO_TEMPLATES)
            detailed_answer = intro + detailed_answer

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]
        return questions, conversations

    def process_data(self, data_item):
        data_labels = data_item['labels']
        masks = data_item['masks']
        caption = data_item['caption']
        tokens_positive = data_item['tokens_positive']
        image_path = data_item['image_path']

        # Function to sort elements based on the start index of each phrase
        def sort_by_start_index(items, order):
            return [items[i] for i in order]

        # Sort phrases based on their appearance in the sentence
        phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        masks = sort_by_start_index(masks, phrase_order)
        data_labels = sort_by_start_index(data_labels, phrase_order)
        tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        bboxes = None

        questions, conversations = self.create_conversations(caption, tokens_positive)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL
        selected_labels = data_labels

        return (
        image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize, questions,
        selected_labels)
    

class GranDfDataset(GCGBaseDataset):
    """
    Human annotated dataset proposed in GLaMM as part of GranDf dataset.
    """
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        self.base_dir = os.path.join(dataset_dir, "GranDf")
        json_path = "GranDf_HA_GCG_train.json"
        image_dir = os.path.join(self.base_dir, "GranDf_HA_images", "train")
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, random_sampling, image_dir, json_path, )
        print('\033[92m' + "----GCG-{}: GranDf-GCG dataset initialized----".format(mode) + '\033[0m')


class OpenPsgGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        json_files = {'validation': "OpenPsgGCG_val.json", 'training': "OpenPsgGCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = os.path.join("coco_2017", "train2017")
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, random_sampling, image_dir, json_path, )
        print('\033[92m' + "----GCG-{}: OpenPSG-GCG dataset initialized----".format(mode) + '\033[0m')


class Flickr30kGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        json_files = {'validation': "flickr_mergedGT_GCG_val.json", 'training': "flickr_mergedGT_GCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = os.path.join("flikcr_30k", "train")
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, random_sampling, image_dir, json_path, )
        # Filter out images smaller than the minimum size
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]
        self.validation = validation
        print('\033[92m' + "----GCG-{}: Flickr30k-GCG dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, ann_file):
        # Load annotations and filter out images with very short captions
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            if len(data_infos) == 1000 and self.validation:
                # Only limited images for validation
                break
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{ann_file}'!"
        print(f'Removed {removed_img_count} images.')
        return data_infos

    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'bboxes': [], 'labels': [], 'bboxes_ignore': [], 'caption': img_info['caption'], 'masks': [],
                       'tokens_positive': []}
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0 or ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            annotations['bboxes'].append(bbox)
            tokens_positive = ann['tokens_positive']
            gt_label = [img_info['caption'][span[0]:span[1]] for span in tokens_positive]
            annotations['labels'].append(gt_label[0])
            annotations['tokens_positive'].append(tokens_positive[0])

            rle = ann['sam_mask']
            mask_decoded = mask.decode(rle).astype(np.uint8)
            annotations['masks'].append(mask_decoded)

        # Convert bounding boxes to numpy arrays
        annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations[
            'bboxes'] else np.zeros((0, 4), dtype=np.float32)
        annotations['bboxes_ignore'] = np.array(annotations['bboxes_ignore'], dtype=np.float32) if annotations[
            'bboxes_ignore'] else np.zeros((0, 4), dtype=np.float32)

        return annotations

    def __getitem__(self, index):
        img_info = self.data_infos[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        ann_info = self.coco.loadAnns(ann_ids)
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        # Parse annotation info
        ann = self._parse_annotations(img_info, ann_info)
        data_item = {"image_path": image_path, "filename": img_info['file_name'], "width": img_info['width'],
                     "height": img_info['height'], "bbox": ann['bboxes'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}
        return self.process_data(data_item)


class RefCOCOgGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=512, num_classes_per_sample=3, validation=False, random_sampling=True):
        json_files = {'validation': "RefCOCOg_GCG_val.json", 'training': "RefCOCOg_GCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = os.path.join("coco_2014", "train2014")
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, random_sampling, image_dir, json_path, )
        print('\033[92m' + "----GCG-{}: RefCOCOg-GCG dataset initialized----".format(mode) + '\033[0m')
    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['img_file_name']}
        width, height = Image.open(image_path).size
        orig_caption = ann_info['caption'].strip('"').strip()
        annotations['caption'] = orig_caption.lower()

        for detail in ann_info['refs']:
            phrase = detail['sentence']
            if phrase.lower() in annotations['caption']:
                annotations['labels'].append(phrase)
                index = annotations['caption'].find(phrase)
                end_index = index + len(phrase) if index != -1 else -1
                annotations['tokens_positive'].append([index, end_index])

                # Convert segmentation to binary mask
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    rles = mask.frPyObjects([seg], height, width)
                    m = mask.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                annotations['masks'].append(binary_mask)

        # Sort tokens_positive and corresponding lists
        tokens_positive = annotations['tokens_positive']
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
        annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
        annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]

        # Trimming overlapping intervals
        for i in range(len(tokens_positive)):
            for j in range(i + 1, len(tokens_positive)):
                # If there is overlap
                if tokens_positive[i][1] >= tokens_positive[j][0]:
                    # Modify the end index of phrase i to be one less than the start index of phrase j
                    tokens_positive[i][1] = tokens_positive[j][0] - 1
                    # Modify the phrases to reflect the change in indices
                    annotations['labels'][i] = orig_caption[tokens_positive[i][0]:tokens_positive[i][1] + 1]
                    break  # Exit inner loop since i was modified

        return annotations

    def __getitem__(self, index):
        for _ in range(len(self.data_infos)):
            ann_dict = self.data_infos[index]
            ann_info = next(iter(ann_dict.values()))
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['labels']) > 0:
                break
            index = (index + 1) % len(self.data_infos)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}

        return self.process_data(data_item)


class LaPGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=512, num_classes_per_sample=3, validation=False, random_sampling=True,
                 use_dual_stream=False):
        json_files = {'validation': "test.json", 'training': "train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = os.path.join(dataset_dir, "train", "images") if not validation else os.path.join(dataset_dir, "test", "images")
        mode = "Val" if validation else "Train"
        epoch_samples = epoch_samples
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, image_dir=image_dir, json_path=json_path, use_dual_stream=use_dual_stream)
        print('\033[92m' + "----GCG-{}: LaP-Forensic Dataset initialized----".format(mode) + '\033[0m')

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['img_file_name']}
        width, height = Image.open(image_path).size
        orig_caption = ann_info['caption'].strip('"').strip()
        annotations['caption'] = orig_caption
        for detail in ann_info['refs']:
            phrase = detail['sentence']
            if phrase in annotations['caption']:
                annotations['labels'].append(phrase)
                index = annotations['caption'].find(phrase)
                end_index = index + len(phrase) if index != -1 else -1
                annotations['tokens_positive'].append([index, end_index])

                # Convert segmentation to binary mask
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    seg = np.array(seg)
                    rles = mask.frPyObjects([seg], height, width)
                    m = mask.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                annotations['masks'].append(binary_mask)
        # Sort tokens_positive and corresponding lists
        tokens_positive = annotations['tokens_positive']
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
        annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
        annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]

        return annotations
    

    def __getitem__(self, index):
        for _ in range(len(self.data_infos)):
            ann_dict = self.data_infos[index]
            ann_info = next(iter(ann_dict.values()))
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['labels']) > 0:
                break
            index = (index + 1) % len(self.data_infos)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}

        return self.process_data(data_item)
