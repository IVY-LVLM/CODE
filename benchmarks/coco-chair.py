import random
from datasets import load_dataset
import datetime
import json
import os
import re
import time
from collections import defaultdict
from email.policy import default
import shutil
import numpy as np
import pandas as pd
import PIL
import pytz
from datasets import load_dataset
from PIL import Image
import torch
from tqdm import tqdm
import urllib.request
from pathlib import Path

from tools.read_yaml import *
from benchmarks.chair._chair import CHAIR, print_metrics
from .base_eval_dataset import BaseEvalDataset
import pickle

from pycocotools.coco import COCO

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

result_dict = defaultdict(lambda: defaultdict(lambda: ""))


class COCOChairDataset(BaseEvalDataset):
    def __init__(self, data_path: str = "coco-chair", split="karpathy", default_output_path="COCO-CHAIR"):
        super().__init__("coco-chair", data_path)
        self.data_path = os.path.join(get_data_folder(), data_path)
        self.default_output_path = os.path.join(get_log_folder(), default_output_path)
        self.split = split

        annotation_file_path = os.path.join(get_data_folder(), "coco/annotations/instances_val2014.json")
        caption_file_path = os.path.join(get_data_folder(), "coco/annotations/captions_val2014.json")
        with open(annotation_file_path, "r") as f:
            lines = f.readlines()

        coco_anns = json.loads(lines[0])
        coco = COCO(caption_file_path)
        img_ids = coco.getImgIds()
        num_samples = 500
        sampled_img_ids = random.sample(img_ids, num_samples)
        img_files = []
        for cur_img_id in sampled_img_ids:
            cur_img = coco.loadImgs(cur_img_id)[0]
            cur_img_path = cur_img["file_name"]
            img_files.append(os.path.join(self.data_path, 'val2014', cur_img_path))
        self.img_files = img_files
        self.annotation = coco.imgs

    def _evaluate(self, model, result_file_manager):
        super()._evaluate(model, result_file_manager)
        if os.path.exists(self.default_output_path) == False:
            os.mkdir(self.default_output_path)

        cur_prompt = model.get_coco_caption_prompt()

        annotation_path = os.path.join(self.data_path, "annotations")
        evaluator = CHAIR(annotation_path, self.annotation) 
        evaluator.get_annotations()

        image_id_list = []
        prediction_list = []
        for row_idx, img_file in tqdm(enumerate(self.img_files)):
            filename = img_file
            image_id = int(filename.split('_')[-1].split('.')[0])
            
            if self.result_file_manager.is_absent_sample(row_idx) == False:
                response = self.result_file_manager.get_results(row_idx, 'raw_pred')
                image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                print('skipping: ', row_idx, image_id)
            else:
                image = Image.open(img_file).convert('RGB')

                response = model.generate(cur_prompt, image, dataset_name="coco-chair/val2014", image_path = os.path.basename(img_file))
                
                model_info = self.result_file_manager.get_model_infos(model) 
                self.result_file_manager.save_result(image_id=image_id, question=cur_prompt, gt_ans="", pred_ans=response, raw_pred=response, mmvp_gpt_grade=None, category_pope_mme="", model_info=model_info)
            image_id_list.append(image_id)

            prediction_list.append(response)

        cap_dict = evaluator.compute_chair(prediction_list, self.annotation, image_id_list) 
        total_score = defaultdict(list)
        total_score['CHAIRs'].append(cap_dict['overall_metrics']['CHAIRs'])
        total_score['CHAIRi'].append(cap_dict['overall_metrics']['CHAIRi'])
        total_score['BLEU-1'].append(cap_dict['overall_metrics']['BLEU-1'])
        total_score['BLEU-4'].append(cap_dict['overall_metrics']['BLEU-4'])
        self.result_file_manager.save_evaluation(total_score)