from collections import defaultdict
from email.policy import default
import json
import numpy as np
from tqdm import tqdm
import copy as cp
from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
import os
import datetime
import pytz
utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

def parse_choice(question):
    lines = question.split('\n')
    choices = {}
    for line in lines:
        if line == '':
            continue
        if line[0] == 'A':
            choices['A'] = line[2:].strip()
        elif line[0] == 'B':
            choices['B'] = line[2:].strip()
        elif line[0] == 'C':
            choices['C'] = line[2:].strip()
        elif line[0] == 'D':
            choices['D'] = line[2:].strip()
        elif line[0] == 'E':
            choices['E'] = line[2:].strip()
    return choices

def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

class RealworldQADataset(BaseEvalDataset):
    def __init__(self, data_path: str = "REALWORLDQA", split="test", default_output_path='REALWORLDQA'):
        super().__init__("realworldqa", data_path)
        
        self.default_output_path = os.path.join(get_log_folder(), default_output_path)

    def _evaluate(self, model, result_file_manager):
        super()._evaluate(model, result_file_manager)

        dataset = load_dataset("xai-org/RealworldQA")
        
        correct = 0
        wrong = 0
        question_type = ""
        gpt_grade = ""
        for row_idx, data in enumerate(tqdm(dataset['test'], desc="Realworld-QA")):
            image_id = row_idx
            image = data['image']
            question = data['question']
            gt_answer = data['answer']
            image_path = 'images/' + str(image_id) + '.jpg'
            if self.result_file_manager.is_absent_sample(row_idx) == False:
                pred_ans = self.result_file_manager.get_results(row_idx, 'pred_ans')
                image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                gpt_grade = self.result_file_manager.get_results(row_idx, 'gpt_grade')
                print('skipping: ', row_idx, image_id)
            else:
                cur_prompt = question
                raw_pred = model.generate(cur_prompt, image, 'realworldqa', image_path = image_path) 
                pred_ans = can_infer_option(raw_pred, parse_choice(question))
                if pred_ans == False:
                    pred_ans = raw_pred
                
                model_info = self.result_file_manager.get_model_infos(model)
                self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans=gt_answer, pred_ans=pred_ans, raw_pred=raw_pred, mmvp_gpt_grade=gpt_grade, category_pope_mme=question_type, model_info=model_info)
            
            if pred_ans.lower() == gt_answer.lower():
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        result_dict = defaultdict(str)
        result_dict['accuracy'] = str(accuracy)

        self.result_file_manager.save_evaluation(result_dict)

        print('Accuracy: {:.2f}'.format(accuracy))

