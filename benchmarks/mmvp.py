from collections import defaultdict
from email.policy import default
import json
import numpy as np
from tqdm import tqdm

from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
import os
import datetime
from PIL import Image  
import PIL  
import pandas as pd
from openai import OpenAI
import time
import re
import pytz
from typing import Union

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)
NUM_SECONDS_TO_SLEEP = 10

def get_yes_no_answer(question):
    while True:
        try:
            response = client.chat.completions.create(
                model='gpt-4-0125-preview', #'gpt-4-0613',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.'
                }, {
                    'role': 'user',
                    'content': question,
                }],
                temperature=0.0,  # TODO: figure out which temperature is best for evaluation
            )
            break
        except Exception as e:
            print(e)
            time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response.choices[0].message.content
    answer = answer.replace('.', '')
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."

class MMVPDataset(BaseEvalDataset):
    def __init__(self, data_path: str = "MMVP", split="test", default_output_path="MMVP"):
        super().__init__("mmvp", data_path)
        
        self.default_output_path = os.path.join(get_log_folder(), data_path)

    def _evaluate(self, model, result_file_manager):
        super()._evaluate(model, result_file_manager)

        benchmark_dir = 'benchmarks/mmvp/Questions.csv'
        df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
        num_correct, num_total = 0, 0
        index, round_correct = 0, 0

        for i, (row_idx, row) in enumerate(tqdm(df.iterrows(), desc="MMVP")):
        # Construct the 'prompts' string
            if self.result_file_manager.is_absent_sample(row_idx) == False:
                pred_ans = self.result_file_manager.get_results(row_idx, 'pred_ans')
                image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                gpt_grade = self.result_file_manager.get_results(row_idx, 'gpt_grade')
                print('skipping: ', row_idx, image_id)
            else:
                cur_prompt = row['Question'] + " " + row['Options']

                photo_id = row_idx+1
                image_id = str(photo_id)
                image_path = os.path.join(get_data_folder(), 'mmvp/', 'MMVP Images', f"{photo_id}.jpg")
                image = Image.open(image_path).convert('RGB')
                response = model.generate(cur_prompt, image, 'mmvp', image_path = image_path) 
                question, correct_answer, pred_ans = cur_prompt, row["Correct Answer"], response
                
                question4gpt = f"Given the following question {question}, the correct answer is {correct_answer}. Does the following answer correctly answers the question, answer:{response}?"

                gpt_grade = get_yes_no_answer(question4gpt)

                model_info = self.result_file_manager.get_model_infos(model)
                self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans=correct_answer, pred_ans=pred_ans, raw_pred=response, mmvp_gpt_grade=gpt_grade, category_pope_mme=None, model_info=model_info)

            index += 1
            if gpt_grade=="yes":
                round_correct += 1
            if index == 2:
                index = 0
                if round_correct == 2:
                    num_correct += 1
                round_correct = 0

                num_total += 1 

        accuracy = num_correct/num_total
        result_dict = defaultdict(str)
        result_dict['accuracy'] = str(accuracy)
        self.result_file_manager.save_evaluation(result_dict)
        print(f"The accuracy is {accuracy}")