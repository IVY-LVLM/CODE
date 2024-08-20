from collections import defaultdict
import os
import datetime
from tqdm import tqdm, trange

from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
from typing import Union


class PopeDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path="lmms-lab/POPE",
        split="test",
        default_output_path="POPE",
        batch_size=1
    ):
        super().__init__("pope", data_path) #, max_batch_size=batch_size)
        default_output_path = os.path.join(get_log_folder(), default_output_path)

        print("Loading dataset from", data_path)
        self.data = load_dataset(data_path, split=split)
        print("Dataset loaded")
        self.default_output_path = default_output_path
        if not os.path.exists(default_output_path):
            os.makedirs(default_output_path)

    def parse_pred(self, text):
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "").lower()
        words = text.split(" ")

        if "not" in words or "no" in words:
            return "no"
        else:
            return "yes"

    def _evaluate(self, model, result_file_manager):
        super()._evaluate(model, result_file_manager)

        metrics = {
            "adversarial": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            # "popular": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            # "random": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "overall": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
        }

        row_idx = 0
        with tqdm(total=len(self.data), desc="POPE") as pbar:
            for ri, pope_data in enumerate(self.data):
                question = pope_data["question"]
                gt_ans = pope_data["answer"]
                image = pope_data["image"]
                image_id = pope_data["image_source"]
                category = pope_data["category"]

                if category != "adversarial":
                    pbar.update(1)
                    row_idx = row_idx + 1
                    continue

                if self.result_file_manager.is_absent_sample(row_idx) == False:
                    pred_ans = self.result_file_manager.get_results(row_idx, 'pred_ans')
                    image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                    print('skipping: ', row_idx, image_id)
                else:
                    response = model.generate(question, image, 'pope', image_path = image_id + '.jpg')

                    pred_ans = self.parse_pred(response)

                    model_info = self.result_file_manager.get_model_infos(model)
                    self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans=gt_ans, pred_ans=pred_ans, raw_pred=response, mmvp_gpt_grade=None, category_pope_mme=category, model_info=model_info)

                answer = gt_ans

                if pred_ans == "yes":
                    metrics[category]["yes_count"] += 1
                    metrics["overall"]["yes_count"] += 1
                else:
                    metrics[category]["no_count"] += 1
                    metrics["overall"]["no_count"] += 1

                if pred_ans == answer and pred_ans == "yes":
                    metrics[category]["TP"] += 1
                    metrics["overall"]["TP"] += 1
                elif pred_ans == answer and pred_ans == "no":
                    metrics[category]["TN"] += 1
                    metrics["overall"]["TN"] += 1
                elif pred_ans != answer and pred_ans == "yes":
                    metrics[category]["FP"] += 1
                    metrics["overall"]["FP"] += 1
                else:
                    metrics[category]["FN"] += 1
                    metrics["overall"]["FN"] += 1

                pbar.update(1)
                row_idx = row_idx + 1

        for category in metrics:
            print(f"----------- {category} -----------")

            TP = metrics[category]["TP"]
            TN = metrics[category]["TN"]
            FP = metrics[category]["FP"]
            FN = metrics[category]["FN"]
            yes_count = metrics[category]["yes_count"]
            no_count = metrics[category]["no_count"]

            print("TP\tFP\tTN\tFN\t")
            print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

            if TP + FP == 0:
                metrics[category]["precision"] = precision = 0
            else:
                metrics[category]["precision"] = precision = float(TP) / float(TP + FP)

            if TP + FN == 0:
                metrics[category]["recall"] = recall = 0
            else:
                metrics[category]["recall"] = recall = float(TP) / float(TP + FN)

            if precision + recall == 0:
                metrics[category]["f1"] = f1 = 0
            else:
                metrics[category]["f1"] = f1 = 2 * precision * recall / float(precision + recall)

            metrics[category]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)

            if yes_count + no_count == 0:
                metrics[category]["yes_ratio"] = yes_ratio = 0
            else:
                metrics[category]["yes_ratio"] = yes_ratio = yes_count / float(yes_count + no_count)

            print("Accuracy: {}".format(acc))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1 score: {}".format(f1))
            print("Yes ratio: {}".format(yes_ratio))

            result_dict = {}
            result_dict[category] = {} 
            result_dict[category]['acc'] = acc
            result_dict[category]['precision'] = precision
            result_dict[category]['recall'] = recall
            result_dict[category]['f1'] = f1
            result_dict[category]['yes_ratio'] = yes_ratio
            self.result_file_manager.save_evaluation(result_dict)
            

        # print(f"----------- overall -----------")

        # TP = metrics["overall"]["TP"]
        # TN = metrics["overall"]["TN"]
        # FP = metrics["overall"]["FP"]
        # FN = metrics["overall"]["FN"]
        # yes_count = metrics["overall"]["yes_count"]
        # no_count = metrics["overall"]["no_count"]

        # print("TP\tFP\tTN\tFN\t")
        # print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

        # metrics["overall"]["precision"] = precision = float(TP) / float(TP + FP)
        # metrics["overall"]["recall"] = recall = float(TP) / float(TP + FN)
        # metrics["overall"]["f1"] = f1 = 2 * precision * recall / float(precision + recall)
        # metrics["overall"]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)
        # metrics["overall"]["yes_ratio"] = yes_ratio = float(yes_count) / float(yes_count + no_count)

        # print("Accuracy: {}".format(acc))
        # print("Precision: {}".format(precision))
        # print("Recall: {}".format(recall))
        # print("F1 score: {}".format(f1))
        # print("Yes ratio: {}".format(yes_ratio))

        # result_dict = {}
        # result_dict['overall'] = {} 
        # result_dict['overall']['acc'] = acc
        # result_dict['overall']['precision'] = precision
        # result_dict['overall']['recall'] = recall
        # result_dict['overall']['f1'] = f1
        # result_dict['overall']['yes_ratio'] = yes_ratio
        # self.result_file_manager.save_evaluation(result_dict)

        return metrics

    def calc_value(self, metrics):
        with open ('pope_result.txt', 'a') as f:
            f.write(f"===========================================================================\n")
            f.write(f"*** FILENAME: {self.result_file_manager.filename}\n")
        for category in metrics:
            with open ('pope_result.txt', 'a') as f:
                f.write(f"----------- {category} -----------")
            print(f"----------- {category} -----------")

            TP = metrics[category]["TP"]
            TN = metrics[category]["TN"]
            FP = metrics[category]["FP"]
            FN = metrics[category]["FN"]
            yes_count = metrics[category]["yes_count"]
            no_count = metrics[category]["no_count"]

            print("TP\tFP\tTN\tFN\t")
            print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))
            
            with open ('pope_result.txt', 'a') as f:
                f.write("TP\tFP\tTN\tFN\t")
                f.write('\n')
                f.write("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))
                f.write('\n')

            if TP + FP == 0:
                metrics[category]["precision"] = precision = 0
            else:
                metrics[category]["precision"] = precision = float(TP) / float(TP + FP)

            if TP + FN == 0:
                metrics[category]["recall"] = recall = 0
            else:
                metrics[category]["recall"] = recall = float(TP) / float(TP + FN)

            if precision + recall == 0:
                metrics[category]["f1"] = f1 = 0
            else:
                metrics[category]["f1"] = f1 = 2 * precision * recall / float(precision + recall)

            metrics[category]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)

            if yes_count + no_count == 0:
                metrics[category]["yes_ratio"] = yes_ratio = 0
            else:
                metrics[category]["yes_ratio"] = yes_ratio = yes_count / float(yes_count + no_count)

            print("Accuracy: {}".format(acc))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1 score: {}".format(f1))
            print("Yes ratio: {}".format(yes_ratio))

            with open ('pope_result.txt', 'a') as f:
                f.write("Accuracy: {}".format(acc))
                f.write('\n')
                f.write("Precision: {}".format(precision))
                f.write('\n')
                f.write("Recall: {}".format(recall))
                f.write('\n')
                f.write("F1 score: {}".format(f1))
                f.write('\n')
                f.write("Yes ratio: {}".format(yes_ratio))
                f.write('\n')

            result_dict = {}
            result_dict[category] = {} 
            result_dict[category]['acc'] = acc
            result_dict[category]['precision'] = precision
            result_dict[category]['recall'] = recall
            result_dict[category]['f1'] = f1
            result_dict[category]['yes_ratio'] = yes_ratio