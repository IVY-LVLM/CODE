import datetime
import json
import os
import signal

from tools.read_yaml import get_log_folder

default_result_log_folder = get_log_folder()

'''
* Result file name format
{benchmark}_{model}_results_{Month}{Day}_{version}.jsonl

* Result file content
- firstline: temperature, max_new_tokens
- others: image_id, question, gt_ans, pred_ans, raw_pred, gpt_grade, category

* Result file example
{"temperature": 0.0, "max_new_tokens": 2048}
{"image_id": "1", "question": "Are the butterfly's wings closer to being open or closed? (a) Open (b) Closed", "gt_ans": "(a)", "pred_ans": "Open", "raw_pred": "Open", "gpt_grade": "yes", "category": null}
{"image_id": "2", "question": "Are the butterfly's wings closer to being open or closed? (a) Open (b) Closed", "gt_ans": "(b)", "pred_ans": "Open", "raw_pred": "Open", "gpt_grade": "no", "category": null, }
'''

class ResultFileManager:
    def __init__(self, model, benchmark, continue_file, new_file) -> None:
        self.model = model
        self.benchmark = benchmark
        self.default_result_log_folder = os.path.join(default_result_log_folder, benchmark.upper())
        if not os.path.exists(self.default_result_log_folder):
            os.mkdir(self.default_result_log_folder)
        if continue_file != "" and self.model in continue_file:
            if os.path.exists(os.path.join(self.default_result_log_folder, continue_file)):
                self.filename = continue_file
                self.filepath = os.path.join(self.default_result_log_folder, continue_file)
                self.cur_reviews = self._read_lines()
                print('Continue with the previous result file: ', continue_file)
                return
        self.filename = self._get_new_result_filename()
        print('<<<<<=====Result file=====>>>>>')
        print("Here's the list of result files for the given model and benchmark: ")
        print('1. New result file')
        result_files_list = self._list_result_files()
        for idx, file in enumerate(result_files_list):
            print("{}. {}".format(idx + 2, file))

                
        # This function raises an exception when the input time expires
        def timeout_handler(signum, frame):
            raise TimeoutError

        # Set the signal handler for the SIGALRM signal
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            # Set an alarm for 5 seconds
            signal.alarm(5)
            # Input with prompt
            if new_file:
                filename_idx = 1
            else:
                filename_idx = input(f"Enter the number of your selection(1-{len(result_files_list) + 1}): ")
            # Cancel the alarm after the input is received
            signal.alarm(0)
        except TimeoutError:
            print("!!!!! You did not respond in time, setting default value.")
            filename_idx = 1

        assert 1 <= int(filename_idx) <= len(result_files_list) + 1, "Invalid input"
        if int(filename_idx) == 1:
            self.filepath = os.path.join(self.default_result_log_folder, self.filename)
            self.cur_reviews = []
            print('New result file: ', self.filename)
            from pathlib import Path
            Path(self.filepath).touch()
        else:
            self.filename = result_files_list[int(filename_idx) - 2]
            self.filepath = os.path.join(self.default_result_log_folder, self.filename)
            self.cur_reviews = self._read_lines()
            print('Continue with the previous result file: ', self.filename)
        print("===========================================================================")
        print()
        print()

    def _list_result_files(self, date=None) -> list:
        if date == None:
            result_files = [f for f in os.listdir(self.default_result_log_folder) if f.startswith("{}_{}_results".format(self.benchmark, self.model))]
        else:
            result_files = [f for f in os.listdir(self.default_result_log_folder) if f.startswith("{}_{}_results_{}".format(self.benchmark, self.model, date))]
        return sorted(result_files)

    def _get_new_result_filename(self):
        date = "{}{}".format("%02d"%datetime.datetime.now().month, datetime.datetime.now().day)
        result_files_list = self._list_result_files()
        version_list = []
        for filename in result_files_list:
            if date in filename:
                version_list.append(int(filename.split("_v")[-1].split('_')[0].split(".")[0]))
        version = max(version_list) + 1 if version_list else 1
        new_filename = "{}_{}_results_{}_v{}.jsonl".format(self.benchmark, self.model, date, version)
        return new_filename

    def _check_file_existence(self, filename:str) -> bool:
        return os.path.exists(os.path.join(self.default_result_log_folder, filename)) and len(self.cur_reviews) != 0 
    
    def get_model_infos(self, model):
        if hasattr(model, "temperature"):
            temperature = model.temperature
        else:
            temperature = -1.0
        if hasattr(model, "max_new_tokens"):
            max_new_tokens = model.max_new_tokens
        else:
            max_new_tokens = -1
        if hasattr(model, "num_beams"):
            num_beams = model.num_beams
        else:
            num_beams = -1
        if hasattr(model, "do_sample"):
            do_sample = model.do_sample
        else:
            do_sample = False
        if hasattr(model, "top_p"):
            top_p = model.top_p
        else:
            top_p = -1.0
        if hasattr(model, "alt_text") or hasattr(model, "alt-text"):
            alt_text = model.alt_text
        else:
            alt_text = False
        if hasattr(model, "contrastive"):
            contrastive = model.contrastive
        else:
            contrastive = False
        if hasattr(model, "opera_decoding"):
            opera_decoding = model.opera_decoding
        else:
            opera_decoding = False
        if hasattr(model, "vcd_decoding"):
            vcd_decoding = model.vcd_decoding
        else:
            vcd_decoding = False
        if hasattr(model, "cd_alpha"):
            cd_alpha = model.cd_alpha
        else:
            cd_alpha = -1.0
        model_info = {"temperature": temperature, "max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "alt-text": alt_text, "contrastive": contrastive, "cd_alpha": cd_alpha, "opera_decoding": opera_decoding, "vcd_decoding": vcd_decoding}
        return model_info

    def save_result(self, image_id, question, gt_ans, pred_ans, raw_pred, model_info, mmvp_gpt_grade="", category_pope_mme=""):
        if not self._check_file_existence(self.filename):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(model_info) + "\n")
            self.cur_reviews = self._read_lines()
        else:
            for key, value in model_info.items():
                if key in self.cur_reviews[0]:
                    assert self.cur_reviews[0][key] == value, "{} is different from the previous one".format(key)

        result_dict = {}
        result_dict['image_id'] = image_id
        result_dict['question'] = question
        result_dict['gt_ans'] = gt_ans
        result_dict['pred_ans'] = pred_ans
        result_dict['raw_pred'] = raw_pred
        result_dict['gpt_grade'] = mmvp_gpt_grade
        result_dict['category'] = category_pope_mme
        with open(self.filepath, "a", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False)
            f.write("\n") 
    
    def _read_lines(self):
        if os.path.isfile(os.path.expanduser(self.filepath)):
            cur_reviews = [json.loads(line) for line in open(os.path.expanduser(self.filepath))]
        else:
            cur_reviews = []
        return cur_reviews
    
    def is_absent_sample(self, idx):
        return idx + 1 >= len(self.cur_reviews) 
    
    def get_results(self, row, item_name):
        return self.cur_reviews[row + 1][item_name]
    
    def save_evaluation(self, evaluation):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(evaluation) + "\n")