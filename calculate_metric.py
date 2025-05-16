# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-only
import os
import re
import tempfile
from functools import partial

import pandas as pd
import numpy as np
from utils import * 



def evaluate(eval_file):
    import pandas as pd

    data = pd.read_excel(eval_file)

    # initialize three dataframes for DocVQA, ChartQA, and TableVQA
    DocVQA_df = pd.DataFrame(columns=data.columns)
    CharQA_df = pd.DataFrame(columns=data.columns)
    TableVQA_df = pd.DataFrame(columns=data.columns)

    for i in range(len(data)):
        line = data.iloc[i]
        benchmark_name = line["index"].split("-")[0]
        if benchmark_name == "DocVQA":
            DocVQA_df = DocVQA_df.append(line, ignore_index=True)
        elif benchmark_name == "ChartQA":
            CharQA_df = CharQA_df.append(line, ignore_index=True)
        elif benchmark_name == "TableVQA":
            TableVQA_df = TableVQA_df.append(line, ignore_index=True)
        else:
            raise ValueError(f"Unknown benchmark name {benchmark_name}")

    # calculate three subset separately
    # 1. DocVQA
    data = DocVQA_df
    assert 'answer' in data and 'prediction' in data
    data['prediction'] = [str(x) for x in data['prediction']]
    data['answer'] = [str(x) for x in data['answer']]
    lt = len(data)
    pool = mp.Pool(16)
    lines = [data.iloc[i] for i in range(lt)]
    DocVQA_res = pool.map(partial(process_line_WildDoc, method='anls'), lines)
    hit = hit_calculate(DocVQA_res, "DocVQA")
    DocVQA_overall = np.mean(hit) * 100
    DocVQA_consistency_score = calculate_consistency_WildDoc(DocVQA_res)

    # 2. ChartQA
    data = CharQA_df
    assert 'answer' in data and 'prediction' in data
    data['prediction'] = [str(x) for x in data['prediction']]
    data['answer'] = [str(x) for x in data['answer']]
    lt = len(data)
    pool = mp.Pool(16)
    lines = [data.iloc[i] for i in range(lt)]
    ChartQA_res = pool.map(partial(process_line_WildDoc, method='relaxed_accuracy'), lines)
    hit = hit_calculate(ChartQA_res, "ChartQA")
    ChartQA_overall = np.mean(hit) * 100
    ChartQA_consistency_score = calculate_consistency_WildDoc(ChartQA_res)
    
    # 3. TableVQA
    data = TableVQA_df
    assert 'answer' in data and 'prediction' in data
    import pandas as pd

    data['prediction'] = data['prediction'].str.replace('^Answer: ', '', regex=True)
    ##################
    TableVQA_res = {"fintabnetqa": [], "vtabfact": [], "vwtq": [], "vwtq_syn": []}
    lt = len(data)
    for i in range(lt):
        line = data.iloc[i]
        ret = {'index':line["index"]}
        ans = line['answer']
        pred = line["prediction"]
        
        subset = line["index"].split("-")[1]
        if subset == "fintabnetqa":
            pred, preds = fintabnet_normalize(pred)
            gt, gts = fintabnet_normalize(ans)
            correct = 1 if gt == pred or any(_pred == _gt for _pred in preds for _gt in gts) else 0
        elif subset == "vtabfact":
            pred = pred.lower()
            gt = ans
            if 'true' in pred and 'false' in pred:
                correct = 0
            elif 'true' in pred and gt == '1':
                correct = 1
            elif 'false' in pred and gt == '0':
                correct = 1
            else:
                correct = 0
        elif subset == "vwtq_syn" or subset =="vwtq":
            pred = str(pred).replace('||', '|')
            if pred == "nan":
                pred = ""
            gt = ans
            original_strings = tsv_unescape_list(gt)
            target_values = to_value_list(original_strings)

            predicted_strings = tsv_unescape_list(pred)
            predicted_values = to_value_list(predicted_strings)
            correct = 1 if check_denotation(target_values, predicted_values) else 0
        ret["pred"] = pred
        ret["gt"] = gt
        ret["match"] = correct
        TableVQA_res[subset].append(ret)
    
    # subset of TableVQA
    # TableVQA_subset = [np.mean(hit_calculate(x, "ChartQA")) for x in TableVQA_res.values()]
    TableVQA_overall = np.mean([np.mean(hit_calculate(x, "TableVQA")) for x in TableVQA_res.values()]) * 100 
    TableVQA_consistency_score = np.mean([calculate_consistency_WildDoc(x) for x in TableVQA_res.values()])

    # 4. Overall
    WildDoc_res = DocVQA_res + ChartQA_res + TableVQA_res['fintabnetqa'] + TableVQA_res['vtabfact'] + TableVQA_res['vwtq'] + TableVQA_res['vwtq_syn']
    WildDoc_consistency = calculate_consistency_WildDoc(WildDoc_res)
    WildDoc_overall = calculate_overall_accuracy_WildDoc(WildDoc_res)

    eval_results = {
        "DocVQA": {
            "Overall": DocVQA_overall,
            "Consistency": DocVQA_consistency_score
        },
        "ChartQA": {
            "Overall": ChartQA_overall,
            "Consistency": ChartQA_consistency_score
        },
        "TableVQA": {
            "Overall": TableVQA_overall,
            "Consistency": TableVQA_consistency_score
        },
        "WildDoc": {
            "Overall": WildDoc_overall,
            "Consistency": WildDoc_consistency
        }
    }

    # 转换为长格式DataFrame
    ret_df = pd.DataFrame([
        {"Task": task, "Metric": metric, "Score": score}
        for task, metrics in eval_results.items()
        for metric, score in metrics.items()
    ])

    print(eval_results)

    return ret_df

if __name__ == '__main__':
    # 命令行获取eval_file参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default='./results/Qwen2.5-VL-72B-Instruct_WildDoc.xlsx')
    args = parser.parse_args()
    ret = evaluate(eval_file = args.eval_file)
    print(ret)