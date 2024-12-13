import os
import json
import re
import argparse
# from word2number import w2n


prompt_sys = 'You are a math assistant who solves problems step by step.'
prompt = (
        f"Please solve the following math problem step by step and provide the final answer. "
        "Do not add any extra output. The final answer should be clearly marked with ####<answer>"
    )


# 英文数字到数字的映射
number_words = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", 
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"
}


def convert_words_to_numbers(text: str) -> str:
    '''
    提取英文数字并转换成数字

    @param text: 给出的文字

    @return 提取出的数字(str)
    '''
    for word, num in number_words.items():
        text = re.sub(r'\b' + word + r'\b', num, text)  # 匹配单词并替换
    return text


def extract_finalans(model_reply: str) -> list:
    '''
    提取模型回答中的正确答案

    @param model_reply: 模型的回答(str)

    @return 提取到的回答中的数字（可能的回答）(list[float])
    '''
    model_reply = model_reply.replace(prompt_sys, '').replace(prompt, '').replace('systemuser.question:', '')
    moodel_reply = model_reply.replace('\\', '').replace('\n', '').lower()
    final_ans = model_reply

    # print('model_reply:', model_reply)
    # 提取结论句
    if 'boxed' in final_ans:
        final_ans = final_ans[final_ans.find('boxed') + len('boxed{') : final_ans.rindex('}')]
    else:
        for sentence in moodel_reply.split('. '):
            if any(word in sentence for word in ['therefore', 'so', 'thus', 'hence', 'final']):
                final_ans = sentence
    
    if 'text' in final_ans:
        final_ans = final_ans[final_ans.find('text') + len('text{') : final_ans.rindex('}')]

    return final_ans.lower().strip().replace(' ', '')


def extract_ground_truth(ground_truth: str) -> float: 
    '''
    提取ground truth

    @param ground_truth: 真实答案(str)

    @return 提取出的真是答案(float)
    '''
    ground_truth = ground_truth.replace('\\', '').lower().strip().replace(' ', '')
    if 'text' in ground_truth:
        ground_truth = ground_truth[ground_truth.find('text') + len('text{') : ground_truth.rindex('}')]
    if '^circ' in ground_truth:
        ground_truth = ground_truth.replace('^circ', '')
    if 'dfrac' in ground_truth:
        ground_truth = ground_truth.replace('dfrac', 'frac')
    if '$' in ground_truth:
        ground_truth = ground_truth.replace('$', '')
    if 'xin' in ground_truth:
        ground_truth = ground_truth.replace('x in', '')
    if 'x=' in ground_truth:
        ground_truth = ground_truth.replace('x=', '')
    # print('ground truth:', ground_truth, '\n')
    return ground_truth


def calculate_accuracy(ground_truth: list[float], model_outputs: list[list[int]]) -> float:
    """
    计算准确率：如果模型的预测列表中有正确的答案，则认为这道题目是正确的。
    
    @param ground_truth: 真实答案列表 (list[float])
    @param model_outputs: 模型输出的预测列表 (list[list[int]]), 每个列表包含一个模型对该题的多个候选答案
    
    @return: 准确率(float)
    """
    correct_count = 0  # 记录正确的预测数量
    
    for true_answer, outputs in zip(ground_truth, model_outputs):
        # 检查模型输出列表中是否包含正确答案
        print(f'outputs: {outputs}')
        print(f'true_answer: {true_answer}')
        if true_answer in outputs:
            correct_count += 1
            print('Correct\n')
        else:
            print('Incorrect\n')
    
    # 计算准确率
    accuracy = correct_count / len(ground_truth) if len(ground_truth) > 0 else 0
    return accuracy


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Run segmentation on test images")
    parser.add_argument('--result', type=str, help='Path to the result') 
    parser.add_argument('--gt', type=str, help='Path to the ground truth', default=None) 
    args = parser.parse_args()
    
    # 支持json和jsonl两种格式的结果文件
    if args.result.endswith('.jsonl'):
        with open(args.result, 'r') as f:
            results = [json.loads(line) for line in f]
    else:
        with open(args.result, 'r') as f:
            results = json.load(f)
        
    gts = []
    ans = []
    for res in results:
        ans.append(extract_finalans(res['prediction']))
        if 'ground_truth' in res:       # 如果有ground_truth，直接提取
            gts.append(extract_ground_truth(res['ground_truth']))
        
        else:       # 如果没有ground_truth，尝试从数据文件中读取
            if args.gt is None:
                raise ValueError('Ground truth is missing')
            with open(args.gt, 'r') as f:
                ground_truth_list = [json.loads(line) for line in f]
            for question in ground_truth_list:
                if question['question'] == res['question']:
                    gts.append(extract_ground_truth(res['ground_truth']))
                    break

    accuracy = calculate_accuracy(ground_truth=gts, model_outputs=ans)
    print(f'Accuracy: {accuracy}')
