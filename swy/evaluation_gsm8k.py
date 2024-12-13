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
    model_reply = model_reply.replace('\\', '').replace('\n', '').lower()

    # 提取包含"final answer"的部分
    final_ans = model_reply[model_reply.find('final answer'):] if 'final answer' in model_reply else ''

    # 查找包含 therefore/so 等词的句子
    result_sentence = None
    for sentence in model_reply.split('. '):
        if any(word in sentence for word in ['therefore', 'so', 'thus', 'hence']):
            result_sentence = sentence

    if result_sentence:
        # 提取 result_sentence 中的数字
        possible_numbers = re.findall(r'\d+\.?\d*', convert_words_to_numbers(result_sentence))
        final_ans += ' ' + ' '.join(possible_numbers)

    # 使用正则表达式去除所有字母和符号，仅保留数字和英文数字
    final_ans = re.sub(r'[^a-z0-9\s]', '', final_ans)  # 去除非字母、数字和空格的字符
    
    # 将英文数字转换为阿拉伯数字
    final_ans = convert_words_to_numbers(final_ans)
    final_ans = final_ans.strip()  # 去掉两端的空格
    print("Filtered final ans (with numbers and English words):", final_ans)

    final_ans = re.sub(r'[^0-9\s]', '', final_ans)  # 去除非数字和空格的字符
    final_ans = final_ans.strip().split()
    print("Filtered final ans (only numbers):", final_ans)
    final_ans = [float(num) for num in final_ans]  # 显式转换为浮动类型
    return final_ans



def extract_ground_truth(ground_truth: str) -> float: 
    '''
    提取ground truth

    @param ground_truth: 真实答案(str)

    @return 提取出的真是答案(float)
    '''
    ground_truth = ground_truth.lower().split('#')[-1].replace(' ', '').replace(',', '')
    print('ground truth:', ground_truth, '\n')
    try:
        return float(ground_truth)
    except ValueError:
        # 如果是英文数字（如 "twenty four"），尝试转换为阿拉伯数字
        # return w2n.word_to_num(ground_truth)
        gt = convert_words_to_numbers(ground_truth)
        gt = gt.strip()
        return float(gt)


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
        if any(abs(output - true_answer) < 1e-6 for output in outputs):
            correct_count += 1
    
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
                    gts.append(extract_ground_truth(question['answer']))
                    break
    
    accuracy = calculate_accuracy(ground_truth=gts, model_outputs=ans)
    print(f'Accuracy: {accuracy}')
