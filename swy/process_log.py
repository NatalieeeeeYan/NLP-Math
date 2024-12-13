import json

# 定义日志文件和JSONL文件路径
log_file_path = "/root/autodl-tmp/nlp-math/log/test_sft_lora1212.log"
jsonl_file_path = "dataset/gsm8k/test.jsonl"
output_records = []

# 提取日志文件中的 Question 和 Answer
def extract_log_data(log_file):
    extracted_data = []
    with open(log_file, "r", encoding="utf-8") as file:
        current_question = None
        current_answer = None
        
        for line in file:
            if line.startswith("Question:"):
                if current_question and current_answer:
                    extracted_data.append({"question": current_question.strip(), "prediction": current_answer.strip()})
                current_question = line[len("Question:"):].strip()
                current_answer = ""
            elif line.startswith("Model's Answer:"):
                continue
            else:
                if current_answer is not None:
                    current_answer += line.strip() + " "
        
        # 添加最后一个 question-answer 对
        if current_question and current_answer:
            extracted_data.append({"question": current_question.strip(), "prediction": current_answer.strip()})
    
    return extracted_data

# 从 JSONL 文件读取 ground-truth 数据
def load_ground_truth(jsonl_file):
    ground_truth = {}
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            ground_truth[record["question"].strip()] = record["answer"].strip()
    return ground_truth

# 读取日志数据和 ground-truth 数据
log_data = extract_log_data(log_file_path)
ground_truth_data = load_ground_truth(jsonl_file_path)

# 将日志数据和 ground-truth 数据合并
for record in log_data:
    question = record["question"]
    prediction = record["prediction"]
    answer = ground_truth_data.get(question, None)
    if answer is not None:
        output_records.append({"question": question, "ground_truth": answer, "prediction": prediction})

# 打印或保存结果
output_file_path = "results/qw25_05_sft_lora.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(output_records, output_file, indent=4, ensure_ascii=False)

print(f"Extracted {len(output_records)} records and saved to {output_file_path}.")
