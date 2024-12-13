import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import json

# 加载模型和分词器
model_name = "Qwen2.5-Math-1.5B"
model_path = './models/Qwen2.5-Math-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# # 预测用
# model_path = './models/qwen2.5_sft_finetuned'
# model_name = 'qwen2.5_sft_finetuned'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
print("Model loaded successfully!")

# 定义数据集类
class MathDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # 构造输入文本
        prompt = (
            f"Please solve the following math problem step by step and provide the final answer. "
            f"The final answer should be clearly marked with ####<answer>.\n\nQuestion: {question}\n\n"
        )
        
        # 对问题和答案进行编码
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt").input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # 忽略填充token的损失
        
        return {'input_ids': inputs['input_ids'].squeeze(), 'labels': labels.squeeze()}

# 加载训练数据
data = []
with open("./dataset/gsm8k/train.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 创建数据集并划分为训练集和验证集
dataset = MathDataset(data, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./qwenMath15_sft_results',  # 模型保存路径
    evaluation_strategy="epoch",  # 每个epoch后评估一次
    learning_rate=5e-6,  # 学习率
    per_device_train_batch_size=4,  # 每个设备上的训练批次大小
    per_device_eval_batch_size=4,  # 每个设备上的评估批次大小
    num_train_epochs=3,  # 训练的epoch数
    save_strategy="epoch",  # 每个epoch后保存模型
    # logging_dir='./logs',  # 日志保存路径
    # logging_steps=10,  # 日志记录间隔
    save_total_limit=2,  # 最多保存2个检查点
    report_to="none",  # 禁用默认的日志记录工具
    # fp16=True,  # 使用混合精度训练
    load_best_model_at_end=True,  # 使用验证集表现最好的模型
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 训练集
    eval_dataset=val_dataset,    # 验证集
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model('./models/qwenMath15_sft_finetuned')
tokenizer.save_pretrained('./models/qwenMath15_sft_finetuned')
print("Model fine-tuning completed and saved.")

# 验证微调后的模型
def solve_math_with_llm(data: list) -> list:
    '''
    使用微调后的模型对数学问题进行推理。
    '''
    predictions = []
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for d in data:
            question = d['question']
            prompt = (
                f"Please solve the following math problem step by step and provide the final answer. "
                f"The final answer should be clearly marked with ####<answer>.\n\nQuestion: {question}\n\n"
            )
            
            # 对输入进行编码
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            
            # 生成模型输出
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 提取最终答案
            final_answer = output.replace(prompt, '').strip()
            print("Question:", question)
            print("Model's Answer:", final_answer)
            
            # 保存结果
            predictions.append({
                'question': question,
                'prediction': final_answer,
                'ground_truth': d['answer']
            })
    return predictions

# 加载测试数据
test_data = []
with open("./dataset/gsm8k/test.jsonl", 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

# 测试微调后的模型
predicted_results = solve_math_with_llm(test_data)

# 保存预测结果
with open('./results/qwenMath15_sft_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(predicted_results, f, ensure_ascii=False, indent=4)

print("Test results saved successfully!")
