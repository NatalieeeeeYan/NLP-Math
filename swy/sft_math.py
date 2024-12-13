import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import json

# 加载模型和分词器
model_name = "Qwen2.5-0.5B"
model_path = './models/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

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
        question = item['problem']
        solution = item['solution']
        
        # 构造输入文本：问题 + 解题步骤
        prompt = (
            f"Please solve the following math problem step by step:\n\n"
            f"Question: {question}\n\nSolution: {solution}\n"
        )
        
        # 对问题和答案进行编码
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = inputs.input_ids.clone()  # 使用输入文本作为标签

        # 忽略填充token的损失
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {'input_ids': inputs['input_ids'].squeeze(), 'labels': labels.squeeze()}

# 加载训练数据
data = []
with open("./dataset/MATH/train.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 创建数据集并划分为训练集和验证集
dataset = MathDataset(data, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./qwen2505_sft_finetuned_MATH',  # 模型保存路径
    evaluation_strategy="epoch",  # 每个epoch后评估一次
    learning_rate=5e-6,  # 学习率
    per_device_train_batch_size=4,  # 每个设备上的训练批次大小
    per_device_eval_batch_size=4,  # 每个设备上的评估批次大小
    num_train_epochs=3,  # 训练的epoch数
    save_strategy="epoch",  # 每个epoch后保存模型
    save_total_limit=2,  # 最多保存2个检查点
    report_to="none",  # 禁用默认的日志记录工具
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
print(f"len of training set: {len(trainer.train_dataset)}\tvalidation set: {len(trainer.eval_dataset)}")

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model('./models/qwen2505_sft_finetuned_MATH')
tokenizer.save_pretrained('./models/qwen2505_sft_finetuned_MATH')
print("Model fine-tuning completed and saved.")

