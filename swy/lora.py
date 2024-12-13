import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from tqdm import tqdm
import json


# 加载模型和分词器
model_name = "Qwen2.5-Math-1.5B"
model_path = './models/Qwen2.5-Math-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# 使用 LoRA 进行微调
lora_config = LoraConfig(
    r=8,  # 低秩适配维度
    lora_alpha=32,  # LoRA的缩放因子
    lora_dropout=0.1,  # LoRA dropout比例
    target_modules=["q_proj", "v_proj"]  # LoRA应用的层
)

# 使用 LoRA 配置更新模型
model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
print("Model loaded and LoRA applied successfully!")


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
        
        # 生成输入文本
        prompt = f"Please solve the following math problem step by step and provide the final answer. The final answer should be clearly marked with ####<answer>.\n\nQuestion: {question}\n\n"
        
        # 对问题和答案进行编码
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt").input_ids
        
        # 返回输入和标签
        return {'input_ids': inputs['input_ids'].squeeze(), 'labels': labels.squeeze()}

# 加载数据集
data = []
with open("./dataset/gsm8k/train.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 划分训练集和验证集
dataset = MathDataset(data, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-6)

# 验证过程
def evaluate_model(model, dataloader):
    model.eval()  # 切换到评估模式
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False, ncols=100):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 训练过程
def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=5):
    best_val_loss = float('inf')  # 初始化最佳验证损失
    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # 在验证集上评估
        val_loss = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained('./models/qwen2.5_sft_lora_best')
            tokenizer.save_pretrained('./models/qwen2.5_sft_lora_best')
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")


# 微调模型
train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=5)

# 保存最终模型
model.save_pretrained('./models/qwenMath15_sft_lora')
tokenizer.save_pretrained('./models/qwenMath15_sft_lora')
print("Model fine-tuning (sft lora) completed and saved.")
