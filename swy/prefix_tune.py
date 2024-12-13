import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import json

# 加载模型和分词器
model_path = './models/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# 定义 Prefix Tuning 模块
class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=20, dtype=torch.float32):
        super().__init__()
        self.prefix_length = prefix_length
        self.embedding_dim = config.hidden_size
        self.dtype = dtype

        # 初始化前缀嵌入
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.embedding_dim, dtype=self.dtype)
        )

    def forward(self, batch_size, device):
        # 扩展前缀嵌入以匹配批次大小
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return prefix

# 注入 Prefix Tuning 到模型
class PrefixTunedModel(nn.Module):
    def __init__(self, base_model, prefix_tuning):
        super().__init__()
        self.base_model = base_model
        self.prefix_tuning = prefix_tuning

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.size(0), input_ids.size(1)
        device = input_ids.device

        # 获取前缀嵌入
        prefix_embeddings = self.prefix_tuning(batch_size, device)

        # 获取模型原始的词嵌入
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)

        # 拼接前缀嵌入和输入嵌入
        extended_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)

        # 调整 attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.prefix_tuning.prefix_length, device=device)
            extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None

        # 调整 labels 长度
        if labels is not None:
            # 用 -100 填充前缀部分，忽略这些位置的损失
            prefix_labels = torch.full((batch_size, self.prefix_tuning.prefix_length), -100, device=device)
            extended_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            extended_labels = None

        # 调用模型的 forward 方法
        outputs = self.base_model(
            inputs_embeds=extended_embeddings,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )
        return outputs

# 初始化 Prefix Tuning
prefix_tuning = PrefixTuning(base_model.config, prefix_length=20, dtype=base_model.get_input_embeddings().weight.dtype)
model = PrefixTunedModel(base_model, prefix_tuning)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
print("Model loaded with injected Prefix Tuning!")

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
    output_dir='./qwen_prefix_tuning_results',  # 模型保存路径
    evaluation_strategy="epoch",  # 每个epoch后评估一次
    learning_rate=5e-5,  # 学习率
    per_device_train_batch_size=4,  # 每个设备上的训练批次大小
    per_device_eval_batch_size=4,  # 每个设备上的评估批次大小
    num_train_epochs=3,  # 训练的epoch数
    save_strategy="epoch",  # 每个epoch后保存模型
    save_total_limit=2,  # 最多保存2个检查点
    report_to="none",  # 禁用默认的日志记录工具
    load_best_model_at_end=True,  # 使用验证集表现最好的模型
)

# 定义 Trainer
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
torch.save(model.state_dict(), './models/qwen_prefix_tuning_model.pt')
tokenizer.save_pretrained('./models/qwen_prefix_tuning_tokenizer')
print("Model fine-tuning with manual Prefix Tuning completed and saved.")
 