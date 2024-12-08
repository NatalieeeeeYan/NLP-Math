import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load the model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_path = './model/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
print("Model loaded successfully!")

# Define the prompt for generating math reasoning and final answer
def generate_math_prompt(question: str) -> str:
    '''
    @param: question The math question to be solved
    
    @return: prompt The formatted prompt for the model
    '''
    prompt = (
        f"Please solve the following math problem step by step and provide the final answer. "
        "Do not add any extra output. The final answer should be clearly marked with ####<answer>.\n\n"
        f"Question: {question}\n\n"
    )
    return prompt

# Inference function to solve math problems
def solve_math_with_llm(data: list) -> list:
    '''
    @param: data A list of instances, each containing a math question and its correct answer.
    
    @return: predictions A list of instances with the math question, correct answer, and predicted result.
    '''
    predictions = []
    for d in data:
        question = d['question']
        correct_answer = d['answer']
        prompt = generate_math_prompt(question)
        
        # Generate the prompt and get the model's output
        messages = [
            {"role": "system", "content": "You are a math assistant who solves problems step by step."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        # Decode the output
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract the reasoning and final answer from the model output
        # reasoning, final_answer = output.split('####<answer>') if '####<answer>' in output else (output, 'No answer found')
        
        # Clean up the reasoning part (if any)
        # reasoning = reasoning.strip()
        # final_answer = final_answer.strip()
        
        print("Question:", question)
        # print("Reasoning:", reasoning)
        print("Model's Answer:", output)
        
        # Save the results
        predictions.append({
            'question': question,
            'ground_truth': correct_answer,
            'prediction': output
        })
    return predictions


# Load the dataset (assuming it's in a JSONL format with 'question' and 'answer' keys)
data = []
with open("./dataset/gsm8k/test.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Get the predictions
predicted_results = solve_math_with_llm(data)

# Save the results to a JSON file for later analysis
with open('./results1207.json', 'w', encoding='utf-8') as f:
    json.dump(predicted_results, f, ensure_ascii=False, indent=4)

print("Results saved successfully!")