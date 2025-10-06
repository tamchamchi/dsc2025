# %%
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
NUM_CLASSES = 3 # number of classes in the csv

max_seq_length = 2048+1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

# model_name = "unsloth/Llama-3.2-3B-Instruct";load_in_4bit = False
model_name = "unsloth/Qwen2.5-7B";load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,load_in_4bit = load_in_4bit,
    max_seq_length = max_seq_length,
    dtype = dtype,
)

number_token_ids = []
for i in range(0, NUM_CLASSES):
    number_token_ids.append(tokenizer.encode(str(i), add_special_tokens=False)[0])
# keep only the number tokens from lm_head
par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])

old_shape = model.lm_head.weight.shape
old_size = old_shape[0]
print(par.shape)
print(old_shape)

model.lm_head.weight = par

print(f"Đã thu nhỏ lm_head của mô hình về kích thước: {model.lm_head.weight.shape}")
model = PeftModel.from_pretrained(model, "./src/checkpoint/Qwen2.5-7B")

print("Model loaded successfully.")


# %%
# Save the current (trimmed) lm_head and bias
trimmed_lm_head = model.lm_head.weight.data.clone()
trimmed_lm_head_bias = model.lm_head.bias.data.clone() if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None else torch.zeros(len(number_token_ids), device=trimmed_lm_head.device)

# Create a new lm_head with shape [old_size, hidden_dim]
hidden_dim = trimmed_lm_head.shape[1]
new_lm_head = torch.full((old_size, hidden_dim), 0, dtype=trimmed_lm_head.dtype, device=trimmed_lm_head.device)
new_lm_head_bias = torch.full((old_size,), -1000.0, dtype=trimmed_lm_head_bias.dtype, device=trimmed_lm_head_bias.device)

# Fill in the weights and bias for the allowed tokens (number_token_ids)
for new_idx, orig_token_id in enumerate(number_token_ids):
    new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
    new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]

# Update the model's lm_head weight and bias
with torch.no_grad():
    new_lm_head_module = torch.nn.Linear(hidden_dim, old_size, bias=True, device=model.device)
    new_lm_head_module.weight.data.copy_(new_lm_head)
    new_lm_head_module.bias.data.copy_(new_lm_head_bias)
    model.lm_head.modules_to_save["default"] = new_lm_head_module

print(f"Remade lm_head: shape = {model.lm_head.weight.shape}. Allowed tokens: {number_token_ids}")

# %%
prompt = """### INSTRUCTION ###
You are a fact-checking system.  
Classify the "Response" to the "Question" into one of three classes:  
- class 0: no → Correct answer, consistent with context, no extra info.  
- class 1: intrinsic → Contradicts or distorts context info.  
- class 2: extrinsic → Adds info not in context.  
### INPUT ###
Context: {context}  
Question: {question}  
Response: {response}  
### OUTPUT ###
The correct answer is: class {label}"""

# %%
# THAY THẾ TOÀN BỘ CELL 7 BẰNG ĐOẠN MÃ NÀY

# 1. Tạo một dictionary để ánh xạ nhãn chữ sang số
label_map = {
    'no': 0,
    'intrinsic': 1,
    'extrinsic': 2
}

# 2. Định nghĩa lại hàm formatting với việc sử dụng map và strip()
def formatting_prompts_func(dataset_, prompt_template):
    texts = []
    for index, row in dataset_.iterrows():
        # Lấy dữ liệu và strip() để loại bỏ khoảng trắng thừa
        context_ = str(row.get('context', '')).strip()
        question_ = str(row.get('prompt_corrected', '')).strip()
        response_ = str(row.get('response', '')).strip()
        
        # Lấy nhãn chữ và loại bỏ khoảng trắng thừa
        label_text = str(row.get('label', '')).strip()
        
        # *** DÒNG QUAN TRỌNG NHẤT: Ánh xạ nhãn chữ sang nhãn số ***
        label_ = label_map.get(label_text, '') # Dùng .get để tránh lỗi nếu có nhãn lạ
        
        # Điền thông tin vào khuôn mẫu
        text = prompt_template.format(
            context=context_, 
            question=question_, 
            response=response_, 
            label=label_
        )
        
        texts.append(text)
        
    return texts

# %%
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- Bước 1: Chuẩn bị ---

# Định nghĩa tên file input và output
input_file_path = './src/data/vihallu-private-test.csv'
output_file_path = 'qwen2.5-7b'

# Tải dữ liệu từ file test
try:
    test_df = pd.read_csv(input_file_path)
    print(f"Đã tải thành công file '{input_file_path}' với {len(test_df)} hàng.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{input_file_path}'. Vui lòng kiểm tra lại đường dẫn.")
    # Dừng thực thi cell nếu không tìm thấy file
    assert False, "File test không tồn tại."

# Tạo lại dictionary ánh xạ từ ID số sang nhãn chữ (quan trọng để ra đúng định dạng)
id_to_label = {
    0: 'no',
    1: 'intrinsic',
    2: 'extrinsic'
}

# Sử dụng lại prompt template cho inference
# Đảm bảo biến 'prompt' từ các cell trước vẫn tồn tại
inference_prompt_template = prompt.rsplit("class {label}", 1)[0] + "class "
# inference_prompt_template = prompt.rsplit("lớp {label}", 1)[0] + "lớp "

# --- Bước 2: Chạy dự đoán trên tập test ---

# Khởi tạo các biến
predictions = []
batch_size = 1  # Có thể tăng batch size để inference nhanh hơn
device = model.device

# Đặt mô hình ở chế độ đánh giá (quan trọng!)
model.eval()

with torch.inference_mode():
    for i in tqdm(range(0, len(test_df), batch_size), desc="Đang dự đoán trên file test"):
        batch_df = test_df.iloc[i:i+batch_size]
        
        # Lấy ID của các hàng trong batch
        batch_ids = batch_df['id'].tolist()
        
        # Tạo prompts cho batch hiện tại
        prompts = []
        for _, row in batch_df.iterrows():
            prompts.append(
                inference_prompt_template.format(
                    context=str(row.get('context', '')),
                    question=str(row.get('prompt', '')),
                    response=str(row.get('response', ''))
                )
            )

        # Tokenize và đưa lên GPU
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)
        
        # Lấy logits từ mô hình
        logits = model(**inputs).logits
        
        # Chỉ lấy logits của token cuối cùng
        last_idxs = inputs.attention_mask.sum(1) - 1
        last_logits = logits[torch.arange(len(batch_df)), last_idxs, :]
        
        # Lấy xác suất chỉ cho các token số ("0", "1", "2", "3")
        probs = F.softmax(last_logits, dim=-1)[:, number_token_ids]
        
        # Lấy ID dự đoán (0, 1, 2, hoặc 3)
        predicted_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        
        # Chuyển đổi ID dự đoán sang nhãn chữ và lưu kết quả
        for j in range(len(batch_df)):
            pred_id = predicted_ids[j]
            pred_label = id_to_label.get(pred_id, "N/A") # Dùng .get để an toàn
            
            predictions.append({
                'id': batch_ids[j],
                'predict_label': pred_label
            })

# --- Bước 3: Tạo và lưu file submission ---

# Chuyển danh sách kết quả thành DataFrame
submission_df = pd.DataFrame(predictions)

# Lưu DataFrame ra file CSV, không bao gồm cột index
submission_df.to_csv(output_file_path, index=False)

print(f"\n✅ Đã tạo file submission thành công tại: '{output_file_path}'")
print("Xem trước 5 dòng đầu của file submission:")
print(submission_df.head())