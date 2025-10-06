from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm.auto import tqdm

max_seq_length = 1024*3
load_in_4bit=False
NUM_CLASSES = 3

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

label_to_id = {
    'no': 0,
    'intrinsic': 1,
    'extrinsic': 2
}

id_to_label = {v: k for k, v in label_to_id.items()}

def init_model(model_name, checkpoint):

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/{model_name}",
    load_in_4bit = load_in_4bit,
    max_seq_length = max_seq_length,
    cache_dir="/mnt/mmlab2024nas/anhndt"
    )

    number_token_ids = []
    for i in range(0, NUM_CLASSES):
        number_token_ids.append(tokenizer.encode(str(i), add_special_tokens=False)[0])
    # keep only the number tokens from lm_head
    par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])

    old_shape = model.lm_head.weight.shape
    old_size = old_shape[0]

    model.lm_head.weight = par

    model = PeftModel.from_pretrained(model, checkpoint)

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

    return model, tokenizer, number_token_ids

def formatting_prompts_func(dataset_, prompt_template):
    texts = []
    for index, row in dataset_.iterrows():
        # Lấy dữ liệu và strip() để loại bỏ khoảng trắng thừa
        context_ = str(row.get('context', '')).strip()
        question_ = str(row.get('prompt', '')).strip()
        response_ = str(row.get('response', '')).strip()
        
        # Lấy nhãn chữ và loại bỏ khoảng trắng thừa
        label_text = str(row.get('label', '')).strip()
        
        # *** DÒNG QUAN TRỌNG NHẤT: Ánh xạ nhãn chữ sang nhãn số ***
        label_ = label_to_id.get(label_text, '') # Dùng .get để tránh lỗi nếu có nhãn lạ
        
        # Điền thông tin vào khuôn mẫu
        text = prompt_template.format(
            context=context_, 
            question=question_, 
            response=response_, 
            label=label_
        )
        
        texts.append(text)
        
    return texts

def evaluate_and_save_probabilities(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    prompt_template: str,
    number_token_ids: list[int],
    output_csv_path: str,
    is_private_test: bool = False
) -> None:
    """
    Đánh giá mô hình, trích xuất xác suất và lưu kết quả vào file CSV.
    Nếu is_private_test=True, file CSV sẽ chứa 'id' và các cột xác suất.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu để đánh giá.
                           Nếu is_private_test=False, phải có: 'id', 'label', 'context', 'prompt', 'response'.
                           Nếu is_private_test=True, phải có: 'id', 'context', 'prompt', 'response'.
        model: Mô hình đã được huấn luyện.
        tokenizer: Tokenizer tương ứng.
        device (torch.device): Thiết bị chạy mô hình ('cpu' hoặc 'cuda').
        batch_size (int): Kích thước batch.
        prompt_template (str): Mẫu prompt.
        number_token_ids (List[int]): Danh sách ID của các token lớp.
        output_csv_path (str): Đường dẫn lưu file CSV.
        is_private_test (bool): Nếu True, chỉ lưu 'id' và xác suất. Mặc định là False.
    """
    # --- Bước 1: Ánh xạ, kiểm tra và chuẩn bị ---
    prob_column_names = ['prob_no', 'prob_intrinsic', 'prob_extrinsic']
    
    # Cập nhật các cột cần thiết dựa trên cờ
    if is_private_test:
        required_cols = ['id', 'context', 'prompt', 'response']
        print("Running in PRIVATE TEST mode. Output will contain 'id' and probabilities.")
    else:
        required_cols = ['id', 'label', 'context', 'prompt', 'response']

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain the following columns for this mode: {required_cols}")

    eval_df = df.copy()

    print("Sorting dataset by token length for efficient batching...")
    eval_df['token_length'] = eval_df.apply(
        lambda row: len(tokenizer.encode(
            prompt_template.format(
                context=row["context"],
                question=row["prompt"],
                response=row["response"]
            ), add_special_tokens=True
        )),
        axis=1
    )
    val_df_sorted = eval_df.sort_values(by='token_length').reset_index(drop=True)
    print("Sorting complete.")

    # --- Bước 2: Chạy vòng lặp đánh giá ---
    all_results = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(val_df_sorted), batch_size), desc="Evaluating"):
            batch_df = val_df_sorted.iloc[i:i+batch_size]
            
            prompts = [
                prompt_template.format(
                    context=row["context"],
                    question=row["prompt"],
                    response=row["response"]
                )
                for _, row in batch_df.iterrows()
            ]

            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=tokenizer.model_max_length
            ).to(device)
            
            logits = model(**inputs).logits
            
            last_idxs = inputs.attention_mask.sum(1) - 1
            last_logits = logits[torch.arange(len(batch_df)), last_idxs, :]
            
            target_logits = last_logits[:, number_token_ids]
            target_probs = F.softmax(target_logits, dim=1).float().cpu().numpy()
            
            # Luôn lấy ID vì cả hai chế độ đều cần nó
            ids = batch_df['id'].tolist()
            if not is_private_test:
                true_labels = batch_df['label'].tolist()

            for j in range(len(batch_df)):
                # Luôn bắt đầu dictionary kết quả với ID
                result_row = {"id": ids[j]}
                
                # Chỉ thêm 'label' nếu không phải private test
                if not is_private_test:
                    result_row["label"] = true_labels[j]

                # Luôn thêm các cột xác suất
                for k, col_name in enumerate(prob_column_names):
                    result_row[col_name] = target_probs[j, k]
                
                all_results.append(result_row)

    # --- Bước 3: Tạo DataFrame và lưu file ---
    results_df = pd.DataFrame(all_results)
    
    # Luôn sắp xếp theo ID để đảm bảo thứ tự nhất quán
    results_df = results_df.sort_values(by='id').reset_index(drop=True)
    
    results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu kết quả đánh giá vào file: '{output_csv_path}'")
    print(f"Tổng số mẫu đã xử lý: {len(results_df)}")

if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Fact-checking inference script")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Tên mô hình HuggingFace, ví dụ: Qwen3-4B-Base")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint, ví dụ: Qwen3-4B-Base")
    parser.add_argument("--train_path", type=str, default="/src/data/vihallu-train.csv",
                        help="Đường dẫn file CSV train")
    parser.add_argument("--private_path", type=str, default="/src/data/vihallu-private-test.csv",
                        help="Đường dẫn file CSV private test")
    parser.add_argument("--save_dir", type=str, default="/src/data/prod",
                        help="Thư mục lưu output CSV")

    args = parser.parse_args()

    model_name = args.model_name
    checkpoint = args.checkpoint
    inference_prompt_template = prompt.rsplit("class {label}", 1)[0] + "class "

    # Load model
    model, tokenizer, number_token_ids = init_model(model_name, checkpoint)

    # Tạo thư mục nếu chưa có
    os.makedirs(args.save_dir, exist_ok=True)

    # Đọc dữ liệu train
    df = pd.read_csv(args.train_path)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=36)

    train_output = os.path.join(args.save_dir, f"train_prod_{model_name.replace('/', '_')}.csv")
    eval_output = os.path.join(args.save_dir, f"eval_prod_{model_name.replace('/', '_')}.csv")

    # Train set
    evaluate_and_save_probabilities(df=train_df, 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=model.device, 
                                    prompt_template=inference_prompt_template, 
                                    batch_size=1, 
                                    number_token_ids=number_token_ids,
                                    output_csv_path=train_output,
                                    is_private_test=False
                                    )
    # Eval set
    evaluate_and_save_probabilities(df=eval_df, 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=model.device, 
                                    prompt_template=inference_prompt_template, 
                                    batch_size=1, 
                                    number_token_ids=number_token_ids,
                                    output_csv_path=eval_output,
                                    is_private_test=False
                                    )

    # Private test
    private_df = pd.read_csv(args.private_path)
    private_output = os.path.join(args.save_dir, f"private_prod_{model_name.replace('/', '_')}.csv")

    evaluate_and_save_probabilities(df=private_df, 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=model.device, 
                                    prompt_template=inference_prompt_template, 
                                    batch_size=1, 
                                    number_token_ids=number_token_ids,
                                    output_csv_path=private_output,
                                    is_private_test=True
                                    )


