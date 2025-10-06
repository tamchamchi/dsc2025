# Source Code Directory - DSC 2025

**🚀 Quick Start cho Ban Tổ Chức:**
```bash
chmod +x all_in_one.sh
./all_in_one.sh
```
➡️ **Output:** `submit.csv` - File submission cuối cùng

## 📦 Model Checkpoints

**🔗 Download Link:** [Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1uXaU4NIpOKnXL6Io-zeKhBaggXvcGjZX?usp=sharing)

**Yêu cầu:** Tải xuống và giải nén các model checkpoints vào thư mục `checkpoint/` trước khi chạy script.

**Cấu trúc checkpoint cần thiết:**
```
checkpoint/
├── Qwen3-4B-Base/          # Qwen3-4B fine-tuned model
├── Qwen2.5-7B-Instruct/    # Qwen2.5-7B fine-tuned model  
├── Qwen2.5-7B/             # Qwen2.5-7B base model
└── ...                     # Các models khác
```

---

Thư mục `src` chứa solution hoàn chỉnh cho cuộc thi DSC 2025. Ban tổ chức chỉ cần chạy script `all_in_one.sh` để tự động tạo ra file submission.

## Cấu trúc thư mục

```
src/
├── ensemble_model.py       # Mô hình ensemble neural network
├── get_prod.py            # Tạo predictions từ các mô hình đã train
├── predict.py             # Script inference cơ bản (empty)
├── predict_2class.py      # Prediction cho classification 2 class
├── predict_qwen2.5_7b.py  # Prediction với mô hình Qwen2.5-7B
├── final_submit.py        # Tạo file submission cuối cùng
├── get_diff_res.py        # So sánh kết quả giữa các mô hình
├── all_in_one.sh          # Script chạy toàn bộ pipeline
├── environment.yml        # Môi trường conda
├── data/                  # Dữ liệu và kết quả
├── checkpoint/            # Model checkpoints
└── notebook/              # Jupyter notebooks
```

## Mô tả các file chính

### 🤖 Model và Prediction

#### `ensemble_model.py`
Implemention mô hình ensemble neural network để kết hợp predictions từ nhiều mô hình.

**Tính năng:**
- Neural network với 2 hidden layers (32 → 16 → num_classes)
- Dropout regularization
- Support multi-class classification
- Training và validation functions

**Sử dụng:**
```python
from ensemble_model import EnsembleNN, train_ensemble

# Tạo model
model = EnsembleNN(input_dim=6, num_classes=3, dropout_p=0.5)

# Training
train_ensemble(features_train, labels_train, features_val, labels_val, num_classes=3)
```

#### `get_prod.py`
Script chính để tạo predictions từ các mô hình đã fine-tune.

**Tính năng:**
- Load pre-trained models (Qwen3-4B, Qwen2.5-7B)
- Batch inference với tqdm progress
- Support multiple model types
- Output prediction probabilities và final labels

**Parameters:**
- `--model_name`: Tên mô hình (Qwen3-4B-Base, Qwen2.5-7B-Instruct, etc.)
- `--checkpoint`: Đường dẫn đến checkpoint
- `--train_path`: File dữ liệu training
- `--private_path`: File test private
- `--save_dir`: Thư mục lưu kết quả

**Prompt template:**
```
### INSTRUCTION ###
You are a fact-checking system.
Classify the "Response" to the "Question" into one of three classes:
- class 0: no → Correct answer, consistent with context
- class 1: intrinsic → Contradicts or distorts context info
- class 2: extrinsic → Adds info not in context
```

#### `predict_2class.py`
Prediction cho task classification 2 class (có thể là binary classification).

#### `predict_qwen2.5_7b.py`
Prediction specialized cho mô hình Qwen2.5-7B.

### 📊 Data Processing và Analysis

#### `get_diff_res.py`
Utility để so sánh kết quả prediction giữa các mô hình khác nhau.

**Tính năng:**
- So sánh 2 file CSV theo ID
- Hiển thị thống kê khác biệt
- Export chi tiết các dòng khác nhau
- Flexible column naming

**Sử dụng:**
```python
from get_diff_res import compare_results

compare_results("model1_result.csv", "model2_result.csv", 
                key_col="id", label_col="predict_label")
```

#### `final_submit.py`
Script tạo file submission cuối cùng bằng cách kết hợp multiple predictions.

**Logic:**
1. Đọc các file submit partial (submit_01.csv, submit_02.csv, submit_12.csv)
2. Concatenate tất cả predictions
3. Merge với Qwen2.5-7B baseline
4. Override predictions cho các ID đã có
5. Save final submission

### 🔄 Pipeline Management

#### `all_in_one.sh` ⭐ **MAIN SCRIPT**
**Script chính cho ban tổ chức** - Chạy toàn bộ pipeline từ inference đến tạo file submission cuối cùng.

**Chức năng:**
- Tự động generate predictions từ tất cả trained models
- Chạy ensemble model để kết hợp kết quả
- So sánh và validate results
- Tạo file submission cuối cùng: `submit.csv`

**Sử dụng:**
```bash
chmod +x all_in_one.sh
./all_in_one.sh
```

**Output:** File `submit.csv` sẵn sàng để nộp bài

**Pipeline flow:**
1. `get_prod.py` → Generate predictions từ multiple models
2. `ensemble_model.py` → Ensemble results  
3. `get_diff_res.py` → Quality check
4. `final_submit.py` → Create final submission

### ⚙️ Environment

#### `environment.yml`
Conda environment configuration với tất cả dependencies cần thiết.

**Key packages:**
- Python 3.13
- PyTorch ecosystem
- Transformers & Unsloth
- Data science stack (pandas, sklearn, etc.)
- GPU support (CUDA 12.4)

## Workflow sử dụng

### 🎯 Cho Ban Tổ Chức (Recommended)
```bash
# Chạy một lệnh duy nhất
./all_in_one.sh

# Kết quả: submit.csv được tạo tự động
```

### 🔧 Cho Development (Chi tiết)

#### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate dsc
```

#### 2. Generate Predictions
```bash
# Từng mô hình riêng lẻ
python get_prod.py --model_name Qwen3-4B-Base --checkpoint ./checkpoint/Qwen3-4B-Base

# Hoặc chạy tất cả
bash all_in_one.sh
```

#### 3. Ensemble Models
```python
python ensemble_model.py
```

#### 4. Compare Results
```python
python get_diff_res.py --csv1 result1.csv --csv2 result2.csv
```

#### 5. Final Submission
```python
python final_submit.py
```

## 📤 Output

**File chính:** `submit.csv`
- Format: `id,predict_label`
- Sẵn sàng để nộp bài cho cuộc thi
- Được tạo tự động bởi `all_in_one.sh`

## Input/Output Format

### Input Data Format
```csv
id,context,prompt,response,label
abc123,"Context text...","Question?","Response text","no"
```

### Output Prediction Format
```csv
id,predict_label
abc123,"no"
def456,"intrinsic"
```

## Customization

### Thay đổi Model
Để sử dụng mô hình khác, update trong `get_prod.py`:
```python
model_name = "your-model-name"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    # ... other parameters
)
```

### Thay đổi Prompt Template
Update prompt trong `get_prod.py`:
```python
prompt = """Your custom prompt template..."""
```

### Ensemble Configuration
Modify network architecture trong `ensemble_model.py`:
```python
class EnsembleNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        # Custom architecture here
```

## Troubleshooting

### Memory Issues
- Giảm `max_seq_length` trong `get_prod.py`
- Enable `load_in_4bit = True`
- Reduce batch size trong inference

### Model Loading
- Kiểm tra đường dẫn checkpoint
- Verify model compatibility với Unsloth
- Check CUDA availability

### Performance
- Use GPU inference với `device="cuda"`
- Enable mixed precision
- Optimize prompt length

---

## 🏆 Submission Instructions

**Cho Ban Tổ Chức:**
1. **Tải model checkpoints** từ [Google Drive](https://drive.google.com/drive/folders/1uXaU4NIpOKnXL6Io-zeKhBaggXvcGjZX?usp=sharing)
2. Giải nén và đặt vào thư mục `checkpoint/`
3. Đảm bảo có môi trường Python với dependencies (xem `environment.yml`)
4. Đảm bảo có GPU để inference (khuyến nghị)
5. Chạy lệnh: `./all_in_one.sh`
6. Lấy file `submit.csv` làm kết quả cuối cùng

**Thời gian chạy dự kiến:** 15-30 phút (tùy thuộc GPU)

**Requirements:**
- CUDA-capable GPU (khuyến nghị)
- RAM: 16GB+ 
- Storage: 15GB+ cho models và data
- Internet để tải checkpoints

---

**Note**: Thư mục này chứa production code đã được optimize cho cuộc thi. Để development và experimentation, sử dụng notebooks trong thư mục `notebook/`.