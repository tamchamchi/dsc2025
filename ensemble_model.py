import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class EnsembleNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_ensemble(features_train: np.ndarray,
                   labels_train: np.ndarray,
                   features_val: np.ndarray,
                   labels_val: np.ndarray,
                   num_classes: int,
                   epochs: int = 20,
                   batch_size: int = 64,
                   lr: float = 0.001,
                   verbose: bool = True):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = features_train.shape[1]
    model = EnsembleNN(input_dim=input_dim, num_classes=num_classes, dropout_p=0.5).to(DEVICE)

    X_train = torch.tensor(features_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)
    X_val = torch.tensor(features_val, dtype=torch.float32)
    y_val = torch.tensor(labels_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0); n += xb.size(0)

        # eval
        model.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(DEVICE))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.append(preds); all_y.append(yb.numpy())
        all_preds = np.concatenate(all_preds); all_y = np.concatenate(all_y)
        acc = accuracy_score(all_y, all_preds)

        if verbose:
            print(f"[Epoch {epoch:02d}] loss={total_loss/max(n,1):.4f} | val_acc={acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, {"val_acc": best_val_acc}

def predict_on_private_test(
    df: pd.DataFrame, 
    model: torch.nn.Module,          # Model PyTorch đã huấn luyện
    scaler: StandardScaler,          # scaler đã fit từ train
    feature_columns: list, 
    label_encoder: LabelEncoder,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Dự đoán nhãn trên một tập dữ liệu test riêng tư và lưu kết quả.

    Args:
        df (pd.DataFrame): DataFrame của tập test, phải chứa cột 'id' và các cột đặc trưng.
        model (torch.nn.Module): Mô hình EnsembleNN đã được huấn luyện.
        scaler (StandardScaler): scaler đã fit trên dữ liệu huấn luyện.
        feature_columns (list): Danh sách tên các cột đặc trưng mà mô hình yêu cầu.
        label_encoder (LabelEncoder): Bộ mã hóa nhãn đã được fit trên dữ liệu huấn luyện.
        output_csv_path (str): Đường dẫn để lưu file CSV submission.

    Returns:
        pd.DataFrame: Một DataFrame chứa hai cột: 'id' và 'predict_label'.
    """
    print(f"--- Bắt đầu dự đoán trên tập dữ liệu test với {len(df)} mẫu ---")
    
    # --- Kiểm tra dữ liệu ---
    if not all(col in df.columns for col in feature_columns):
        raise ValueError("DataFrame test thiếu các cột đặc trưng cần thiết.")
    if 'id' not in df.columns:
        raise ValueError("DataFrame test phải có cột 'id'.")

    # --- Chuẩn bị dữ liệu test ---
    X_private = df[feature_columns].to_numpy(dtype=np.float32)
    X_private = scaler.transform(X_private)

    # --- Dự đoán ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_private, dtype=torch.float32, device=DEVICE))
        numeric_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    print("✅ Dự đoán dạng số hoàn tất.")

    # --- Decode sang dạng chữ ---
    string_predictions = label_encoder.inverse_transform(numeric_predictions)
    print("✅ Chuyển đổi nhãn sang dạng chữ hoàn tất.")

    # --- Tạo và lưu kết quả ---
    submission_df = pd.DataFrame({
        'id': df['id'],
        'predict_label': string_predictions
    })
    submission_df.to_csv(output_csv_path, index=False)
    print(f"✅ Đã lưu file submission vào: '{output_csv_path}'")
    
    return submission_df

def train():
    train_1 = pd.read_csv("./data/prod/train_prod_Qwen3-4B-Base.csv")
    train_2 = pd.read_csv("./data/prod/train_prod_Qwen2.5-7B.csv")
    train_3 = pd.read_csv("./data/prod/train_prod_Qwen2.5-7B-Instruct.csv")

    eval_1 = pd.read_csv("./data/prod/eval_prod_Qwen3-4B-Base.csv")
    eval_2 = pd.read_csv("./data/prod/eval_prod_Qwen2.5-7B.csv")
    eval_3 = pd.read_csv("./data/prod/eval_prod_Qwen2.5-7B-Instruct.csv")

    train_df = pd.merge(
    pd.merge(train_1, train_2, on=['id', 'label'], how='inner'),
    train_3,
    on=['id', 'label'],
    how='inner'
    )

    eval_df = pd.merge(
        pd.merge(eval_1, eval_2, on=['id', 'label'], how='inner'),
        eval_3,
        on=['id', 'label'],
        how='inner'
    )



    feature_columns = [col for col in train_df.columns if col not in ['id', 'label']]
    target_column = 'label'

    X_train = train_df[feature_columns]
    y_train_str = train_df[target_column]

    X_eval  = eval_df[feature_columns]
    y_eval_str  = eval_df[target_column]


    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_eval  = label_encoder.transform(y_eval_str)

    num_classes = len(label_encoder.classes_)
    print(f"Số lớp: {num_classes} -> {list(label_encoder.classes_)}")

    scaler = StandardScaler()
    Xtr_np = scaler.fit_transform(X_train.to_numpy(dtype=np.float32))
    Xev_np = scaler.transform(X_eval.to_numpy(dtype=np.float32))
    ytr_np = y_train
    yev_np = y_eval

    model_nn, stats = train_ensemble(
        features_train=Xtr_np,
        labels_train=ytr_np,
        features_val=Xev_np,
        labels_val=yev_np,
        num_classes=num_classes,
        epochs=50,
        batch_size=128,
        lr=0.005,
        verbose=True
    )

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_nn.eval()
    with torch.no_grad():
        logits = model_nn(torch.tensor(Xev_np, dtype=torch.float32, device=DEVICE))
        y_pred = torch.argmax(logits, dim=-1).cpu().numpy()

    # Báo cáo kết quả
    accuracy = accuracy_score(y_eval, y_pred)

    print(f"Độ chính xác (Accuracy) của EnsembleNN: {accuracy:.4f}")

    class_names = label_encoder.classes_
    print("\nBáo cáo phân loại (Classification Report):")
    print(classification_report(y_eval, y_pred, target_names=class_names, digits=4))
    return model_nn, scaler, feature_columns, label_encoder

if __name__ == "__main__":
    model_nn, scaler, feature_columns, label_encoder = train()

    private_1 = pd.read_csv("./data/prod/private_prod_Qwen3-4B-Base.csv")
    private_2 = pd.read_csv("./data/prod/private_prod_Qwen2.5-7B.csv")
    private_3 = pd.read_csv("./data/prod/private_prod_Qwen2.5-7B-Instruct.csv")
    
    private_df = pd.merge(
        pd.merge(private_1, private_2, on=['id'], how='inner'),
        private_3,
        on=['id'],
        how='inner'
    )

    submission_df = predict_on_private_test(
        df=private_df,
        model=model_nn,
        scaler=scaler,
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        output_csv_path="data/submission_ensemble_nn.csv"
    )



