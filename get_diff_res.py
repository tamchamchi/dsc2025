import pandas as pd

def compare_results(csv1: str, csv2: str, key_col: str = "id", label_col: str = "predict_label", return_csv: bool = True):
    print(f"Đang đọc {csv1} và {csv2} ...")
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Kiểm tra cột
    for c in [key_col, label_col]:
        if c not in df1.columns:
            raise ValueError(f"{csv1} thiếu cột '{c}'")
        if c not in df2.columns:
            raise ValueError(f"{csv2} thiếu cột '{c}'")

    # Gộp theo id
    merged = pd.merge(df1[[key_col, label_col]], df2[[key_col, label_col]], on=key_col, how="outer", suffixes=('_file1', '_file2'))

    # Các trường hợp khác nhau
    diff = merged[merged[f"{label_col}_file1"] != merged[f"{label_col}_file2"]]
    same = len(merged) - len(diff)

    print(f"\nTổng số dòng: {len(merged)}")
    print(f"Giống nhau : {same}")
    print(f"Khác nhau  : {len(diff)}")

    if len(diff) > 0:
        print("\nChi tiết các dòng khác nhau:")
        print(diff.to_string(index=False))

    # Xuất ra file nếu muốn
    if return_csv:
        diff.to_csv(f"./data/diff_results.csv", index=False)
        print("\n✅ Đã lưu các dòng khác nhau vào diff_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1", type=str, required=True)
    parser.add_argument("--csv2", type=str, required=True)

    args = parser.parse_args()

    compare_results(args.csv1, args.csv2)