# @Author : Serein
# @Time : 2025/10/29 15:59
import pandas as pd
from pathlib import Path

# 输入与输出路径
src_dir = Path("../data/mp_20")
dst_dir = Path("../data/test_data")
dst_dir.mkdir(parents=True, exist_ok=True)

# 要处理的文件名
splits = ["train", "val", "test"]

for split in splits:
    src_file = src_dir / f"{split}.csv"
    dst_file = dst_dir / f"{split}.csv"

    # 读取 CSV 文件
    df = pd.read_csv(src_file)

    # 随机抽取10条（如果总行数<10，则取全部）
    sample_df = df.sample(n=min(10, len(df)), random_state=42)

    # 保存格式不变
    sample_df.to_csv(dst_file, index=False)

    print(f"✅ {split}.csv → {dst_file} ({len(sample_df)} rows)")
