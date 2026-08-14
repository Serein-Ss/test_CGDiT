# @Author : Serein
# @Time : 2026/3/13 17:29
import json
import os
import string
import argparse
from pathlib import Path

try:
    from pyxtal.symmetry import Group
except ImportError:
    print("错误: 找不到 pyxtal 库！请在 cgdit 环境中运行此脚本。")
    exit(1)


def generate_json_files(output_dir=".", repeats=5):
    unconstrained_queries = []
    constrained_queries = []

    print("正在从 PyXtal 动态提取 230 个空间群的真实 Wyckoff 位点...")

    # 遍历 1 到 230 的所有空间群
    for sg in range(1, 231):
        g = Group(sg)

        # 探测该空间群实际支持的 Wyckoff 字母
        valid_letters = []
        for char in string.ascii_lowercase:
            try:
                _ = g[char].ops
                valid_letters.append(char)
            except:
                pass

        # 根据空间群实际情况选择测试位点
        if len(valid_letters) >= 2:
            # 取前两个 Wyckoff 位点
            selected_wyckoffs = [valid_letters[0], valid_letters[1]]
        elif len(valid_letters) == 1:
            # 只有一个 Wyckoff 位点
            selected_wyckoffs = [valid_letters[0]]
        else:
            print(f"警告：空间群 {sg} 未找到合法的 Wyckoff 位点。")
            continue

        # ⭐ 关键修改：所有位点统一使用 C 元素
        selected_atoms = ["C"] * len(selected_wyckoffs)

        # 每个空间群生成 5 个结构
        for _ in range(repeats):

            # 方法一：不指定元素
            unconstrained_queries.append({
                "spacegroup_number": sg,
                "wyckoff_letters": selected_wyckoffs,
                "atom_types": None
            })

            # 方法二：指定元素 (全部 C)
            constrained_queries.append({
                "spacegroup_number": sg,
                "wyckoff_letters": selected_wyckoffs,
                "atom_types": selected_atoms
            })

    # 保存 JSON
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unconstrained_path = output_dir / "unconstrained_queries.json"
    constrained_path = output_dir / "constrained_queries.json"
    with unconstrained_path.open("w", encoding="utf-8") as f:
        json.dump(unconstrained_queries, f, indent=4)

    with constrained_path.open("w", encoding="utf-8") as f:
        json.dump(constrained_queries, f, indent=4)

    print(f"✅ 成功生成 JSON 配置文件！")
    print(f" - unconstrained_queries.json: {len(unconstrained_queries)} 个生成任务")
    print(f" - constrained_queries.json: {len(constrained_queries)} 个生成任务")
    print(f" (固定元素类型为 C)")


def cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--repeats", default=5, type=int)
    args = parser.parse_args(argv)
    generate_json_files(args.output_dir, args.repeats)


if __name__ == "__main__":
    cli()
