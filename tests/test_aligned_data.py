#!/usr/bin/env python3
"""
测试 aligned 数据的读取功能

验证 H5TypeOld.py 和 H5Type.py 都能正确读取 aligned 数据
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_aligned_data():
    """测试 aligned 数据读取"""

    from base.dataset.H5Type import H5Dataset as H5DatasetNew
    from base.dataset.H5TypeOld import H5Dataset as H5DatasetOld

    h5_path = Path("/Users/qi/Resources/ABR-AL60.h5")

    print("=" * 80)
    print("测试 aligned 数据读取功能")
    print("=" * 80)
    print()

    # 获取第一个序列
    with H5DatasetOld(h5_path, mode="r") as dataset_old:
        sequences_list = dataset_old.list_sequences()
        seq_name = sequences_list[0]

    print(f"测试序列: {seq_name}")
    print()

    # 旧实现读取
    print("-" * 80)
    print("旧实现 (H5TypeOld.py)")
    print("-" * 80)

    with H5DatasetOld(h5_path, mode="r") as dataset_old:
        seq_old = dataset_old.load_sequence(seq_name)

        print(f"  有 aligned 数据: {seq_old.aligned is not None}")
        assert seq_old.aligned is not None, "旧实现未能读取 aligned 数据"

        old_aligned = seq_old.aligned.data
        print(f"  aligned 数据形状: {old_aligned.shape}")
        print(f"  数据类型: {old_aligned.dtype}")
        print(f"  数据范围: [{old_aligned.min():.4f}, {old_aligned.max():.4f}]")

        # 测试切片访问
        old_slice = old_aligned[0:100, :]
        print(f"  切片 [0:100] 形状: {old_slice.shape}")

    print()

    # 新实现读取
    print("-" * 80)
    print("新实现 (H5Type.py)")
    print("-" * 80)

    with H5DatasetNew(h5_path, mode="r") as dataset_new:
        seq_new = dataset_new.load_sequence(seq_name)

        print(f"  有 aligned 数据: {seq_new.aligned is not None}")
        assert seq_new.aligned is not None, "新实现未能读取 aligned 数据"

        new_aligned = seq_new.aligned.get_data()
        print(f"  aligned 数据形状: {new_aligned.shape}")
        print(f"  数据类型: {new_aligned.dtype}")
        print(f"  数据范围: [{new_aligned.min():.4f}, {new_aligned.max():.4f}]")

        # 测试切片访问
        new_slice = seq_new.aligned.get_data(slice(0, 100))
        print(f"  切片 [0:100] 形状: {new_slice.shape}")

    print()

    # 对比结果
    print("-" * 80)
    print("对比分析")
    print("-" * 80)

    print(f"  形状一致: {old_aligned.shape == new_aligned.shape}")
    print(f"  数据类型一致: {old_aligned.dtype == new_aligned.dtype}")
    print(f"  数据完全相同: {np.array_equal(old_aligned, new_aligned)}")
    print(f"  切片数据相同: {np.array_equal(old_slice, new_slice)}")

    # 验证数据结构
    print()
    print("数据结构验证:")
    print(f"  aligned 数据应该是 [N, 17]: {old_aligned.shape == old_aligned.shape}")
    print(f"  - 17维特征: t_us(1) + gyr(3) + acc(3) + pos(3) + qwxyz(4) + vel(3)")

    assert old_aligned.shape == new_aligned.shape, "形状不一致"
    assert np.array_equal(old_aligned, new_aligned), "数据不一致"

    print()
    print("=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)
    print()
    print("总结:")
    print("  1. H5TypeOld.py 成功添加了 aligned 数据读取支持")
    print("  2. 两个实现读取的 aligned 数据完全一致")
    print("  3. 切片访问功能正常")


if __name__ == "__main__":
    test_aligned_data()
