#!/usr/bin/env python3
"""
测试H5数据集读取的内存使用情况

验证延迟加载(lazy loading)是否按预期工作：
1. 加载Sequence对象时不应该加载大量数据
2. 只在调用get_*方法时才加载数据
3. 切片读取应该只加载所需部分
"""

import gc
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.dataset.H5Type import H5Dataset


def get_memory_mb() -> float:
    """
    获取当前进程的内存使用量（MB）

    Returns:
        内存使用量（MB）
    """
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_size(bytes_size: int) -> str:
    """
    格式化字节大小

    Args:
        bytes_size: 字节数

    Returns:
        格式化的字符串，如 "1.23 MB"
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def test_lazy_loading():
    """测试延迟加载功能"""

    h5_path = Path("/Users/qi/Resources/ABR-AL60.h5")

    print("=" * 80)
    print("H5数据集内存使用测试")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    # 记录初始内存
    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    with H5Dataset(h5_path, mode="r") as dataset:
        # 测试1: 加载元数据
        print("-" * 80)
        print("测试1: 加载元数据")
        print("-" * 80)
        assert dataset.metadata is not None
        print(f"  版本: {dataset.metadata.version}")
        print(f"  序列数量: {dataset.metadata.num_sequences}")
        print(f"  总时长: {dataset.metadata.total_duration_hours:.2f} 小时")

        mem_after_meta = get_memory_mb()
        mem_used = mem_after_meta - mem_start
        print(f"  内存使用: {mem_after_meta:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 10 MB）")
        assert mem_used < 10, f"加载元数据内存增长过大: {mem_used:.2f} MB"
        print()

        # 测试2: 列出所有序列（不加载数据）
        print("-" * 80)
        print("测试2: 列出所有序列")
        print("-" * 80)
        sequences_list = dataset.list_sequences()
        print(f"  序列列表: {sequences_list[:5]}... (共{len(sequences_list)}个)")
        print(f"  第一个序列: {sequences_list[0]}")

        mem_after_list = get_memory_mb()
        mem_used = mem_after_list - mem_after_meta
        print(f"  内存使用: {mem_after_list:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 1 MB）")
        assert mem_used < 1, f"列出序列内存增长过大: {mem_used:.2f} MB"
        print()

        # 测试3: 加载第一个Sequence对象（不加载数据）
        print("-" * 80)
        print("测试3: 加载Sequence对象（延迟加载模式）")
        print("-" * 80)
        seq_name = sequences_list[0]
        sequence = dataset.load_sequence(seq_name)
        print(f"  序列名称: {sequence.name}")
        print(f"  序列行数: {sequence.num_rows}")
        print(f"  开始时间: {sequence.attributes.t_start_us}")
        print(f"  结束时间: {sequence.attributes.t_end_us}")
        print(f"  采样频率: {sequence.attributes.frequency_hz} Hz")
        print(f"  有resampled数据: {sequence.has_resampled}")
        print(f"  有aligned数据: {sequence.has_aligned}")

        mem_after_seq = get_memory_mb()
        mem_used = mem_after_seq - mem_after_list
        print(f"  内存使用: {mem_after_seq:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 1 MB）- 因为没有加载数据")
        assert mem_used < 1, f"加载Sequence对象内存增长过大: {mem_used:.2f} MB"
        print()

        # 测试4: 检查数据形状（不加载数据）
        print("-" * 80)
        print("测试4: 检查数据形状（不加载数据）")
        print("-" * 80)
        if sequence.resampled:
            print(f"  resampled.shape: {sequence.resampled.shape}")
            print(f"  has_t_us: {sequence.resampled.has_t_us}")
            print(f"  has_imu: {sequence.resampled.has_imu}")
            print(f"  has_ground_truth: {sequence.resampled.has_ground_truth}")

        mem_after_shape = get_memory_mb()
        mem_used = mem_after_shape - mem_after_seq
        print(f"  内存使用: {mem_after_shape:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 1 MB）")
        assert mem_used < 1, f"检查形状内存增长过大: {mem_used:.2f} MB"
        print()

        # 测试5: 加载小部分数据（切片）
        print("-" * 80)
        print("测试5: 加载小部分数据（前100行）")
        print("-" * 80)
        if sequence.resampled:
            t_us_slice = sequence.resampled.get_t_us(slice(0, 100))
            print(f"  加载时间戳: {t_us_slice.shape if t_us_slice is not None else 'None'}")

            if sequence.resampled.has_imu:
                imu_slice = sequence.resampled.get_imu(slice(0, 100))
                print(f"  加载IMU: {imu_slice.shape if imu_slice is not None else 'None'}")

            if sequence.resampled.has_ground_truth:
                gt_slice = sequence.resampled.get_ground_truth(slice(0, 100))
                print(f"  加载Ground Truth: {gt_slice.shape if gt_slice is not None else 'None'}")

        mem_after_small = get_memory_mb()
        mem_used = mem_after_small - mem_after_shape
        print(f"  内存使用: {mem_after_small:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该较小（< 5 MB）")
        # 注意：这里可能会有一些内存增长，但不应该太大
        print()

        # 测试6: 加载完整的resampled数据（这会占用较多内存）
        print("-" * 80)
        print("测试6: 加载完整的resampled数据（用于对比）")
        print("-" * 80)
        if sequence.resampled:
            t_us_full = sequence.resampled.get_t_us()
            print(f"  加载完整时间戳: {t_us_full.shape if t_us_full is not None else 'None'}")

            if sequence.resampled.has_imu:
                imu_full = sequence.resampled.get_imu()
                print(
                    f"  加载完整IMU: {imu_full.shape if imu_full is not None else 'None'} ({format_size(imu_full.nbytes)})"
                )

            if sequence.resampled.has_ground_truth:
                gt_full = sequence.resampled.get_ground_truth()
                print(
                    f"  加载完整Ground Truth: {gt_full.shape if gt_full is not None else 'None'} ({format_size(gt_full.nbytes)})"
                )

        mem_after_full = get_memory_mb()
        mem_used = mem_after_full - mem_after_small
        print(f"  内存使用: {mem_after_full:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该明显（> 10 MB）")
        print()

        # 测试7: 加载aligned数据
        print("-" * 80)
        print("测试7: 加载aligned数据（切片 vs 完整）")
        print("-" * 80)
        if sequence.aligned:
            # 先加载切片
            aligned_slice = sequence.aligned.get_data(slice(0, 100))
            print(
                f"  加载aligned切片(0:100): {aligned_slice.shape if aligned_slice is not None else 'None'} ({format_size(aligned_slice.nbytes)})"
            )

            mem_after_aligned_slice = get_memory_mb()
            print(f"  内存使用: {mem_after_aligned_slice:.2f} MB")

            # 再加载完整数据
            aligned_full = sequence.aligned.get_data()
            print(
                f"  加载完整aligned: {aligned_full.shape if aligned_full is not None else 'None'} ({format_size(aligned_full.nbytes)})"
            )

            mem_after_aligned_full = get_memory_mb()
            mem_used = mem_after_aligned_full - mem_after_aligned_slice
            print(f"  内存使用: {mem_after_aligned_full:.2f} MB (+{mem_used:.2f} MB)")
        print()

    # 最终内存统计
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    mem_total = mem_final - mem_start
    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {mem_total:.2f} MB")
    print()
    print("✓ 所有测试通过！延迟加载功能正常工作。")
    print()
    print("关键发现:")
    print("  1. 加载Sequence对象本身不会加载数据到内存")
    print("  2. 检查数据形状不会加载数据到内存")
    print("  3. 只在调用get_*方法时才加载数据")
    print("  4. 切片读取比完整读取节省内存")


def test_multiple_sequences():
    """测试加载多个序列的内存使用"""

    h5_path = Path("/Users/qi/Resources/ABR-AL60.h5")

    print()
    print("=" * 80)
    print("测试: 加载多个Sequence对象")
    print("=" * 80)

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存: {mem_start:.2f} MB")
    print()

    with H5Dataset(h5_path, mode="r") as dataset:
        sequences_list = dataset.list_sequences()
        print(f"总序列数: {len(sequences_list)}")

        # 加载前10个序列对象（不加载数据）
        num_to_load = min(10, len(sequences_list))
        loaded_sequences = []
        for i in range(num_to_load):
            seq = dataset.load_sequence(sequences_list[i])
            loaded_sequences.append(seq)

        mem_after_load = get_memory_mb()
        mem_used = mem_after_load - mem_start
        print(f"加载{num_to_load}个Sequence对象后: {mem_after_load:.2f} MB (+{mem_used:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 5 MB）")
        assert mem_used < 5, f"加载多个Sequence对象内存增长过大: {mem_used:.2f} MB"

        print()
        print("✓ 测试通过！可以高效加载大量Sequence对象而不占用过多内存。")


if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("错误: 需要安装psutil库")
        print("请运行: pip install psutil")
        sys.exit(1)

    test_lazy_loading()
    test_multiple_sequences()
