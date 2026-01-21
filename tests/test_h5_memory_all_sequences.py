#!/usr/bin/env python3
"""
测试依次读取所有序列的内存消耗

对比 H5TypeOld.py 和 H5Type.py 在加载所有序列时的内存使用情况
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Dict

import psutil

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_size(bytes_size: int) -> str:
    """格式化字节大小"""
    size: float = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def test_old_load_all_sequences(h5_path: Path) -> Dict[str, Any]:
    """测试旧实现加载所有序列的内存使用"""
    from base.dataset.H5TypeOld import H5Dataset as H5DatasetOld

    print("=" * 80)
    print("【旧实现 H5TypeOld.py】加载所有序列的内存测试")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    memory_snapshots = []

    with H5DatasetOld(h5_path, mode="r") as dataset:
        sequences_list = dataset.list_sequences()
        total_sequences = len(sequences_list)
        print(f"总序列数: {total_sequences}")
        print()

        print("-" * 80)
        print("开始依次加载所有序列...")
        print("-" * 80)
        print(f"{'序号':<8} {'序列名称':<30} {'内存(MB)':<12} {'增长(MB)':<12}")
        print("-" * 80)

        mem_prev = mem_start

        for idx, seq_name in enumerate(sequences_list, 1):
            # 加载序列
            _sequence = dataset.load_sequence(seq_name)

            # 记录内存
            mem_current = get_memory_mb()
            mem_delta = mem_current - mem_prev

            # 每5个序列或者最后一个序列打印一次
            if idx % 5 == 0 or idx == total_sequences:
                print(
                    f"{idx:<8} {seq_name:<30} {mem_current:<12.2f} {mem_delta:<12.2f}"
                )

            # 保存快照
            memory_snapshots.append(
                {
                    "index": idx,
                    "name": seq_name,
                    "memory_mb": mem_current,
                    "delta_mb": mem_delta,
                }
            )

            mem_prev = mem_current

    print()
    print("=" * 80)
    print("【旧实现】测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    total_growth = mem_final - mem_start

    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {total_growth:.2f} MB")
    print(f"平均每序列增长: {total_growth / total_sequences:.2f} MB")
    print()

    return {
        "initial": mem_start,
        "final": mem_final,
        "total_growth": total_growth,
        "avg_per_sequence": total_growth / total_sequences,
        "snapshots": memory_snapshots,
    }


def test_new_load_all_sequences(h5_path: Path) -> Dict[str, Any]:
    """测试新实现加载所有序列的内存使用"""
    from base.dataset.H5Type import H5Dataset as H5DatasetNew

    print()
    print("=" * 80)
    print("【新实现 H5Type.py】加载所有序列的内存测试（延迟加载）")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    memory_snapshots = []

    with H5DatasetNew(h5_path, mode="r") as dataset:
        sequences_list = dataset.list_sequences()
        total_sequences = len(sequences_list)
        print(f"总序列数: {total_sequences}")
        print()

        print("-" * 80)
        print("开始依次加载所有序列对象（不加载实际数据）...")
        print("-" * 80)
        print(f"{'序号':<8} {'序列名称':<30} {'内存(MB)':<12} {'增长(MB)':<12}")
        print("-" * 80)

        mem_prev = mem_start

        for idx, seq_name in enumerate(sequences_list, 1):
            # 加载序列对象（不加载实际数据）
            _sequence = dataset.load_sequence(seq_name)

            # 记录内存
            mem_current = get_memory_mb()
            mem_delta = mem_current - mem_prev

            # 每5个序列或者最后一个序列打印一次
            if idx % 5 == 0 or idx == total_sequences:
                print(
                    f"{idx:<8} {seq_name:<30} {mem_current:<12.2f} {mem_delta:<12.2f}"
                )

            # 保存快照
            memory_snapshots.append(
                {
                    "index": idx,
                    "name": seq_name,
                    "memory_mb": mem_current,
                    "delta_mb": mem_delta,
                }
            )

            mem_prev = mem_current

    print()
    print("=" * 80)
    print("【新实现】测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    total_growth = mem_final - mem_start

    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {total_growth:.2f} MB")
    print(f"平均每序列增长: {total_growth / total_sequences:.2f} MB")
    print()

    return {
        "initial": mem_start,
        "final": mem_final,
        "total_growth": total_growth,
        "avg_per_sequence": total_growth / total_sequences,
        "snapshots": memory_snapshots,
    }


def test_new_load_all_sequences_with_data(h5_path: Path) -> Dict[str, Any]:
    """测试新实现加载所有序列并加载数据的内存使用"""
    from base.dataset.H5Type import H5Dataset as H5DatasetNew

    print()
    print("=" * 80)
    print("【新实现 H5Type.py】加载所有序列并加载数据的内存测试")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    memory_snapshots = []

    with H5DatasetNew(h5_path, mode="r") as dataset:
        sequences_list = dataset.list_sequences()
        total_sequences = len(sequences_list)
        print(f"总序列数: {total_sequences}")
        print()

        print("-" * 80)
        print("开始依次加载所有序列对象并加载数据...")
        print("-" * 80)
        print(
            f"{'序号':<8} {'序列名称':<30} {'内存(MB)':<12} {'增长(MB)':<12} {'数据大小':<15}"
        )
        print("-" * 80)

        mem_prev = mem_start

        for idx, seq_name in enumerate(sequences_list, 1):
            # 加载序列对象（不加载实际数据）
            sequence = dataset.load_sequence(seq_name)

            # 加载实际数据
            data_size = 0
            if sequence.resampled:
                if sequence.resampled.has_imu:
                    imu = sequence.resampled.get_imu()
                    if imu is not None:
                        data_size += imu.nbytes
                if sequence.resampled.has_ground_truth:
                    gt = sequence.resampled.get_ground_truth()
                    if gt is not None:
                        data_size += gt.nbytes

            # 记录内存
            mem_current = get_memory_mb()
            mem_delta = mem_current - mem_prev

            # 每5个序列或者最后一个序列打印一次
            if idx % 5 == 0 or idx == total_sequences:
                print(
                    f"{idx:<8} {seq_name:<30} {mem_current:<12.2f} {mem_delta:<12.2f} {format_size(data_size):<15}"
                )

            # 保存快照
            memory_snapshots.append(
                {
                    "index": idx,
                    "name": seq_name,
                    "memory_mb": mem_current,
                    "delta_mb": mem_delta,
                    "data_size": data_size,
                }
            )

            mem_prev = mem_current

    print()
    print("=" * 80)
    print("【新实现 + 加载数据】测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    total_growth = mem_final - mem_start

    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {total_growth:.2f} MB")
    print(f"平均每序列增长: {total_growth / total_sequences:.2f} MB")
    print()

    return {
        "initial": mem_start,
        "final": mem_final,
        "total_growth": total_growth,
        "avg_per_sequence": total_growth / total_sequences,
        "snapshots": memory_snapshots,
    }


def compare_all_results(
    old_results: Dict[str, Any],
    new_results: Dict[str, Any],
    new_with_data_results: Dict[str, Any],
) -> None:
    """对比所有测试结果"""
    print()
    print("=" * 80)
    print("【综合对比分析】")
    print("=" * 80)
    print()

    print(f"{'测试场景':<40} {'总增长(MB)':<15} {'平均/序列(MB)':<15}")
    print("-" * 80)

    print(
        f"{'旧实现 (加载所有序列 + 数据)':<40} {old_results['total_growth']:<15.2f} {old_results['avg_per_sequence']:<15.2f}"
    )
    print(
        f"{'新实现 (仅加载序列对象)':<40} {new_results['total_growth']:<15.2f} {new_results['avg_per_sequence']:<15.2f}"
    )
    print(
        f"{'新实现 (加载序列 + 数据)':<40} {new_with_data_results['total_growth']:<15.2f} {new_with_data_results['avg_per_sequence']:<15.2f}"
    )
    print()

    print("-" * 80)
    print()

    # 关键发现
    print("关键发现:")
    print()

    # 1. 仅加载序列对象的对比
    saving_1 = old_results["total_growth"] - new_results["total_growth"]
    saving_pct_1 = (saving_1 / old_results["total_growth"]) * 100

    print("1. 仅加载序列对象（不访问数据）:")
    print(f"   ✓ 旧实现增长: {old_results['total_growth']:.2f} MB")
    print(f"   ✓ 新实现增长: {new_results['total_growth']:.2f} MB")
    print(f"   ✓ 节省内存: {saving_1:.2f} MB ({saving_pct_1:.1f}%)")
    print("   ✓ 原因: 新实现使用延迟加载,不在加载序列时读取实际数据")
    print()

    # 2. 加载数据的对比
    saving_2 = old_results["total_growth"] - new_with_data_results["total_growth"]
    saving_pct_2 = (saving_2 / old_results["total_growth"]) * 100

    print("2. 加载序列对象 + 加载数据:")
    print(f"   ✓ 旧实现增长: {old_results['total_growth']:.2f} MB")
    print(f"   ✓ 新实现增长: {new_with_data_results['total_growth']:.2f} MB")
    if saving_2 > 0:
        print(f"   ✓ 节省内存: {saving_2:.2f} MB ({saving_pct_2:.1f}%)")
    else:
        print(f"   ✗ 额外内存: {abs(saving_2):.2f} MB (由于HDF5缓存等机制)")
    print("   → 说明: 当需要访问数据时,两种实现的内存占用接近")
    print()

    # 3. 灵活性对比
    print("3. 灵活性对比:")
    print("   ✓ 新实现可以选择性加载数据:")
    print(
        "     - 仅加载序列对象: ~{:.2f} MB/序列".format(new_results["avg_per_sequence"])
    )
    print(
        "     - 加载序列 + 数据: ~{:.2f} MB/序列".format(
            new_with_data_results["avg_per_sequence"]
        )
    )
    print(
        "   ✗ 旧实现必须加载所有数据: ~{:.2f} MB/序列".format(
            old_results["avg_per_sequence"]
        )
    )
    print()

    # 4. 实际应用场景
    print("4. 实际应用场景建议:")
    print()
    print("   场景1: 需要浏览所有序列的元数据（不访问实际数据）")
    print(f"     • 新实现: ~{new_results['total_growth']:.2f} MB")
    print(f"     • 旧实现: ~{old_results['total_growth']:.2f} MB")
    print(f"     • 节省: {saving_1:.2f} MB ({saving_pct_1:.1f}%)")
    print()
    print("   场景2: 需要访问所有序列的数据")
    print(f"     • 新实现: ~{new_with_data_results['total_growth']:.2f} MB")
    print(f"     • 旧实现: ~{old_results['total_growth']:.2f} MB")
    print("     • 内存占用相近,但新实现支持更灵活的访问方式")
    print()
    print("   场景3: 只需访问部分序列的数据")
    print("     • 新实现优势: 可以选择性加载需要的数据")
    print("     • 旧实现劣势: 必须加载所有序列的全部数据")
    print()

    # 5. 可扩展性
    print("5. 可扩展性分析:")
    print()
    # 假设有100个序列
    hypothetical_sequences = 100
    old_total = old_results["avg_per_sequence"] * hypothetical_sequences
    new_meta_only = new_results["avg_per_sequence"] * hypothetical_sequences

    print(f"   假设有 {hypothetical_sequences} 个序列:")
    print(f"   • 旧实现（全部加载数据）: ~{old_total:.2f} MB")
    print(f"   • 新实现（仅元数据）: ~{new_meta_only:.2f} MB")
    print(f"   • 节省: {old_total - new_meta_only:.2f} MB")
    print()

    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


def main() -> None:
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试H5数据集加载内存")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="/Users/qi/Resources/ABR-AL60.h5",
        help="H5文件路径",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)

    if not h5_path.exists():
        print(f"错误: 测试文件不存在: {h5_path}")
        sys.exit(1)

    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "H5数据集依次加载所有序列内存对比测试" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # 测试旧实现
    old_results = test_old_load_all_sequences(h5_path)

    # 测试新实现（仅加载序列对象）
    new_results = test_new_load_all_sequences(h5_path)

    # 测试新实现（加载序列对象 + 数据）
    new_with_data_results = test_new_load_all_sequences_with_data(h5_path)

    # 综合对比
    compare_all_results(old_results, new_results, new_with_data_results)


if __name__ == "__main__":
    main()
