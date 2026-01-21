#!/usr/bin/env python3
"""
对比 H5TypeOld.py 和 H5Type.py 的内存使用情况

主要对比点：
1. 加载Sequence对象的内存占用
2. 检查数据形状的内存占用
3. 切片读取的内存占用
4. 完整读取的内存占用
"""

import gc
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_size(bytes_size: int) -> str:
    """格式化字节大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def test_old_implementation(h5_path: Path):
    """测试旧实现的内存使用"""
    from base.dataset.H5TypeOld import H5Dataset as H5DatasetOld

    print("=" * 80)
    print("【旧实现 H5TypeOld.py】内存使用测试")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    results = {}

    with H5DatasetOld(h5_path, mode="r") as dataset:
        # 测试1: 加载元数据
        print("-" * 80)
        print("测试1: 加载元数据")
        print("-" * 80)
        assert dataset.metadata is not None
        print(f"  版本: {dataset.metadata.version}")
        print(f"  序列数量: {dataset.metadata.num_sequences}")

        mem_after = get_memory_mb()
        results['metadata'] = mem_after - mem_start
        print(f"  内存使用: {mem_after:.2f} MB (+{results['metadata']:.2f} MB)")
        print()

        # 测试2: 列出所有序列
        print("-" * 80)
        print("测试2: 列出所有序列")
        print("-" * 80)
        sequences_list = dataset.list_sequences()
        print(f"  序列数量: {len(sequences_list)}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['list_sequences'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['list_sequences']:.2f} MB)")
        print()

        # 测试3: 加载第一个Sequence对象（注意：旧实现会立即加载数据！）
        print("-" * 80)
        print("测试3: 加载第一个Sequence对象")
        print("-" * 80)
        print("  ⚠️  警告: 旧实现在加载Sequence时会加载所有数据到内存！")
        seq_name = sequences_list[0]
        sequence = dataset.load_sequence(seq_name)
        print(f"  序列名称: {sequence.name}")
        print(f"  序列行数: {sequence.attributes.num_rows}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['load_sequence'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['load_sequence']:.2f} MB)")
        print(f"  ⚠️  这包含了加载所有 resampled 数据的内存占用")
        print()

        # 测试4: 访问数据（数据已经在内存中）
        print("-" * 80)
        print("测试4: 访问数据（数据已在内存中）")
        print("-" * 80)
        if sequence.resampled:
            print(f"  t_us: {sequence.resampled.t_us.shape if sequence.resampled.t_us is not None else 'None'}")
            if sequence.resampled.imu is not None:
                print(f"  imu: {sequence.resampled.imu.shape} ({format_size(sequence.resampled.imu.nbytes)})")
            if sequence.resampled.ground_truth is not None:
                print(f"  ground_truth: {sequence.resampled.ground_truth.shape} ({format_size(sequence.resampled.ground_truth.nbytes)})")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['access_data'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['access_data']:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该为0（数据已在内存中）")
        print()

    # 最终统计
    print("=" * 80)
    print("【旧实现】测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    results['total'] = mem_final - mem_start
    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {results['total']:.2f} MB")
    print()

    return results


def test_new_implementation(h5_path: Path):
    """测试新实现的内存使用"""
    from base.dataset.H5Type import H5Dataset as H5DatasetNew

    print("=" * 80)
    print("【新实现 H5Type.py】内存使用测试（延迟加载）")
    print("=" * 80)
    print(f"测试文件: {h5_path}")
    print(f"文件大小: {format_size(h5_path.stat().st_size)}")
    print()

    gc.collect()
    mem_start = get_memory_mb()
    print(f"初始内存使用: {mem_start:.2f} MB")
    print()

    results = {}

    with H5DatasetNew(h5_path, mode="r") as dataset:
        # 测试1: 加载元数据
        print("-" * 80)
        print("测试1: 加载元数据")
        print("-" * 80)
        assert dataset.metadata is not None
        print(f"  版本: {dataset.metadata.version}")
        print(f"  序列数量: {dataset.metadata.num_sequences}")

        mem_after = get_memory_mb()
        results['metadata'] = mem_after - mem_start
        print(f"  内存使用: {mem_after:.2f} MB (+{results['metadata']:.2f} MB)")
        print()

        # 测试2: 列出所有序列
        print("-" * 80)
        print("测试2: 列出所有序列")
        print("-" * 80)
        sequences_list = dataset.list_sequences()
        print(f"  序列数量: {len(sequences_list)}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['list_sequences'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['list_sequences']:.2f} MB)")
        print()

        # 测试3: 加载第一个Sequence对象（延迟加载，不加载数据）
        print("-" * 80)
        print("测试3: 加载Sequence对象（延迟加载模式）")
        print("-" * 80)
        print("  ✓ 新实现只加载元数据，不加载数值数据")
        seq_name = sequences_list[0]
        sequence = dataset.load_sequence(seq_name)
        print(f"  序列名称: {sequence.name}")
        print(f"  序列行数: {sequence.num_rows}")
        print(f"  有resampled数据: {sequence.has_resampled}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['load_sequence'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['load_sequence']:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 1 MB）")
        print()

        # 测试4: 检查数据形状（不加载数据）
        print("-" * 80)
        print("测试4: 检查数据形状（不加载数据）")
        print("-" * 80)
        if sequence.resampled:
            print(f"  resampled.shape: {sequence.resampled.shape}")
            print(f"  has_t_us: {sequence.resampled.has_t_us}")
            print(f"  has_imu: {sequence.resampled.has_imu}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['check_shape'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['check_shape']:.2f} MB)")
        print(f"  ✓ 预期: 内存增长应该很小（< 1 MB）")
        print()

        # 测试5: 加载小部分数据（切片）
        print("-" * 80)
        print("测试5: 加载小部分数据（前100行）")
        print("-" * 80)
        if sequence.resampled:
            if sequence.resampled.has_t_us:
                t_us_slice = sequence.resampled.get_t_us(slice(0, 100))
                print(f"  加载时间戳: {t_us_slice.shape if t_us_slice is not None else 'None'}")

            if sequence.resampled.has_imu:
                imu_slice = sequence.resampled.get_imu(slice(0, 100))
                print(f"  加载IMU: {imu_slice.shape if imu_slice is not None else 'None'}")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['load_slice'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['load_slice']:.2f} MB)")
        print()

        # 测试6: 加载完整数据
        print("-" * 80)
        print("测试6: 加载完整的resampled数据")
        print("-" * 80)
        if sequence.resampled:
            if sequence.resampled.has_t_us:
                t_us_full = sequence.resampled.get_t_us()
                print(f"  加载完整时间戳: {t_us_full.shape if t_us_full is not None else 'None'}")

            if sequence.resampled.has_imu:
                imu_full = sequence.resampled.get_imu()
                print(f"  加载完整IMU: {imu_full.shape if imu_full is not None else 'None'} ({format_size(imu_full.nbytes)})")

        mem_before = mem_after
        mem_after = get_memory_mb()
        results['load_full'] = mem_after - mem_before
        print(f"  内存使用: {mem_after:.2f} MB (+{results['load_full']:.2f} MB)")
        print()

    # 最终统计
    print("=" * 80)
    print("【新实现】测试总结")
    print("=" * 80)
    mem_final = get_memory_mb()
    results['total'] = mem_final - mem_start
    print(f"初始内存: {mem_start:.2f} MB")
    print(f"最终内存: {mem_final:.2f} MB")
    print(f"总增长: {results['total']:.2f} MB")
    print()

    return results


def compare_results(old_results: dict, new_results: dict):
    """对比两个实现的结果"""
    print("=" * 80)
    print("【对比分析】旧实现 vs 新实现")
    print("=" * 80)
    print()

    print(f"{'测试项目':<30} {'旧实现 (MB)':<15} {'新实现 (MB)':<15} {'差异 (MB)':<15} {'改进':<15}")
    print("-" * 90)

    # 对比各个阶段
    comparisons = [
        ("加载元数据", 'metadata', 'metadata'),
        ("列出序列", 'list_sequences', 'list_sequences'),
        ("加载Sequence对象", 'load_sequence', 'load_sequence'),
        ("总内存增长", 'total', 'total'),
    ]

    for label, old_key, new_key in comparisons:
        if old_key in old_results and new_key in new_results:
            old_val = old_results[old_key]
            new_val = new_results[new_key]
            diff = old_val - new_val

            if diff > 0:
                improvement = f"↓ {diff:.2f} MB"
            elif diff < 0:
                improvement = f"↑ {abs(diff):.2f} MB"
            else:
                improvement = "无变化"

            print(f"{label:<30} {old_val:<15.2f} {new_val:<15.2f} {diff:<15.2f} {improvement:<15}")

    print()
    print("-" * 90)
    print()

    # 关键发现
    print("关键发现:")
    print()

    # 1. 加载Sequence对象时的内存占用
    load_seq_diff = old_results.get('load_sequence', 0) - new_results.get('load_sequence', 0)
    print(f"1. 加载Sequence对象:")
    if load_seq_diff > 1:
        print(f"   ✓ 新实现节省了 {load_seq_diff:.2f} MB 内存")
        print(f"     原因: 新实现使用延迟加载，不在初始化时加载数据")
    else:
        print(f"   ✗ 差异不明显 ({load_seq_diff:.2f} MB)")

    print()

    # 2. 总内存占用
    total_diff = old_results.get('total', 0) - new_results.get('total', 0)
    print(f"2. 总内存占用:")
    if total_diff > 1:
        print(f"   ✓ 新实现节省了 {total_diff:.2f} MB 内存")
        print(f"     改进比例: {(total_diff / old_results.get('total', 1) * 100):.1f}%")
    else:
        print(f"   ✗ 差异不明显 ({total_diff:.2f} MB)")

    print()

    # 3. 灵活性
    print("3. 灵活性对比:")
    print("   ✓ 新实现支持切片读取，可以按需加载数据")
    print("   ✓ 新实现可以在不加载全部数据的情况下检查数据形状")
    print("   ✓ 旧实现必须在加载Sequence时加载全部数据")

    print()

    # 4. 适用场景
    print("4. 适用场景建议:")
    print("   • 新实现 (H5Type.py):")
    print("     - 适用于大型数据集")
    print("     - 需要随机访问或切片访问数据")
    print("     - 内存受限的环境")
    print()
    print("   • 旧实现 (H5TypeOld.py):")
    print("     - 适用于小型数据集")
    print("     - 需要频繁访问全部数据")
    print("     - 对延迟不敏感的场景")

    print()


def main():
    """主测试函数"""
    h5_path = Path("/Users/qi/Resources/ABR-AL60.h5")

    if not h5_path.exists():
        print(f"错误: 测试文件不存在: {h5_path}")
        sys.exit(1)

    try:
        import psutil
    except ImportError:
        print("错误: 需要安装psutil库")
        print("请运行: pip install psutil")
        sys.exit(1)

    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "H5数据集内存使用对比测试" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # 测试旧实现
    old_results = test_old_implementation(h5_path)

    print("\n\n")
    # 测试新实现
    new_results = test_new_implementation(h5_path)

    # 对比结果
    print("\n\n")
    compare_results(old_results, new_results)

    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
