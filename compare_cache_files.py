import numpy as np
import os

def compare_cache_files():
    base_dir = r"D:\code\test_qian\test_cache_data\1_imgs"

    # 比较所有三个文件的两个版本
    for i in range(1, 4):
        lsnet_file = os.path.join(base_dir, f"test_{i:03d}_te_outputs_lsnet.npz")
        standard_file = os.path.join(base_dir, f"test_{i:03d}_te_outputs.npz")

        print(f"\nComparing test_{i:03d}:")

        if not os.path.exists(lsnet_file):
            print(f"  LSNet file {lsnet_file} not found")
            continue
        if not os.path.exists(standard_file):
            print(f"  Standard file {standard_file} not found")
            continue

        # 加载数据
        lsnet_data = np.load(lsnet_file)
        standard_data = np.load(standard_file)

        print(f"  LSNet file keys: {list(lsnet_data.keys())}")
        print(f"  Standard file keys: {list(standard_data.keys())}")

        # 检查共同的键
        common_keys = set(lsnet_data.keys()) & set(standard_data.keys())
        print(f"  Common keys: {sorted(common_keys)}")

        # 比较共同的数组
        all_identical = True
        for key in sorted(common_keys):
            lsnet_array = lsnet_data[key]
            standard_array = standard_data[key]

            print(f"  {key}: shape {lsnet_array.shape}, dtype {lsnet_array.dtype}")

            if lsnet_array.shape != standard_array.shape:
                print(f"    ERROR: Different shapes! LSNet: {lsnet_array.shape}, Standard: {standard_array.shape}")
                all_identical = False
                continue

            if not np.array_equal(lsnet_array, standard_array):
                print(f"    ERROR: Arrays are not identical!")
                # 检查最大差异
                diff = np.abs(lsnet_array - standard_array)
                max_diff = np.max(diff)
                print(f"    Max difference: {max_diff}")
                all_identical = False
            else:
                print(f"    ✓ Identical")

        if all_identical:
            print(f"  ✓ All common arrays identical for test_{i:03d}")
        else:
            print(f"  ✗ Differences found in common arrays for test_{i:03d}")

        # 检查LSNet特有的键
        lsnet_only_keys = set(lsnet_data.keys()) - set(standard_data.keys())
        if lsnet_only_keys:
            print(f"  LSNet-only keys: {sorted(lsnet_only_keys)}")
            for key in sorted(lsnet_only_keys):
                array = lsnet_data[key]
                print(f"    {key}: shape {array.shape}, dtype {array.dtype}")

if __name__ == "__main__":
    compare_cache_files()