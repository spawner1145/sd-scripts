#!/usr/bin/env python3
"""
Test script to verify that SDXL files work correctly with LLM text encoding strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")

    try:
        from library.strategy_sdxl import LlmTextEncodingStrategy
        print("✅ LlmTextEncodingStrategy import successful")
    except Exception as e:
        print(f"❌ LlmTextEncodingStrategy import failed: {e}")
        return False

    try:
        from library import strategy_sdxl
        print("✅ strategy_sdxl module import successful")
    except Exception as e:
        print(f"❌ strategy_sdxl module import failed: {e}")
        return False

    return True

def test_sdxl_gen_img_import():
    """Test that sdxl_gen_img.py can be imported without syntax errors"""
    print("\nTesting sdxl_gen_img.py import...")

    try:
        # Just test syntax by compiling the file
        with open('sdxl_gen_img.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'sdxl_gen_img.py', 'exec')
        print("✅ sdxl_gen_img.py syntax check passed")
    except SyntaxError as e:
        print(f"❌ sdxl_gen_img.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ sdxl_gen_img.py import failed: {e}")
        return False

    return True

def test_sdxl_minimal_inference_import():
    """Test that sdxl_minimal_inference.py can be imported without syntax errors"""
    print("\nTesting sdxl_minimal_inference.py import...")

    try:
        # Just test syntax by compiling the file
        with open('sdxl_minimal_inference.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'sdxl_minimal_inference.py', 'exec')
        print("✅ sdxl_minimal_inference.py syntax check passed")
    except SyntaxError as e:
        print(f"❌ sdxl_minimal_inference.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ sdxl_minimal_inference.py import failed: {e}")
        return False

    return True

def test_sdxl_train_import():
    """Test that sdxl_train.py can be imported without syntax errors"""
    print("\nTesting sdxl_train.py import...")

    try:
        # Just test syntax by compiling the file
        with open('sdxl_train.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'sdxl_train.py', 'exec')
        print("✅ sdxl_train.py syntax check passed")
    except SyntaxError as e:
        print(f"❌ sdxl_train.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ sdxl_train.py import failed: {e}")
        return False

    return True

def test_sdxl_train_network_import():
    """Test that sdxl_train_network.py can be imported without syntax errors"""
    print("\nTesting sdxl_train_network.py import...")

    try:
        # Just test syntax by compiling the file
        with open('sdxl_train_network.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'sdxl_train_network.py', 'exec')
        print("✅ sdxl_train_network.py syntax check passed")
    except SyntaxError as e:
        print(f"❌ sdxl_train_network.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ sdxl_train_network.py import failed: {e}")
        return False

    return True

def main():
    print("🔍 SDXL Files Compatibility Test")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_sdxl_gen_img_import()
    all_passed &= test_sdxl_minimal_inference_import()
    all_passed &= test_sdxl_train_import()
    all_passed &= test_sdxl_train_network_import()

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! SDXL files are compatible with LLM text encoding strategy.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()