#!/usr/bin/env python3
"""
Test script to verify DeepSeek-OCR installation and basic functionality.

Usage:
    python test_setup.py                    # Run all tests
    python test_setup.py --quick            # Skip model inference test
    python test_setup.py --image test.png   # Test with specific image
"""

import sys
import os
import argparse
import time

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def print_result(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} - {name}")
    if details:
        print(f"         {details}")

def test_imports():
    """Test that all required packages are importable."""
    print_header("Testing Imports")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"),
        ("PIL", "Pillow"),
        ("pdf2image", "pdf2image"),
        ("pdfplumber", "pdfplumber"),
        ("fitz", "PyMuPDF"),
    ]
    
    all_passed = True
    for module, name in packages:
        try:
            __import__(module)
            print_result(name, True)
        except ImportError as e:
            print_result(name, False, str(e))
            all_passed = False
    
    return all_passed

def test_cuda():
    """Test CUDA availability and GPU info."""
    print_header("Testing CUDA")
    
    import torch
    
    cuda_available = torch.cuda.is_available()
    print_result("CUDA Available", cuda_available)
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_result("GPU Detected", True, f"{gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check bfloat16 support
        bf16_support = torch.cuda.is_bf16_supported()
        print_result("BFloat16 Support", bf16_support)
        
        return True
    
    return False

def test_flash_attention():
    """Test flash attention availability."""
    print_header("Testing Flash Attention")
    
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        print_result("Flash Attention", True, f"Version: {version}")
        return True
    except ImportError:
        print_result("Flash Attention", False, "Not installed (will use fallback)")
        return False

def test_model_files(model_path: str):
    """Test that model files exist."""
    print_header("Testing Model Files")
    
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]
    
    safetensor_patterns = [".safetensors", ".bin"]
    
    all_passed = True
    
    # Check required files
    for filename in required_files:
        filepath = os.path.join(model_path, filename)
        exists = os.path.exists(filepath)
        print_result(filename, exists, filepath if exists else "NOT FOUND")
        if not exists:
            all_passed = False
    
    # Check for model weights
    has_weights = False
    for pattern in safetensor_patterns:
        for f in os.listdir(model_path) if os.path.exists(model_path) else []:
            if f.endswith(pattern):
                has_weights = True
                size_gb = os.path.getsize(os.path.join(model_path, f)) / 1e9
                print_result(f"Model weights ({f})", True, f"{size_gb:.2f} GB")
                break
        if has_weights:
            break
    
    if not has_weights:
        print_result("Model weights", False, "No .safetensors or .bin files found")
        all_passed = False
    
    return all_passed

def test_tokenizer(model_path: str):
    """Test tokenizer loading."""
    print_header("Testing Tokenizer")
    
    try:
        from transformers import AutoTokenizer
        
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        elapsed = time.time() - start
        
        print_result("Tokenizer Load", True, f"{elapsed:.2f}s")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print_result("Tokenization", True, f"'{test_text}' -> {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print_result("Tokenizer", False, str(e))
        return False

def test_model_inference(model_path: str, test_image: str = None):
    """Test actual model inference."""
    print_header("Testing Model Inference")
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from PIL import Image
        
        # Create a simple test image if none provided
        if test_image and os.path.exists(test_image):
            print(f"  Using provided image: {test_image}")
        else:
            # Create a simple test image with text
            print("  Creating test image...")
            img = Image.new('RGB', (400, 100), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((10, 40), "Hello World - Test Image", fill='black')
            test_image = "/tmp/test_ocr_image.png"
            img.save(test_image)
        
        # Load model
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("  Loading model (this may take a minute)...")
        start = time.time()
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation='flash_attention_2'
        )
        model = model.eval().cuda().to(torch.bfloat16)
        load_time = time.time() - start
        print_result("Model Load", True, f"{load_time:.2f}s")
        
        # Run inference
        print("  Running inference...")
        prompt = "<image>\nFree OCR."
        
        start = time.time()
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=test_image,
            base_size=640,
            image_size=640,
            crop_mode=False,
            save_results=False
        )
        infer_time = time.time() - start
        
        # Extract result
        if isinstance(result, dict):
            text = result.get("text", result.get("content", str(result)))
        else:
            text = str(result)
        
        print_result("Inference", True, f"{infer_time:.2f}s")
        print(f"\n  OCR Result: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print_result("Model Inference", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-OCR setup")
    parser.add_argument("--model-path", default="/workspace/models/deepseek-ocr",
                        help="Path to model directory")
    parser.add_argument("--quick", action="store_true",
                        help="Skip model inference test")
    parser.add_argument("--image", type=str, default=None,
                        help="Test image path for inference")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  DeepSeek-OCR Setup Test")
    print("="*60)
    print(f"\n  Model Path: {args.model_path}")
    print(f"  Quick Mode: {args.quick}")
    
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["cuda"] = test_cuda()
    results["flash_attn"] = test_flash_attention()
    results["model_files"] = test_model_files(args.model_path)
    results["tokenizer"] = test_tokenizer(args.model_path)
    
    if not args.quick:
        results["inference"] = test_model_inference(args.model_path, args.image)
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        print_result(name, result)
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  \033[92m✓ All tests passed! Ready to run.\033[0m\n")
        return 0
    else:
        print("\n  \033[91m✗ Some tests failed. Check errors above.\033[0m\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
