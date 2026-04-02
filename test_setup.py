#!/usr/bin/env python3
"""
Test script to verify the LoVRR + DeepThinker bridge setup.
Run this before starting the services to check configuration.
"""

import os
import sys

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def check_file(path, description):
    """Check if a file/directory exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {path}")
    return exists

def main():
    print("=" * 60)
    print("LoVRR + DeepThinker Bridge Setup Verification")
    print("=" * 60)
    print()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    all_ok = True
    
    # 1. Check bridge files
    print("1. Bridge Files:")
    bridge_files = [
        ("config.yaml", "Configuration file"),
        ("deepthinker_service.py", "DeepThinker service"),
        ("deepthinker_client.py", "DeepThinker client"),
        ("start_deepthinker_service.sh", "DeepThinker start script"),
        ("start_lovrr.sh", "LoVRR start script"),
        ("start_all.sh", "Combined start script"),
    ]
    for filename, desc in bridge_files:
        if not check_file(os.path.join(script_dir, filename), desc):
            all_ok = False
    print()
    
    # 2. Check LoVRR files
    print("2. LoVRR Files:")
    lovrr_dir = os.path.join(base_dir, "LoVRR-main")
    lovrr_files = [
        ("demo/app.py", "Original demo"),
        ("demo/app_with_deepthinker.py", "DeepThinker integrated demo"),
    ]
    for filename, desc in lovrr_files:
        if not check_file(os.path.join(lovrr_dir, filename), desc):
            all_ok = False
    
    # Check model files
    model_files = [
        ("models/LoVRR-7B", "LoVRR-7B model"),
        ("models/Qwen2-VL-7B-Instruct", "Base Qwen2-VL model"),
    ]
    for filepath, desc in model_files:
        if not check_file(filepath, desc):
            all_ok = False
    print()
    
    # 3. Check DeepThinker files
    print("3. DeepThinker Files:")
    deepthinker_dir = os.path.join(base_dir, "DeepThinker-main")
    deepthinker_files = [
        ("src/qwen-vl-utils", "Qwen VL utils"),
    ]
    for filename, desc in deepthinker_files:
        if not check_file(os.path.join(deepthinker_dir, filename), desc):
            all_ok = False
    
    # Check DeepThinker model
    if not check_file("models/DeepThinker-7B", "DeepThinker-7B model"):
        all_ok = False
    print()
    
    # 4. Check config
    print("4. Configuration:")
    config_path = os.path.join(script_dir, "config.yaml")
    if os.path.exists(config_path) and YAML_AVAILABLE:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        deepthinker_config = config.get('deepthinker_service', {})
        lovrr_config = config.get('lovrr', {})
        
        print(f"  DeepThinker model path: {deepthinker_config.get('model_path', 'NOT SET')}")
        print(f"  DeepThinker service port: {deepthinker_config.get('port', 8002)}")
        print(f"  LoVRR model path: {lovrr_config.get('model_path', 'NOT SET')}")
        print(f"  Use DeepThinker answerer: {lovrr_config.get('use_deepthinker_answerer', True)}")
        
        # Verify model paths in config
        r1_model = deepthinker_config.get('model_path', '')
        if r1_model and not os.path.exists(r1_model):
            print(f"  ❌ DeepThinker model path in config does not exist!")
            all_ok = False
        
        vm_model = lovrr_config.get('model_path', '')
        if vm_model and not os.path.exists(vm_model):
            print(f"  ❌ LoVRR model path in config does not exist!")
            all_ok = False
    elif not YAML_AVAILABLE:
        print("  ⚠️ PyYAML not installed, cannot parse config")
        print("  Config file exists:", os.path.exists(config_path))
    else:
        print("  ❌ Config file not found!")
        all_ok = False
    print()
    
    # 5. Check Python imports (without activating specific envs)
    print("5. Python Environment (current):")
    print(f"  Python: {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    
    if YAML_AVAILABLE:
        print("  ✅ PyYAML available")
    else:
        print("  ❌ PyYAML not installed (pip install pyyaml)")
        all_ok = False
    
    try:
        import requests
        print("  ✅ Requests available")
    except ImportError:
        print("  ❌ Requests not installed (pip install requests)")
        all_ok = False
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All checks passed! Ready to start services.")
        print()
        print("Next steps:")
        print("  1. Start DeepThinker service: ./start_deepthinker_service.sh")
        print("  2. Start LoVRR demo:   ./start_lovrr.sh")
        print("  Or run both: ./start_all.sh")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
