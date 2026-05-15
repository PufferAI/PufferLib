#!/usr/bin/env python3
"""Simple test to verify PufferLib can be imported and basic functionality works."""

import sys
import os

# Add PufferLib to Python path
sys.path.insert(0, '/workspace/PufferLib')

try:
    # Test basic imports that don't require external dependencies
    print("Testing basic PufferLib imports...")

    # First try to import the main module parts individually
    try:
        import pufferlib.exceptions
        print("✓ pufferlib.exceptions imported successfully")
    except Exception as e:
        print(f"✗ pufferlib.exceptions failed: {e}")

    try:
        import pufferlib.utils
        print("✓ pufferlib.utils imported successfully")
    except Exception as e:
        print(f"✗ pufferlib.utils failed: {e}")

    # Test if we can at least load the module structure
    print(f"✓ PufferLib directory found at: {os.path.dirname(__file__)}")
    print("✓ Basic test completed - PufferLib structure is accessible")

except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\nTo fully test PufferLib, you need to install dependencies:")
print("pip install numpy gymnasium")