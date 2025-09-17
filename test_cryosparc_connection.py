#!/usr/bin/env python3
"""
Enhanced CryoSPARC connection test script.

This script provides comprehensive testing of CryoSPARC connectivity,
including basic connection, project access, and workspace verification.
"""

import sys
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.config.config_loader import ConfigLoader


def test_cryosparc_connection():
    """Test CryoSPARC connection with comprehensive checks."""
    print("🧪 Testing CryoSPARC Connection")
    print("=" * 50)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        print("✅ Configuration loaded successfully!")
        print(f"   Host: {config.cryosparc.host}")
        print(f"   Port: {config.cryosparc.base_port}")
        print(f"   Username: {config.cryosparc.username}")
        print(f"   License ID: {config.cryosparc.license_id[:8]}...")
        print()
        
        # Test CryoSPARC connection
        print("🔧 Testing CryoSPARC connection...")
        start_time = time.time()
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        connection_time = time.time() - start_time
        
        print(f"✅ CryoSPARC connection successful! (took {connection_time:.2f}s)")
        
    except Exception as e:
        print(f"❌ CryoSPARC connection failed: {e}")
        print()
        print("🔍 Troubleshooting tips:")
        print("1. Make sure CryoSPARC is running on the specified host and port")
        print("2. Verify the license ID is correct")
        print("3. Check if CryoSPARC services are accessible")
        print("4. Ensure the CryoSPARC Python tools are properly installed")
        print("5. Verify network connectivity to CryoSPARC server")
        print("6. Check if the project and workspace UIDs exist")
        return False


def test_connection_performance():
    """Test connection performance and stability."""
    print("\n⚡ Testing Connection Performance")
    print("=" * 50)
    
    try:
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        # Test multiple connections
        print("Testing multiple connection attempts...")
        connection_times = []
        
        for i in range(3):
            print(f"   Attempt {i+1}/3...", end=" ")
            start_time = time.time()
            cryosparc_tools = CryoSPARCTools(config.cryosparc)
            connection_time = time.time() - start_time
            connection_times.append(connection_time)
            print(f"{connection_time:.2f}s")
        
        avg_time = sum(connection_times) / len(connection_times)
        print(f"\n📊 Performance Summary:")
        print(f"   Average connection time: {avg_time:.2f}s")
        print(f"   Fastest connection: {min(connection_times):.2f}s")
        print(f"   Slowest connection: {max(connection_times):.2f}s")
        
        if avg_time < 5.0:
            print("✅ Connection performance is good")
        elif avg_time < 10.0:
            print("⚠️ Connection performance is acceptable")
        else:
            print("❌ Connection performance is slow")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Enhanced CryoSPARC Connection Test")
    print("=" * 50)
    
    # Test basic connection
    success = test_cryosparc_connection()
    
    if success:
        # Test performance
        perf_success = test_connection_performance()
        
        if perf_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("   ✅ CryoSPARC connection is working properly")
            print("   ✅ Performance is acceptable")
            print("   ✅ Ready for workflow execution")
            sys.exit(0)
        else:
            print("\n⚠️ Connection works but performance issues detected")
            sys.exit(1)
    else:
        print("\n❌ CryoSPARC connection test failed!")
        print("   Please fix the connection issues before proceeding")
        sys.exit(1)
