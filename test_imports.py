#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试所有包的导入
"""

def test_imports():
    """测试所有必需包的导入"""
    
    try:
        import pandas as pd
        print("✅ pandas 导入成功")
    except ImportError as e:
        print("❌ pandas 导入失败:", e)
    
    try:
        import numpy as np
        print("✅ numpy 导入成功")
    except ImportError as e:
        print("❌ numpy 导入失败:", e)
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib 导入成功")
    except ImportError as e:
        print("❌ matplotlib 导入失败:", e)
    
    try:
        import seaborn as sns
        print("✅ seaborn 导入成功")
    except ImportError as e:
        print("❌ seaborn 导入失败:", e)
    
    try:
        import networkx as nx
        print("✅ networkx 导入成功")
    except ImportError as e:
        print("❌ networkx 导入失败:", e)
    
    try:
        from sklearn.model_selection import train_test_split
        print("✅ scikit-learn 导入成功")
    except ImportError as e:
        print("❌ scikit-learn 导入失败:", e)
    
    try:
        import scipy.stats as stats
        print("✅ scipy 导入成功")
    except ImportError as e:
        print("❌ scipy 导入失败:", e)
    
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.search.ScoreBased.GES import ges
        print("✅ causal-learn 导入成功")
    except ImportError as e:
        print("❌ causal-learn 导入失败:", e)
    
    import sys
    print(f"\n🐍 当前Python版本: {sys.version}")
    print(f"📍 Python路径: {sys.executable}")

if __name__ == "__main__":
    test_imports()
