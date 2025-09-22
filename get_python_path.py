#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取PyCharm使用的Python路径
"""

import sys
import os

print("="*50)
print("PyCharm Python环境信息")
print("="*50)
print(f"Python版本: {sys.version}")
print(f"Python可执行文件路径: {sys.executable}")
print(f"包安装路径: {sys.path}")
print("="*50)

# 生成pip安装命令
python_exe = sys.executable
print(f"\n请在终端中运行以下命令来安装缺失的包:")
print(f'"{python_exe}" -m pip install scikit-learn scipy causal-learn')
print("\n或者分别安装:")
print(f'"{python_exe}" -m pip install scikit-learn')
print(f'"{python_exe}" -m pip install scipy')
print(f'"{python_exe}" -m pip install causal-learn')
