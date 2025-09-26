#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试封装后的函数调用
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.modules.animate.preprocess.preprocess_data import run_preprocessing_pipeline
from wan.generate import generate_with_params


def test_preprocessing():
    """
    测试预处理函数调用
    """
    print("开始执行预处理...")
    
    # 调用预处理函数
    run_preprocessing_pipeline(
        ckpt_path="./Wan2.2-Animate-14B/process_checkpoint",
        video_path="./examples/wan_animate/replace/video.mp4",
        refer_path="./examples/wan_animate/replace/image.jpeg",
        save_path="./examples/wan_animate/replace/process_results",
        resolution_area=[1280, 720],
        fps=30,
        replace_flag=True,
        retarget_flag=False,
        use_flux=False,
        iterations=3,
        k=7,
        w_len=1,
        h_len=1
    )
    
    print("预处理执行完成!")


def test_generation():
    """
    测试视频生成函数调用
    """
    print("开始执行视频生成...")
    
    # 调用生成函数
    generate_with_params(
        task="animate-14B",
        ckpt_dir="./Wan2.2-Animate-14B/",
        src_root_path="./examples/wan_animate/replace/process_results/",
        refert_num=1,
        replace_flag=True,
        use_relighting_lora=True,
        size="1280*720"
    )
    
    print("视频生成执行完成!")


def main():
    """
    主函数：依次执行预处理和生成
    """
    print("开始测试封装的函数调用...")
    
    # 执行预处理
    test_preprocessing()
    
    # 执行生成
    test_generation()
    
    print("所有测试完成!")


if __name__ == "__main__":
    main()