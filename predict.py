#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路灯状态检测预测脚本
使用训练好的YOLOv8模型对图像进行预测，识别路灯及其状态
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="路灯状态检测预测脚本")
    parser.add_argument("--model", type=str, default="runs/detect/train10/weights/best.pt", 
                        help="模型权重文件路径")
    parser.add_argument("--source", type=str, default="", 
                        help="输入图像或视频路径，支持：图片/视频路径、文件夹路径、'0'表示摄像头")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="置信度阈值")
    parser.add_argument("--save", action="store_true", 
                        help="是否保存结果")
    parser.add_argument("--output", type=str, default="results", 
                        help="结果保存路径")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误：模型文件 {args.model} 不存在!")
        return
    
    # 检查输入源是否存在
    if args.source and not (args.source.isdigit() or os.path.exists(args.source)):
        print(f"错误：输入源 {args.source} 不存在!")
        return
    
    # 创建输出目录
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 类别名称
    class_names = ["StreetLamp", "Obscured"]
    
    # 进行预测
    print(f"开始预测: {args.source if args.source else '请在界面中选择'}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=args.output if args.save else None,
        exist_ok=True,
        verbose=False
    )
    
    # 处理结果
    for i, result in enumerate(results):
        # 获取原始图像
        img = result.orig_img
        
        # 获取检测框
        boxes = result.boxes
        
        # 统计结果
        total_lamps = 0
        obscured_lamps = 0
        
        # 在图像上绘制检测结果
        for box in boxes:
            # 获取类别ID和置信度
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 确定颜色和标签
            if cls_id == 0:  # StreetLamp
                color = (0, 255, 0)  # 绿色
                label = f"StreetLamp {conf:.2f}"
                total_lamps += 1
            else:  # Obscured
                color = (0, 0, 255)  # 红色
                label = f"Obscured {conf:.2f}"
                obscured_lamps += 1
                total_lamps += 1
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示统计信息
        print(f"图像 {i+1}: 检测到 {total_lamps} 个路灯，其中 {obscured_lamps} 个不亮或不够亮")
        
        # 显示结果
        if not args.save:
            cv2.imshow(f"结果 {i+1}", img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("预测完成!")


if __name__ == "__main__":
    main() 