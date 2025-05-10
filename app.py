#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路灯状态检测Web应用
使用Streamlit创建一个简单的Web界面，用于展示路灯检测模型的预测结果
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def main():
    """主函数"""
    # 设置页面标题和图标
    st.set_page_config(
        page_title="路灯状态检测系统",
        page_icon="💡",
        layout="wide"
    )
    
    # 页面标题
    st.title("💡 路灯状态检测系统")
    st.markdown("基于YOLOv8的路灯状态检测系统，可以识别路灯及其工作状态（亮/不亮/不够亮）")
    
    # 侧边栏配置
    st.sidebar.header("模型配置")
    
    # 模型选择
    model_path = st.sidebar.selectbox(
        "选择模型",
        options=["runs/detect/train10/weights/best.pt", "runs/detect/train10/weights/last.pt"],
        index=0
    )
    
    # 置信度阈值
    conf_threshold = st.sidebar.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    
    # 文件上传
    st.sidebar.header("输入")
    source_option = st.sidebar.radio(
        "选择输入源",
        options=["上传图片", "上传视频", "摄像头"],
        index=0
    )
    
    # 加载模型
    @st.cache_resource
    def load_model(model_path):
        return YOLO(model_path)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        st.error(f"错误：模型文件 {model_path} 不存在!")
        return
    
    # 加载模型
    with st.spinner("加载模型中..."):
        model = load_model(model_path)
    
    # 类别名称
    class_names = ["StreetLamp", "Obscured"]
    
    # 处理上传的图片
    if source_option == "上传图片":
        uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 转换上传的文件为OpenCV格式
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 显示原始图像
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图像")
                st.image(image_rgb, use_column_width=True)
            
            # 进行预测
            with st.spinner("正在检测..."):
                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    verbose=False
                )
                
                result = results[0]
                
                # 获取检测框
                boxes = result.boxes
                
                # 统计结果
                total_lamps = 0
                obscured_lamps = 0
                
                # 在图像上绘制检测结果
                result_img = image.copy()
                
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
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 转换为RGB格式用于显示
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # 显示结果图像
                with col2:
                    st.subheader("检测结果")
                    st.image(result_img_rgb, use_column_width=True)
                
                # 显示统计信息
                st.subheader("检测统计")
                st.write(f"- 检测到 **{total_lamps}** 个路灯")
                st.write(f"- 其中 **{obscured_lamps}** 个不亮或不够亮")
                st.write(f"- 路灯正常工作率: **{(total_lamps - obscured_lamps) / total_lamps * 100:.1f}%**" if total_lamps > 0 else "- 未检测到路灯")
    
    # 处理上传的视频
    elif source_option == "上传视频":
        uploaded_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # 保存上传的视频到临时文件
            temp_file = f"temp_video_{int(time.time())}.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 显示视频
            st.subheader("上传的视频")
            st.video(temp_file)
            
            # 处理视频按钮
            if st.button("开始检测"):
                with st.spinner("正在处理视频..."):
                    # 进行预测
                    results = model.predict(
                        source=temp_file,
                        conf=conf_threshold,
                        save=True,
                        project="results",
                        name="video_result",
                        exist_ok=True
                    )
                    
                    # 显示处理后的视频
                    output_path = os.path.join("results", "video_result", Path(temp_file).name)
                    if os.path.exists(output_path):
                        st.subheader("检测结果")
                        st.video(output_path)
                    else:
                        st.error("处理视频时出错，未能生成结果视频")
            
            # 删除临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # 处理摄像头输入
    else:
        st.sidebar.warning("注意：摄像头功能在某些浏览器中可能不可用")
        
        if st.sidebar.button("开始摄像头检测"):
            st.subheader("摄像头检测")
            
            # 创建占位符
            stframe = st.empty()
            
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            
            # 检查摄像头是否成功打开
            if not cap.isOpened():
                st.error("无法打开摄像头!")
                return
            
            stop_button = st.button("停止检测")
            
            while not stop_button:
                # 读取一帧
                ret, frame = cap.read()
                
                if not ret:
                    st.error("无法读取摄像头画面!")
                    break
                
                # 进行预测
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False
                )
                
                result = results[0]
                
                # 获取检测框
                boxes = result.boxes
                
                # 统计结果
                total_lamps = 0
                obscured_lamps = 0
                
                # 在图像上绘制检测结果
                result_img = frame.copy()
                
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
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 在画面上显示统计信息
                info_text = f"路灯: {total_lamps}, 不亮: {obscured_lamps}"
                cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                # 转换为RGB格式用于显示
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # 更新显示
                stframe.image(result_img_rgb, channels="RGB")
                
                # 检查是否点击了停止按钮
                if stop_button:
                    break
                
                # 添加短暂延迟
                time.sleep(0.01)
            
            # 释放摄像头
            cap.release()
    
    # 页脚
    st.markdown("---")
    st.markdown("💡 **路灯状态检测系统** | 基于YOLOv8 | 2024")


if __name__ == "__main__":
    main() 