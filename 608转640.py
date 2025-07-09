import os
import cv2
import numpy as np

# 设置路径
input_images_dir = "E:\\dataset\\val\\images"  # 输入图像文件夹路径
input_labels_dir = "E:\\dataset\\val\\labels"  # 输入标签文件夹路径
output_images_dir = "E:\\newdataset\\val\\images"  # 输出图像文件夹路径
output_labels_dir = "E:\\newdataset\\val\\labels"  # 输出标签文件夹路径


def check_and_create_dir(path):
    """检查并创建目录，处理权限问题"""
    try:
        os.makedirs(path, exist_ok=True)
        # 测试写入权限
        test_file = os.path.join(path, 'permission_test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"无法创建或写入目录 {path}: {e}")
        return False


# 检查并创建输出目录
if not check_and_create_dir(output_images_dir):
    print(f"请手动创建目录并确保有写入权限: {output_images_dir}")
    exit()

if not check_and_create_dir(output_labels_dir):
    print(f"请手动创建目录并确保有写入权限: {output_labels_dir}")
    exit()

# 原始尺寸和目标尺寸
original_size = 608
target_size = 640

# 计算缩放比例
scale = target_size / original_size

# 处理每张图像
for image_file in os.listdir(input_images_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # 读取图像
            image_path = os.path.join(input_images_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 缩放图像到640x640
            resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            # 保存缩放后的图像
            output_image_path = os.path.join(output_images_dir, image_file)
            if not cv2.imwrite(output_image_path, resized_image):
                print(f"无法保存图像: {output_image_path}")
                continue

            # 处理对应的标签文件
            label_file = os.path.splitext(image_file)[0] + '.txt'
            input_label_path = os.path.join(input_labels_dir, label_file)
            output_label_path = os.path.join(output_labels_dir, label_file)

            if os.path.exists(input_label_path):
                try:
                    with open(input_label_path, 'r') as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = parts[0]
                            # 转换坐标
                            x_center = float(parts[1]) * scale
                            y_center = float(parts[2]) * scale
                            width = float(parts[3]) * scale
                            height = float(parts[4]) * scale

                            # 确保坐标在0-1范围内
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))

                            new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                            if len(parts) > 5:
                                new_line += " " + " ".join(parts[5:])
                            new_lines.append(new_line)

                    # 写入新的标签文件
                    with open(output_label_path, 'w') as f:
                        f.write("\n".join(new_lines))

                except Exception as e:
                    print(f"处理标签文件 {input_label_path} 时出错: {e}")

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")

print("转换完成！")