from pydub import AudioSegment
import os


def adjust_volume(input_file, output_file, volume_change_db):
    """
    调整WAV音频文件的音量

    参数:
        input_file: 输入WAV文件路径
        output_file: 输出WAV文件路径
        volume_change_db: 音量变化值(dB)，正数增加音量，负数减小音量
    """
    try:
        # 加载音频文件
        print(f"正在加载音频文件: {input_file}")
        audio = AudioSegment.from_wav(input_file)

        # 调整音量
        print(f"正在调整音量: {volume_change_db} dB")
        adjusted_audio = audio + volume_change_db

        # 导出调整后的音频
        print(f"正在导出到: {output_file}")
        adjusted_audio.export(output_file, format="wav")

        print("音量调整完成！")

    except Exception as e:
        print(f"处理音频时出错: {str(e)}")


# 直接在代码中设置参数
if __name__ == "__main__":
    # 在这里设置你的参数
    input_file = "D:\\头盔\\语音包\\speech11.wav"  # 输入文件路径
    output_file = "D:\\a\\speech11.wav"  # 输出文件路径
    volume_change_db = 15.0  # 音量调整值(dB)，正数增加音量，负数减小音量

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
    else:
        # 调用音量调整函数
        adjust_volume(input_file, output_file, volume_change_db)