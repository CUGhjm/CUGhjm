import pyttsx3
import soundfile as sf
import os
import logging
from datetime import datetime


def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tts_converter.log'),
            logging.StreamHandler()
        ]
    )


def get_available_voices(engine):
    """获取可用的语音列表"""
    voices = engine.getProperty('voices')
    logging.info("可用的语音列表:")
    for i, voice in enumerate(voices):
        logging.info(f"{i}: ID={voice.id} | 名称={voice.name} | 语言={voice.languages} | 性别={voice.gender}")


def text_to_speech(text, output_file="output.wav", rate=150, volume=1.0,
                   sample_rate=48000, voice_index=None, bit_depth='PCM_24'):
    """
    使用 pyttsx3 将中文文本转为高质量语音并保存为 WAV 文件

    参数:
        text (str): 要转换的文本
        output_file (str): 输出的 WAV 文件名（默认 output.wav）
        rate (int): 语速（默认 150，数值越大语速越快）
        volume (float): 音量（0.0~1.0，默认 1.0）
        sample_rate (int): 采样率（默认 48000 Hz）
        voice_index (int): 语音索引（可选，从可用语音中选择）
        bit_depth (str): 音频位深度（默认 'PCM_24'，可选 'PCM_16', 'PCM_24', 'PCM_32'）
    """
    try:
        start_time = datetime.now()
        logging.info(f"开始转换文本: '{text}'")

        # 初始化引擎
        engine = pyttsx3.init()

        # 打印可用语音
        get_available_voices(engine)

        # 设置语音（如果指定）
        if voice_index is not None:
            voices = engine.getProperty('voices')
            if 0 <= voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
                logging.info(f"已选择语音: {voices[voice_index].name}")
            else:
                logging.warning(f"无效的语音索引 {voice_index}，将使用默认语音")

        # 设置语速和音量
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        # 临时保存为 WAV 文件
        temp_file = "temp_output.wav"
        engine.save_to_file(text, temp_file)
        engine.runAndWait()

        # 使用 soundfile 重新保存文件以设置采样率和位深度
        data, current_sr = sf.read(temp_file)

        # 如果需要，进行采样率转换
        if current_sr != sample_rate:
            from scipy import signal
            logging.info(f"正在从 {current_sr} Hz 重采样到 {sample_rate} Hz...")
            samples = round(len(data) * float(sample_rate) / current_sr)
            data = signal.resample(data, samples)

        # 保存最终文件
        sf.write(output_file, data, samplerate=sample_rate, subtype=bit_depth)
        os.remove(temp_file)  # 删除临时文件

        # 打印保存信息
        abs_path = os.path.abspath(output_file)
        file_size = os.path.getsize(output_file) / 1024  # KB
        duration = len(data) / sample_rate  # 秒

        logging.info(f"语音文件已保存至: {abs_path}")
        logging.info(f"文件大小: {file_size:.2f} KB")
        logging.info(f"音频时长: {duration:.2f} 秒")
        logging.info(f"采样率: {sample_rate} Hz")
        logging.info(f"位深度: {bit_depth}")
        logging.info(f"处理时间: {(datetime.now() - start_time).total_seconds():.2f} 秒")

        return True

    except Exception as e:
        logging.error(f"转换失败: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    setup_logging()

    # 输入要转换的文本
    input_text = "检测到您摔倒，已向紧急联系人拨打电话，并发送当前位置"
    # 设置输出文件名
    output_filename = "D:\\头盔\\语音包\\speech10.wav"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # 调用函数进行转换
    success = text_to_speech(
        text=input_text,
        output_file=output_filename,
        rate=120,  # 中等语速
        volume=0.95,  # 稍低于最大音量以避免失真
        sample_rate=48000,  # 高质量采样率
        voice_index=0,  # 尝试不同的语音索引
        bit_depth='PCM_24'  # 24位深度
    )

    if success:
        logging.info("转换成功完成！")
    else:
        logging.error("转换过程中出现错误")