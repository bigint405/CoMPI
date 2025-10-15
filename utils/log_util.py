from datetime import datetime
import os
import sys

# 默认日志输出流为 sys.stdout
_log_output_stream = sys.stdout

def set_log_file(file_path_or_stream):
    """
    设置 log_print 的输出目标。
    :param file_path_or_stream: 可以是文件路径（str）或已打开的文件对象（支持 .write() 的对象）
    """
    global _log_output_stream
    if isinstance(file_path_or_stream, str):
        _log_output_stream = open(file_path_or_stream, "a", buffering=1)  # 行缓冲
    else:
        _log_output_stream = file_path_or_stream

def log_print(*args, sep=" ", end="\n", flush=True):
    """
    带时间戳的 print()，精确到毫秒+微秒前两位，并分开显示：
    - YYYY-MM-DD HH:MM:SS.mmm uu - message
    :param args: 打印的内容
    :param sep: 分隔符
    :param end: 结束符
    :param flush: 是否立即刷新缓冲区
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")  # 秒
    msec = f"{now.microsecond // 1000:03d}"        # 毫秒
    usec_extra = f"{(now.microsecond // 100) % 100:02d}"  # 额外两位微秒
    formatted_time = f"{timestamp}.{msec}.{usec_extra}"   # 例如：2025-07-01 15:30:45.123.45

    print(f"process {os.getpid()}: {formatted_time} -", *args,
          sep=sep, end=end, flush=flush, file=_log_output_stream)