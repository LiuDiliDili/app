# utils/logger_process.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(log_path, when, log_filename, backup_count=30):
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_filename)
    
    # 创建独立的logger实例
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        try:
            handler = TimedRotatingFileHandler(
                log_file,
                when=when,
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                delay=True  # 延迟创建文件直到有日志写入
            )
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        except Exception as e:
            # 备用方案：控制台输出
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.error(f"Failed to create file handler for {log_filename}: {e}")
    
    # 避免日志传播到根logger
    logger.propagate = False
    return logger
