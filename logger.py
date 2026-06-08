import logging
from logging.handlers import RotatingFileHandler
import os
import requests

class QuantLogger:
    def __init__(self, log_dir="logs", webhook_url=None):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger("SpotsQuantV10")
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加 Handler
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            
            # 控制台输出
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # 文件输出 (最大 10MB, 轮转 5 个文件)
            fh = RotatingFileHandler(os.path.join(log_dir, "trading.log"), maxBytes=10*1024*1024, backupCount=5)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
        self.webhook_url = webhook_url

    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def alert(self, msg):
        """核心交易警报：不仅记入日志，还会发送到 Telegram/企业微信"""
        self.logger.error(f"🚨 ALERT: {msg}")
        if self.webhook_url:
            try:
                # 兼容通用 Webhook
                requests.post(self.webhook_url, json={"text": f"🚨 Quant Bot Alert: {msg}"}, timeout=5)
            except Exception as e:
                self.logger.error(f"Webhook send failed: {e}")

if __name__ == "__main__":
    # 测试 Logger
    log = QuantLogger()
    log.info("系统初始化完成。")
    log.alert("触发 30% 熔断红线！已强制切断实盘！")
