import logging.config
import yaml
from pathlib import Path

def init_logging(config_path='../configs/logging.yml'):
    """加载您的日志配置"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建日志目录
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.config.dictConfig(config)