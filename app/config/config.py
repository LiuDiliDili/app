# app/config.py
import json

CONFIG_PATH = "app/config/config.json"
def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)
    
