import pandas as pd
import logging
import json

def transform_json(input_json, obj_cols=None, obj_params=None):
    """模型输入转换"""
    data = input_json[0]
    if obj_cols:
        for col,param in zip(obj_cols, obj_params):
            data[col] = param

    return [data]