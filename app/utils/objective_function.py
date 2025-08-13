import numpy as np
import requests
import pandas as pd
from config.config import load_config
    


def objective_function(model, data):
        """
        计算适应度的结果
        :param model: 模型或者模型调用的url
        :param data: 输入
        :return: 适应度结果
        """
        CONFIG = load_config()
        if isinstance(model, str):
                # 调用华为盘古大模型的url
                input = {"data": data}
                result = requests.post(model, json=input).json()['result']
        else:
                # 直接调用本地树模型
                input = pd.DataFrame(data)
                result = model.predict(input.reindex(columns=model.feature_names_in_))
                
        if CONFIG.get("TYPE").lower() == "max":       
                return np.array(result[0]).reshape(-1, 1)
        else:
                return 1/(np.array(result[0]).reshape(-1, 1))