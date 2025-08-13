# app/routes.py
from flask import Blueprint, request, jsonify
import json
import joblib
import numpy as np
import traceback

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from solver.ga_solver import GeneticOptimizer
from utils.json_transform import transform_json
from utils.objective_function import objective_function
from utils.logger_process import setup_logger
from config.config import CONFIG_PATH, load_config

CONFIG = load_config()
# 初始化日志记录器
pangu_logger1 = setup_logger(CONFIG["LOG_PATH"], "D", "盘古大模型1#.log", 7)
opt_logger1 = setup_logger(CONFIG["LOG_PATH"], "D", "河伯求解器1#.log", 7)
pangu_logger2 = setup_logger(CONFIG["LOG_PATH"], "D", "盘古大模型2#.log", 7)
opt_logger2 = setup_logger(CONFIG["LOG_PATH"], "D", "河伯求解器2#.log", 7)

try:
    model1_no1_3d = joblib.load(CONFIG["MODEL1_NO1_3D"])
    model1_no1_7d = joblib.load(CONFIG["MODEL1_NO1_7D"])
    model1_no1_28d = joblib.load(CONFIG["MODEL1_NO1_28D"])
    model1_no2_3d = joblib.load(CONFIG["MODEL1_NO2_3D"])
    model1_no2_7d = joblib.load(CONFIG["MODEL1_NO2_7D"])
    model1_no2_28d = joblib.load(CONFIG["MODEL1_NO2_28D"])
    model2_no1_3d = joblib.load(CONFIG["MODEL2_NO1_3D"])
    model2_no2_3d = joblib.load(CONFIG["MODEL2_NO2_3D"])
    model3_no1_3d = joblib.load(CONFIG["MODEL3_NO1_3D"])
    model3_no2_3d = joblib.load(CONFIG["MODEL3_NO2_3D"])
except:
    model1_no1_3d = CONFIG["MODEL1_NO1_3D"]
    model1_no1_7d = CONFIG["MODEL1_NO1_7D"]
    model1_no1_28d = CONFIG["MODEL1_NO1_28D"]
    model1_no2_3d = CONFIG["MODEL1_NO2_3D"]
    model1_no2_7d = CONFIG["MODEL1_NO2_7D"]
    model1_no2_28d = CONFIG["MODEL1_NO2_28D"]
    model2_no1_3d = CONFIG["MODEL2_NO1_3D"]
    model2_no2_3d = CONFIG["MODEL2_NO2_3D"]
    model3_no1_3d = CONFIG["MODEL3_NO1_3D"]
    model3_no2_3d = CONFIG["MODEL3_NO2_3D"]

# 创建蓝图
first = Blueprint('first', __name__)
    
@first.route('/hebo_opt', methods=['POST'])
def hebo_opt():
    CONFIG = load_config()
    objectives = CONFIG["HEBO_CONFIG"]
    iters = CONFIG["HEBO_ITERS"]

    def build_space(objectives):
        return [
            {'name': key, 'type': 'num', 'lb': value['min'], 'ub': value['max']}
            for key,value in objectives.items()
        ]
    
    def objective_func(params):
        original_data = request.json
        param_dict = {name: round(val, 6) for name, val in zip(params.columns.values, params.values[0])}
        result = objective_function(model3_no1_3d, transform_json(original_data, param_dict.keys(), param_dict.values()))
        if CONFIG.get("TYPE").lower() == "max":
            result = -result
            # opt_logger.info(f"参数配置: {param_dict} , 电能单耗MW预测值: {-result[0]}")
        else:
            result = 1/result
            # opt_logger.info(f"参数配置: {param_dict} , 电能单耗MW预测值: {result[0]}")
        return np.array(result[0]).reshape(-1, 1)

    space = DesignSpace().parse(build_space(objectives))
    hebo = HEBO(space, model_name='gp')
    try:
        for _ in range(int(iters)):
            # opt_logger.info(f"第{_+1}次迭代")
            rec = hebo.suggest(n_suggestions=4)
            hebo.observe(rec, objective_func(rec))

        best_params = hebo.best_x.iloc[0]
        best_x_hebo = {col+'_OPT': float(round(val, 6)) for col, val in best_params.items()}
        if CONFIG.get("TYPE").lower() == "max":
            best_y_hebo = -float(round(float(hebo.best_y), 6))
        else:
            best_y_hebo = float(round(float(hebo.best_y), 6))
            
        result = {**best_x_hebo, "三天强度_OPT": best_y_hebo}
        opt_logger1.info(f'"最优参数配置": {best_x_hebo},"三天强度_OPT": {best_y_hebo}')
        return jsonify({"data":[result],"code":200,"Msg":None,"remark":None}), 200
    except Exception as e:
        opt_logger1.info(f'HEBO优化失败: {str(e)}, {traceback.format_exc()}')
        return jsonify({"data":None,"code":0,"Msg":str(e),"remark":None}), 500
    
@first.route('/ga_opt', methods=['POST'])
def ga_optimize_request():
    CONFIG = load_config()
    try:
        ga_config = CONFIG.get("GA_CONFIG", {})
        input = request.json
        optimizer = GeneticOptimizer(input, model3_no1_3d, ga_config, objective_function)
        best_solution, best_score = optimizer.optimize()

        best_x_ga = dict(zip([x + '_OPT' for x in optimizer.obj_names], [float(np.round(x, 6)) for x in best_solution]))
        best_y_ga = float(np.round(best_score, 6)[0][0])
        result = {**best_x_ga, "三天强度_OPT": best_y_ga}
        
        opt_logger1.info(f'"最优参数配置": {best_x_ga},"三天强度_OPT": {best_y_ga}')
        return jsonify({"data":[result],"code":200,"Msg":None,"remark":None}), 200
    except Exception as e:
        opt_logger1.info(f'GA优化失败: {str(e)}, {traceback.format_exc()}')
        return jsonify({"data":None,"code":0,"Msg":str(e),"remark":None}), 500
    
@first.route('/predict', methods=['POST'])
def predict_request():
    CONFIG = load_config()
    try:
        model3_3d = objective_function(model3_no1_3d, request.json)[0][0]
        model2_3d = objective_function(model2_no1_3d, request.json)[0][0]
        model1_3d = objective_function(model1_no1_3d, request.json)[0][0]
        model1_7d = objective_function(model1_no1_7d, request.json)[0][0]
        model1_28d = objective_function(model1_no1_28d, request.json)[0][0]
        if CONFIG.get("TYPE").lower() == "max":
            model3_3d, model2_3d, model1_3d, model1_7d, model1_28d = model3_3d, model2_3d, model1_3d, model1_7d, model1_28d
        else:
            model3_3d, model2_3d, model1_3d, model1_7d, model1_28d = 1/model3_3d, 1/model2_3d, 1/model1_3d, 1/model1_7d, 1/model1_28d
        pangu_logger1.info(f"化验值预测3d: {model2_3d},\
化验值和过程值预测3d: {model3_3d},\
化验值和1d强度预测3d: {model1_3d},\
化验值和1d强度预测7d: {model1_7d},\
化验值和1d强度预测28d: {model1_28d}")
        return jsonify({"data":[{"化验值预测3d":round(model2_3d,6),
"化验值和过程值预测3d":round(model3_3d,6),
"化验值和1d强度预测3d":round(model1_3d,6),
"化验值和1d强度预测7d":round(model1_7d,6),
"化验值和1d强度预测28d":round(model1_28d,6)}],
                        "code":200,
                        "Msg":None,
                        "remark":None}), 200
    except Exception as e:
        pangu_logger1.error(f"Error: {str(e)},建议检查请求体")
        return jsonify({"Msg": str(e)+',建议检查请求体',"code":0,"data":None,"remark":None}), 500
    
# 创建蓝图
second = Blueprint('second', __name__)
    
@second.route('/hebo_opt', methods=['POST'])
def hebo_opt():
    CONFIG = load_config()
    objectives = CONFIG["HEBO_CONFIG"]
    iters = CONFIG["HEBO_ITERS"]

    def build_space(objectives):
        return [
            {'name': key, 'type': 'num', 'lb': value['min'], 'ub': value['max']}
            for key,value in objectives.items()
        ]
    
    def objective_func(params):
        original_data = request.json
        param_dict = {name: round(val, 6) for name, val in zip(params.columns.values, params.values[0])}
        result = objective_function(model3_no2_3d, transform_json(original_data, param_dict.keys(), param_dict.values()))
        if CONFIG.get("TYPE").lower() == "max":
            result = -result
            # opt_logger.info(f"参数配置: {param_dict} , 电能单耗MW预测值: {-result[0]}")
        else:
            result = 1/result
            # opt_logger.info(f"参数配置: {param_dict} , 电能单耗MW预测值: {result[0]}")
        return np.array(result[0]).reshape(-1, 1)

    space = DesignSpace().parse(build_space(objectives))
    hebo = HEBO(space, model_name='gp')
    try:
        for _ in range(int(iters)):
            # opt_logger.info(f"第{_+1}次迭代")
            rec = hebo.suggest(n_suggestions=4)
            hebo.observe(rec, objective_func(rec))

        best_params = hebo.best_x.iloc[0]
        best_x_hebo = {col+'_OPT': float(round(val, 6)) for col, val in best_params.items()}
        if CONFIG.get("TYPE").lower() == "max":
            best_y_hebo = -float(round(float(hebo.best_y), 6))
        else:
            best_y_hebo = float(round(float(hebo.best_y), 6))
            
        result = {**best_x_hebo, "三天强度_OPT": best_y_hebo}
        opt_logger2.info(f'"最优参数配置": {best_x_hebo},"三天强度_OPT": {best_y_hebo}')
        return jsonify({"data":[result],"code":200,"Msg":None,"remark":None}), 200
    except Exception as e:
        opt_logger2.info(f'HEBO优化失败: {str(e)}, {traceback.format_exc()}')
        return jsonify({"data":None,"code":0,"Msg":str(e),"remark":None}), 500
    
@second.route('/ga_opt', methods=['POST'])
def ga_optimize_request():
    CONFIG = load_config()
    try:
        ga_config = CONFIG.get("GA_CONFIG", {})
        input = request.json
        optimizer = GeneticOptimizer(input, model3_no2_3d, ga_config, objective_function)
        best_solution, best_score = optimizer.optimize()

        best_x_ga = dict(zip([x + '_OPT' for x in optimizer.obj_names], [float(np.round(x, 6)) for x in best_solution]))
        best_y_ga = float(np.round(best_score, 6)[0][0])
        result = {**best_x_ga, "三天强度_OPT": best_y_ga}
        
        opt_logger2.info(f'"最优参数配置": {best_x_ga},"三天强度_OPT": {best_y_ga}')
        return jsonify({"data":[result],"code":200,"Msg":None,"remark":None}), 200
    except Exception as e:
        opt_logger2.info(f'GA优化失败: {str(e)}, {traceback.format_exc()}')
        return jsonify({"data":None,"code":0,"Msg":str(e),"remark":None}), 500
    
@second.route('/predict', methods=['POST'])
def predict_request():
    CONFIG = load_config()
    try:
        model3_3d = objective_function(model3_no2_3d, request.json)[0][0]
        model2_3d = objective_function(model2_no2_3d, request.json)[0][0]
        model1_3d = objective_function(model1_no2_3d, request.json)[0][0]
        model1_7d = objective_function(model1_no2_7d, request.json)[0][0]
        model1_28d = objective_function(model1_no2_28d, request.json)[0][0]
        if CONFIG.get("TYPE").lower() == "max":
            model3_3d, model2_3d, model1_3d, model1_7d, model1_28d = model3_3d, model2_3d, model1_3d, model1_7d, model1_28d
        else:
            model3_3d, model2_3d, model1_3d, model1_7d, model1_28d = 1/model3_3d, 1/model2_3d, 1/model1_3d, 1/model1_7d, 1/model1_28d
        pangu_logger2.info(f"化验值预测3d: {model2_3d},\
化验值和过程值预测3d: {model3_3d},\
化验值和1d强度预测3d: {model1_3d},\
化验值和1d强度预测7d: {model1_7d},\
化验值和1d强度预测28d: {model1_28d}")
        return jsonify({"data":[{"化验值预测3d":round(model2_3d,6),
       "化验值和过程值预测3d":round(model3_3d,6),
       "化验值和1d强度预测3d":round(model1_3d,6),
       "化验值和1d强度预测7d":round(model1_7d,6),
       "化验值和1d强度预测28d":round(model1_28d,6)}],
                        "code":200,
                        "Msg":None,
                        "remark":None}), 200
    except Exception as e:
        pangu_logger2.error(f"Error: {str(e)},建议检查请求体")
        return jsonify({"Msg": str(e)+',建议检查请求体',"code":0,"data":None,"remark":None}), 500

config = Blueprint('config', __name__)
@config.route('/config', methods=['POST'])
def config_request():
    data = (request.json)[0]
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'iters' in data:
            config["HEBO_ITERS"] = data['iters']
            config["GA_CONFIG"]["GENS"] = data['iters']
        else:
            config.pop("HEBO_ITERS", None)

        hebo_config = config.get('HEBO_CONFIG', {})
        keys_to_process = set()
        for key in data.keys():
            if key.endswith('_min') or key.endswith('_max'):
                base_key = key[:-4]
                keys_to_process.add(base_key)
        
        hebo_config_keys = list(hebo_config.keys())
        for key in hebo_config_keys:
            if key not in keys_to_process:
                hebo_config.pop(key, None)
        
        for key in keys_to_process:
            min_key = f"{key}_min"
            max_key = f"{key}_max"
            
            if key not in hebo_config:
                hebo_config[key] = {}
            
            if min_key in data:
                hebo_config[key]["min"] = data[min_key]
            if max_key in data:
                hebo_config[key]["max"] = data[max_key]
                
            if not hebo_config[key]:
                del hebo_config[key]
        
        config['HEBO_CONFIG'] = hebo_config
        config["GA_CONFIG"]["OBJECTIVES"] = hebo_config
        # 保存配置文件
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        opt_logger1.info("配置文件已更新")
        return jsonify({"data": [{"配置更新状态": 1}], "code": 200, "Msg": "配置文件已更新", "remark": None}), 200
    except Exception as e:
        opt_logger1.error(f"更新配置文件时出错: {e}")
        return jsonify({"data": None, "code": 0, "Msg": "更新配置文件时出错", "remark": str(e)}), 500
        