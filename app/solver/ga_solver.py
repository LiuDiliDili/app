from flask import request
import random 
from utils.json_transform import transform_json
from config.config import load_config
    


class GeneticOptimizer:
    def __init__(self, input, url, ga_config, fitness_function):
        self.input = input
        # self.input_cols = input_cols
        self.pop_size = ga_config.get("POP_SIZE", 10)
        self.gens = ga_config.get("GENS", 10)
        self.mutate_rate = ga_config.get("MUTATE_RATE", 0.2)
        self.objectives = ga_config.get("OBJECTIVES", [])
        self.url = url
        self.fitness_function = fitness_function
        # 目标名列表
        self.obj_names = [key for key in self.objectives.keys()]
        # 上下限列表
        self.obj_mins = [value["min"] for value in self.objectives.values()]
        self.obj_maxs = [value["max"] for value in self.objectives.values()]


    def optimize(self):
        CONFIG = load_config()
        population = [
            [random.uniform(self.obj_mins[i], self.obj_maxs[i]) for i in range(len(self.objectives))]
            for _ in range(self.pop_size)
        ]
        for gen in range(int(self.gens)):
            scores = [self.fitness_function(self.url, transform_json(self.input, self.obj_names, ind)) for ind in population]

            sorted_pop = [x for _, x in sorted(zip(scores, population), reverse=True)]
            population = sorted_pop[:self.pop_size // 2]
            while len(population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                child = [(p1[i] + p2[i]) / 2 for i in range(len(self.objectives))]
                # 变异
                if random.random() < self.mutate_rate:
                    for i in range(len(child)):
                        delta = (self.obj_maxs[i] - self.obj_mins[i]) * 0.01
                        child[i] += random.uniform(-delta, delta)
                        child[i] = min(max(child[i], self.obj_mins[i]), self.obj_maxs[i])
                population.append(child)
        # 最优结果
        if CONFIG.get("TYPE").lower() == "max":
            final_scores = [self.fitness_function(self.url, transform_json(self.input, self.obj_names, ind)) for ind in population]
            best_idx = final_scores.index(max(final_scores))
        else:
            final_scores = [1/(self.fitness_function(self.url, transform_json(self.input, self.obj_names,ind))) for ind in population]
            best_idx = final_scores.index(min(final_scores))
        
        best = population[best_idx]
        return best, final_scores[best_idx]