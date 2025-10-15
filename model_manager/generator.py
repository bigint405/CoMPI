import argparse
import hashlib
import json
import os
from queue import Queue
import threading
from typing import List
from model_manager.original_models import get_original_model_from_config
import datasets.imagenet_utils as iu
import random
import logging
import torch.nn as nn
from retrain import retrain_with_task

class ModelInfo:
    type: str
    model: nn.Module
    task: List


## 随机选择不重复的任务
def generate_random_groups(num, classes, num_range):
    groups_hash_set = set()  # 用来存储哈希值，保证每组唯一
    groups = []  # 用来存储实际的组，供返回使用
    
    while len(groups) < num:
        group = random.sample(range(num_range), classes)
        
        # 将组转换为列表，并生成哈希值
        group_sorted = sorted(group)  # 排序确保顺序无关
        group_hash = hashlib.sha256(str(group_sorted).encode()).hexdigest()
        
        # 如果哈希值不在集合中，说明这一组没有出现过
        if group_hash not in groups_hash_set:
            groups_hash_set.add(group_hash)  # 将哈希值添加到集合中
            groups.append(group_sorted)  # 添加实际的组（列表形式）到结果中
    
    # 将结果转换为列表并返回
    return groups


def worker_thread(device_id, task_queue, lock, log_dir):
    """
    工作线程函数：每个显卡对应一个线程
    """
    while True:
        # 获取任务
        mi = task_queue.get()
        if mi is None:
            break  # 如果任务队列为空，结束线程

        taskname = mi.type + str(mi.task)

        with lock:
            logger.info(f"Assigning task {taskname} to GPU {device_id}")
        retrain_with_task(mi.model, mi.task, iu.get_train_dir(), iu.get_val_dir(), device_id, os.path.join(log_dir, taskname), taskname)

        task_queue.task_done()

def generate(model_configs, g_list, log_dir):
    catagories = iu.get_all_catagories()
    num_catagories = len(catagories)
    
    task_queue = Queue()
    lock = threading.Lock()  # 用于线程同步，避免多个线程同时访问日志

    # 创建线程池：每个显卡对应一个线程
    threads = []
    for device_num in g_list:
        t = threading.Thread(target=worker_thread, args=(device_num, task_queue, lock, log_dir))
        t.start()
        threads.append(t)

    for config in model_configs:
        logger.info('Config')
        logger.info(config)
        # 构建再训练任务
        task_ids = generate_random_groups(config['num'], config['class'], num_catagories)
        tasks = []
        for tid in task_ids:
            t = [catagories[i] for i in tid]
            tasks.append(t)
        logger.info('Random Tasks: ')
        logger.info(tasks)

        for i in range(config['num']):
            logger.info(f"Retraining model {config['type']} with task {tasks[i]}")
            model = get_original_model_from_config(config)
            if model is None:
                print(f"ERROR: Invalid model type {config['type']}")
                break
            if config['just_retrain_head']:
                for param in model.parameters():
                    param.requires_grad = False
                # for param in model.classifier.parameters():
                for param in model.fc.parameters():
                    param.requires_grad = True
            # 将任务添加到队列
            mi = ModelInfo()
            mi.model = model
            mi.task = tasks[i]
            mi.type = config['type'] + ("(just_head)" if config['just_retrain_head'] else "")
            task_queue.put(mi)

    # 等待所有任务完成
    task_queue.join()

    # 停止所有线程
    for t in threads:
        task_queue.put(None)  # 向队列中添加None来通知线程退出
    for t in threads:
        t.join()


if __name__ == "__main__":
    log_dir = "generator_logs"
    try:
        os.mkdir(log_dir)
        print(f"Folder '{log_dir}' created successfully.")
    except FileExistsError:
        print(f"Folder '{log_dir}' already exists.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(log_dir,'generator.log'), level=logging.INFO)
    logger.info('Started')

    parser = argparse.ArgumentParser(description='generate models')
    parser.add_argument('-f','--filepath', type=str, help='Path to the config JSON file', required=True)
    parser.add_argument('-g','--gpu', type=str, help='GPU ids to use (e.g. -g 0,1,3), default 0', required=False)

    # 解析命令行参数
    args = parser.parse_args()
    if args.gpu:
        # 使用 map 和 split 将字符串转换为整数列表
        g_list = list(map(int, args.gpu.split(',')))
    else:
        g_list = [0]
    # 打开并读取 JSON 文件
    try:
        with open(args.filepath, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {args.filepath} does not exist.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")
        exit(2)

    generate(model_configs, g_list, log_dir)

