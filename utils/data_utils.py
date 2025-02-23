import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка при декодировании JSON из файла '{file_path}'.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
