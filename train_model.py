import csv
import os

import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import settings
from src.dataset_manager.datasetManager import DatasetManager

train_ratio=0.7
dataset_manager = DatasetManager("sepsis_cases_1")
data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))
constr_family = "latestindex"
method = "latestindex"

def check_if_activity_exists(group, activity):
    relevant_activity_idxs = np.where(group[dataset_manager.activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[dataset_manager.label_col] = dataset_manager.pos_label
        return group[:idx]
    else:
        group[dataset_manager.label_col] = dataset_manager.neg_label
        return group


data = data[(data["label"] == dataset_manager.pos_label)].copy()
# second adaptation need
dt_labeled = data.sort_values(dataset_manager.timestamp_col, ascending=True, kind="mergesort").groupby(dataset_manager.case_id_col).apply(
    check_if_activity_exists, activity="Release A")
data = dt_labeled[(dt_labeled["label"] == dataset_manager.pos_label)].copy()
# third adaptation need
dt_labeled = data.sort_values(dataset_manager.timestamp_col, ascending=True, kind="mergesort").groupby(dataset_manager.case_id_col).apply(
    check_if_activity_exists, activity="Admission IC")
dt_labeled.to_csv(os.path.join(settings.dataset_folder, "adapted_sepsis_cases.csv"), sep=";", index=False)


