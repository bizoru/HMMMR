import re
from sys import argv

import pandas as pd
import numpy as np

total_regressions_pattern = re.compile("[0-9]+ Regressions has been done")
to_pattern = re.compile("batch size is [0-9]+")


def extract_regressions_from_log(logfile):
    f = open(logfile, "r")
    for line in f:
        if " Regressions has been done" in line:
            x = total_regressions_pattern.search(line)
            f.close()
            return line[x.start():x.end()]

def extract_time_from_log(logfile, keyword):
    f = open(logfile, "r")
    for line in f:
        if keyword in line:
            break
    f.close()
    return line.split()[3]

def extract_batch_size(logfile):
    f = open(logfile, "r")
    last_batch = ""
    for line in f:
        if "batch size" in line and not "sys" in line:
            last_batch = line
    return last_batch


for log_dir in argv[1:]:
    merged_metrics = pd.read_excel(log_dir+"/metrics/merged_metrics.xlsx", "Metrics")
    cpu_mem = (np.nanmax(merged_metrics["MEM"]) - np.nanmin(merged_metrics["MEM"])) / 1024
    gpu_mem = np.nanmax(merged_metrics["GPU"])
    keyword = "numpy_regression" if "cpu" in log_dir else "massive_multilineal_regresion"
    print log_dir
    print "GPU MEM {}".format(gpu_mem)
    print "CPU MEM {}".format(cpu_mem)
    print "Time {} ns".format(extract_time_from_log(log_dir+"/metrics/tool_log.csv", keyword))
    print "{}".format(extract_regressions_from_log(log_dir+"/metrics/tool_log.csv"))
    if "gpu"  in log_dir:
        print "{}".format(extract_batch_size(log_dir+"/metrics/tool_log.csv"))
    print "*********"*10
