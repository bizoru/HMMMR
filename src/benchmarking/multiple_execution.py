from __future__ import print_function

import argparse
import os
import subprocess
import sys

from analyze_taken_metrics import merge_metrics

CLEAR_MEM_SCRIPT_PATH = "clear_mem.sh"
GPU_ID = "04:00.0"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Executes multiple times the get super kmer script to do a performance assesment")
    parser.add_argument('--num_predictors', dest="num_predictors", default="3",
                        help="Max number of predictors for each benchmark. Default value: 3")
    parser.add_argument('--devices', dest="devices", default="gpu",
                        help="Devices used to perform regression, can be cpu, gpu or both")
    parser.add_argument('--input_files', dest="input_files",
                        help="File containing predictors and target data (Comma separated list)")
    args = parser.parse_args()
    num_predictors = args.num_predictors.split(",")
    devices = args.devices.split(",")
    input_files = args.input_files.split(",")
    return num_predictors, devices, input_files


def _safe_dir_token(value):
    """Allow only characters safe for a path component built from user input."""
    return "".join(c for c in value if c.isalnum() or c in ("-", "_", "."))


def _open_log(path):
    return open(path, "ab")


def execute_metrics_collection(full_output_path):
    path = os.path.join(full_output_path, "metrics")
    os.makedirs(path)

    cpu_log = _open_log(os.path.join(path, "sar_cpu_file.log"))
    memio_log = _open_log(os.path.join(path, "sar_mem_io_file.log"))
    nvidia_log = _open_log(os.path.join(path, "nvidia_gpu.log"))

    env = dict(os.environ, LC_TIME="C")
    process_cpu = subprocess.Popen(
        ["sar", "-P", "ALL", "1", "99999"], stdout=cpu_log, env=env)
    process_memio = subprocess.Popen(
        ["sar", "-b", "-r", "1", "99999"], stdout=memio_log, env=env)
    process_nvidia = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
         "--id=" + GPU_ID, "--format=csv", "-l", "1"],
        stdout=nvidia_log, env=env)
    return process_cpu, process_memio, process_nvidia


def execute_massive_regressions(params):
    output_file = os.path.join(params['output_path'], "output.csv")
    params['output_path'] = output_file
    log_output_path = params['log_output_path']
    cmd = [
        "python", "-u", "../massive_multilinear_regresions.py",
        "-i", params['input_file'],
        "-mp", str(params['num_predictors']),
        "-d", params['device'],
        "-o", output_file,
    ]
    sys.stdout.write("Executing {} \n".format(cmd))
    with open(log_output_path, "ab") as log_fh:
        subprocess.check_call(cmd, stdout=log_fh)


def execute_sleep(seconds):
    subprocess.check_call(["sleep", str(int(seconds))])


def execute_metrics_summary(full_output_path):
    path = os.path.join(full_output_path, "metrics")
    sys.stdout.write("Mergin metrics in {}".format(path))
    merge_metrics(path, 1, "2017-12-09")


def kill_processes(pids):
    sys.stdout.write("Killing metrics collection processes {}\n".format(pids))
    subprocess.call(["killall", "-9", "sar"])
    subprocess.call(["killall", "-9", "nvidia-smi"])
    subprocess.call(["sudo", "sh", CLEAR_MEM_SCRIPT_PATH])


def execute_assesment(num_predictors, device, input_file):
    params = {'input_file': input_file,
              'num_predictors': num_predictors, 'device': device}

    safe_input = _safe_dir_token(os.path.basename(input_file))
    safe_device = _safe_dir_token(device)
    safe_predictors = _safe_dir_token(str(num_predictors))
    full_output_path = "input-{}-maxpredictors-{}-device-{}/".format(
        safe_input, safe_predictors, safe_device)

    print(full_output_path)
    os.makedirs(os.path.join(full_output_path, "tmp"))

    params['output_path'] = full_output_path
    params['log_output_path'] = os.path.join(
        full_output_path, "metrics", "tool_log.csv")

    process_cpu, process_memio, process_nvidia = execute_metrics_collection(full_output_path)

    sys.stdout.write(" ***************************************** \n"
                     "Execution performance assesment \n"
                     "Output path {} \n"
                     "***********************************\n".format(full_output_path))

    sys.stdout.write("Letting metrics collection process to start \n")
    execute_sleep(3)
    execute_massive_regressions(params)
    sys.stdout.write("Letting metrics collection process to finish \n")
    execute_sleep(3)
    kill_processes([process_cpu.pid, process_memio.pid, process_nvidia.pid])
    execute_metrics_summary(full_output_path)


def main():
    num_predictors, devices, input_files = parse_arguments()
    for input_file in input_files:
        for predictors in num_predictors:
            for device in devices:
                execute_assesment(predictors, device, input_file)


if __name__ == '__main__':
    main()
