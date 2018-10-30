import argparse
import os
from shutil import copyfile
import subprocess
import sys
from analyze_taken_metrics import merge_metrics

CLEAR_MEM_SCRIPT_PATH = "clear_mem.sh"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Executes multiple times the get super kmer script to do a performance assesment")
    parser.add_argument('--num_predictors', dest="num_predictors", default="3",
                        help="Max number of predictors for each benchmark. Default value: 3")
    parser.add_argument('--devices', dest="devices", default="gpu",
                        help="Devices used to perform regression, can be cpu, gpu or both")
    parser.add_argument('--input_file', dest="input_file", help="File containing predictors and target data")
    args = parser.parse_args()
    num_predictors = args.num_predictors.split(",")
    devices = args.devices.split(",")
    input_file = args.input_file
    return num_predictors, devices, input_file

def execute_metrics_collection(full_output_path):
    # This should be async
    path = os.path.join(full_output_path, "metrics")
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
    cpu_command = "sar -P ALL 1 99999 > {}/sar_cpu_file.log".format(path)
    memio_command = "sar -b -r 1 99999 > {}/sar_mem_io_file.log".format(path)
    base_command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --id=04:00.0 --format=csv -l 1"

    nvidia_command = "{} | ts %s, >> {}/nvidia_gpu.log ".format(base_command, path)
    process_cpu = subprocess.Popen("LC_TIME='C' exec " + cpu_command, shell=True)
    process_memio = subprocess.Popen("LC_TIME='C' exec " +memio_command, shell=True)
    process_nvidia = subprocess.Popen("LC_TIME='C' exec " +nvidia_command, shell=True)
    return process_cpu, process_memio, process_nvidia

def execute_massive_regressions(params):
    # Sync
    params['output_path'] = "{output_path}/output.csv".format(**params)
    command = "python ../massive_multilinear_regresions.py -i {input_file} -mp {num_predictors} -d {device} -o {output_path}".format(**params)
    command += " | ts %s, > {log_output_path}".format(**params)
    sys.stdout.write("Executing '{}' \n".format(command))
    subprocess.call(command, shell=True)

def execute_sleep(seconds):
    sleep_time =  seconds
    command = "sleep {}".format(sleep_time)
    subprocess.call("exec "+command, shell=True)

def execute_metrics_summary(full_output_path):
    path = os.path.join(full_output_path, "metrics")
    sys.stdout.write("Mergin metrics in {}".format(path))
    merge_metrics(path, 1, "2017-12-09")

def kill_processes(pids):
    sys.stdout.write("Killing metrics collection processes {}\n".format(pids))
    subprocess.call("killall -9 sar", shell=True)
    subprocess.call("killall -9 sar", shell=True)
    subprocess.call("killall -9 nvidia-smi", shell=True)
    subprocess.call("sudo sh {}".format(CLEAR_MEM_SCRIPT_PATH), shell=True)


def execute_assesment(num_predictors, device, input_file):
    params = {'input_file': input_file, 'num_predictors': num_predictors, 'device': device}

    full_output_path = os.path.join("input-{}-maxpredictors-{}-device-{}/".format(input_file.split("/")[-1], str(num_predictors), device))

    print full_output_path
    os.system('mkdir -p {}'.format(full_output_path))
    os.system('mkdir -p {}/tmp'.format(full_output_path))
    # Rewrite for specific output
    params['output_path'] = full_output_path
    params['log_output_path'] = os.path.join(full_output_path, "metrics", "tool_log.csv")

    process_cpu, process_memio, process_nvidia = execute_metrics_collection(full_output_path)

    sys.stdout.write(" ***************************************** \n"\
              "Execution performance assesment \n"\
              "Output path {} \n"\
              "***********************************\n".format(full_output_path))

    sys.stdout.write("Letting metrics collection process to start \n")
    execute_sleep(3)
    execute_massive_regressions(params)
    sys.stdout.write("Letting metrics collection process to finish \n")
    execute_sleep(3)
    kill_processes([process_cpu.pid, process_memio.pid, process_nvidia.pid])
    execute_metrics_summary(full_output_path)


def main():
    num_predictors, devices, input_file = parse_arguments()
    for np in num_predictors:
        for device in devices:
            execute_assesment(np, device, input_file)

if __name__ == '__main__':
    main()
