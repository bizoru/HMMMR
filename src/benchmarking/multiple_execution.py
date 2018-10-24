import argparse
import os
from shutil import copyfile
import subprocess
import sys
from utils.analyze_taken_metrics import merge_metrics

import config
from config import MSPK_PARTITION_PATH, THREADS, TOTAL_CORES, KMC_PATH, CLEAR_MEM_SCRIPT_PATH

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Executes multiple times the get super kmer script to do a performance assesment")
    parser.add_argument('--kmers', dest="kmers", default="31",
                        help="Kmer size to perform performance assesment (Comma separated). Default value: 31")
    parser.add_argument('--mmers', dest="mmers", default="4",
                        help="Mmer size to perform performance assesment (Comma separated)")
    parser.add_argument('--input_files', dest="input_files", help="List of paths to evaluate files (Comma separated)")
    parser.add_argument('--read_sizes', dest="read_sizes",
                        help="Read size of each file specified on --input_files option")
    parser.add_argument('--output_path', dest="output_path", default="output_superkmers",
                        help="Folder where the stats and output will be stored")
    parser.add_argument('--methods', dest="methods", default="kmerscl", help="Which method will be used to process reads (mspk or kmerscl), (comma separated for multiple)")
    parser.add_argument('--n_reads', dest="n_reads", default=None, help="Number of reads in each file (Comma separated values). If not specified this will be estimated")
    args = parser.parse_args()
    kmers = args.kmers.split(",")
    mmers = args.mmers.split(",")
    input_files = args.input_files.split(",")
    read_sizes = args.read_sizes.split(",")
    output_path = args.output_path
    methods = ["kmerscl"] if not args.methods else args.methods.split(",")
    n_reads = None if not args.n_reads else args.n_reads.split(",")
    # assert (len(input_files) == len(read_sizes), "Read sizes options are not of the same lenght of input_files options")
    return kmers, mmers, input_files, read_sizes, output_path, methods, n_reads

def execute_metrics_collection(full_output_path):
    # This should be async
    path = os.path.join(full_output_path, "metrics")
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
    cpu_command = "sar -P ALL 1 99999 > {}/sar_cpu_file.log".format(path)
    memio_command = "sar -b -r 1 99999 > {}/sar_mem_io_file.log".format(path)
    base_command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1"
    if hasattr(config, 'SPECIFIC_GPU'):
        base_command += " -i {}".format(config.SPECIFIC_GPU)
    nvidia_command = "{} | ts %s, >> {}/nvidia_gpu.log ".format(base_command, path)
    process_cpu = subprocess.Popen("LC_TIME='C' exec " + cpu_command, shell=True)
    process_memio = subprocess.Popen("LC_TIME='C' exec " +memio_command, shell=True)
    process_nvidia = subprocess.Popen("LC_TIME='C' exec " +nvidia_command, shell=True)
    return process_cpu, process_memio, process_nvidia

def execute_kmercl(params):
    # Sync
    params['output_path'] = "{output_path}/output_files".format(**params)
    command = "python2 -u getSuperK2_M.py --kmer {kmer} --mmer {mmer} --input_file {input_file} --read_size {read_size} --output_path {output_path}".format(**params)
    print "Executing {}".format(command)
    if params['n_reads']:
        command += " --n_reads {}".format(params['n_reads'])
    command += " | ts %s, > {log_output_path}".format(**params)
    sys.stdout.write("Executing '{}' \n".format(command))
    subprocess.call(command, shell=True)

def execute_kmercl_signature(params):
    # Sync
    params['output_path'] = "{output_path}/output_files".format(**params)
    command = "python2 -u getSuperK2_M_signature.py --kmer {kmer} --mmer {mmer} --input_file {input_file} --read_size {read_size} --output_path {output_path}".format(**params)
    print "Executing {}".format(command)
    if params['n_reads']:
        command += " --n_reads {}".format(params['n_reads'])
    command += " | ts %s, | tee {log_output_path}".format(**params)
    sys.stdout.write("Executing '{}' \n".format(command))
    subprocess.call(command, shell=True)

def execute_kmc(params):
    params['working_dir'] = "{output_path}tmp".format(**params)
    params['output_path'] = "{output_path}/output_files.res".format(**params)
    params['n_cores'] = THREADS
    command = KMC_PATH + " -k{kmer} -p{mmer} -t{n_cores} -fa {input_file} {output_path} {working_dir} | ts %s, | tee {log_output_path}".format(**params)
    sys.stdout.write("Executing '{}' \n".format(command))
    subprocess.call(command, shell=True)

# We need to copy mspk since is not possible to configure the output path and handling cd and stuff will be harder
def copyMSPK(params):
    print "Copying mspk to: {}".format(params['output_path'])
    copy_path = params['output_path']
    copyfile(os.path.join(MSPK_PARTITION_PATH, "Partition.class"), copy_path + "Partition.class")
    copyfile(os.path.join(MSPK_PARTITION_PATH, "Partition$MyThreadStep1.class"), copy_path +"Partition$MyThreadStep1.class")
    copyfile(os.path.join(MSPK_PARTITION_PATH, "guava-19.0.jar"), copy_path + "guava-19.0.jar")

def execute_mspk(params):
    params['output_path'] = os.path.join(params['output_path'])
    params['n_cores'] = THREADS
    command = "cd {output_path} && java -cp guava-19.0.jar: Partition -in {input_file} -k {kmer} -L {read_size} -p {mmer} -t {n_cores} | ts %s, > {log_output_path}".format(**params)
    sys.stdout.write("Executing '{}' \n".format(command))
    subprocess.call(command, shell=True)
    pass

def execute_sleep(seconds):
    sleep_time =  seconds
    command = "sleep {}".format(sleep_time)
    subprocess.call("exec "+command, shell=True)

def execute_metrics_summary(full_output_path):
    path = os.path.join(full_output_path, "metrics")
    sys.stdout.write("Mergin metrics in {}".format(path))
    merge_metrics(path, TOTAL_CORES, "2017-12-09")

def kill_processes(pids):
    sys.stdout.write("Killing metrics collection processes {}\n".format(pids))
    subprocess.call("killall -9 sar", shell=True)
    subprocess.call("killall -9 sar", shell=True)
    subprocess.call("killall -9 nvidia-smi", shell=True)
    subprocess.call("sudo sh {}".format(CLEAR_MEM_SCRIPT_PATH), shell=True)


def delete_output_files(output_path):
    os.system('rm -rf {}/output*'.format(output_path))
    os.system('rm -rf {}/Node*'.format(output_path))

def execute_assesment(kmer, mmer, input_file, read_size, output_path, method, n_reads):
    params = {'mmer': mmer, 'input_file_name': input_file.split("/")[-1], 'kmer': kmer, 'output_path': output_path,
              'read_size': read_size, 'input_file': input_file, "method": method, "n_reads": n_reads}
    full_output_path = os.path.join(params['output_path'], "{method}-k{kmer}-m{mmer}-r{read_size}-{input_file_name}/".format(**params))
    print full_output_path
    os.system('mkdir -p {}'.format(full_output_path))
    os.system('mkdir -p {}/tmp'.format(full_output_path))
    # Rewrite for specific output
    params['output_path'] = full_output_path
    params['log_output_path'] = os.path.join(full_output_path, "metrics", "tool_log.csv")

    # Copy this before metrics collection start
    if method == "mspk":
        copyMSPK(params)

    process_cpu, process_memio, process_nvidia = execute_metrics_collection(full_output_path)

    sys.stdout.write(" ***************************************** \n"\
              "Execution performance assesment \n"\
              "Output path {} \n"\
              "***********************************\n".format(full_output_path))

    sys.stdout.write("Letting metrics collection process to start \n")
    execute_sleep(3)
    if method == "kmerscl":
        execute_kmercl(params)
    if method == "kmerscl_signature":
        execute_kmercl_signature(params)
    if method == "kmc":
        execute_kmc(params)
    if method == "mspk":
        execute_mspk(params)
    if method == "sleep":
        execute_sleep(params)
    sys.stdout.write("Letting metrics collection process to finish \n")
    execute_sleep(3)
    kill_processes([process_cpu.pid, process_memio.pid, process_nvidia.pid])
    execute_metrics_summary(full_output_path)
    # To avoid the file system to get full
    delete_output_files(params['output_path'])

def main():
    kmers, mmers, input_files, read_sizes, output_path, methods, n_reads = parse_arguments()
    for kmer in kmers:
        for mmer in mmers:
            for method in methods:
                for idx, input_file in enumerate(input_files):
                    n_read = None if not n_reads else n_reads[idx]
                    try:
                        execute_assesment(kmer, mmer, input_file, read_sizes[idx], output_path, method, n_read)

                    except Exception as e:
                         sys.stdout.write("Exception {} generated with parameters {} \n".format(str(e), [kmer, mmer, input_file, read_sizes[idx], output_path, method]))

if __name__ == '__main__':
    main()
