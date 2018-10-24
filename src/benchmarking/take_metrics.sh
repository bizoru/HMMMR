mkdir -p $1
LC_TIME='C' sar -P ALL 1 99999 > $1/sar_cpu_file.log
LC_TIME='C' sar -b -r 1 99999 > $1/sar_mem_io_file.log
while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --id=05:00.0 --format=csv | ts %.s >> $1/nvidia_gpu.log ; sleep 1;  done
