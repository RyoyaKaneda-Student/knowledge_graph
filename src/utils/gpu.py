import datetime
import subprocess
import time
import psutil

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)


def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    try:
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE)
        lines = output.decode().split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']
        rev = [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]
    except subprocess.CalledProcessError as e:
        rev = None
    return rev


def print_info(JST):
    now = datetime.datetime.now(JST)
    t = now.strftime('%H:%M:%S')
    s = f"{t}"
    mem = psutil.virtual_memory()
    cpu_used, cpu_total = mem.used, mem.total
    cpu_used_percent = cpu_used/cpu_total*100
    s += "\tCPU: {}/{} ({}%)".format(int(cpu_used), int(cpu_total), int(cpu_used_percent))
    info_ = get_gpu_info()
    if info_ is not None:
        info_ = info_[0]
        s += "\tGPU: {}/{}".format(int(info_['memory.used']), int(info_['memory.total']))
    else:
        s += "\tGPU: GPU: ----------"

    print("\r" + s + '  ', end="")


def main():
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    # print('\n' * 10)
    print('merciserv')
    while True:
        print_info(JST)
        time.sleep(1)


# pprint.pprint(get_gpu_info())

if __name__ == '__main__':
    main()
