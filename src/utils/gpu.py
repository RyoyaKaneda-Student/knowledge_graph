import datetime
import subprocess
import time

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
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    return [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]


def print_info(JST):
    now = datetime.datetime.now(JST)
    info_ = get_gpu_info()[0]
    t = now.strftime('%H:%M:%S')
    s = "{}\tGPU: {:05}/{:05}".format(t, int(info_['memory.used']), int(info_['memory.total']))
    print("\r" + s + '  ', end="")


def main():
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    print('\n'*10)
    print('merciserv')
    while True:
        print_info(JST)
        time.sleep(1)


# pprint.pprint(get_gpu_info())

if __name__ == '__main__':
    main()
