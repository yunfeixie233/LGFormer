import os
import urllib.parse
import urllib.request
import sys

config = sys.argv[1] if len(sys.argv) > 1 else "default_config"
status = sys.argv[2] if len(sys.argv) > 2 else "0"  # 默认状态为0 (结束)
server_name = sys.argv[3] if len(sys.argv) > 3 else "UnknownServer"  # 默认服务器名为 UnknownServer

def sc_send(text, desp='', key='[SENDKEY]'):
    postdata = urllib.parse.urlencode({'text': text, 'desp': desp}).encode('utf-8')
    url = f'https://sctapi.ftqq.com/{key}.send'
    req = urllib.request.Request(url, data=postdata, method='POST')
    with urllib.request.urlopen(req) as response:
        result = response.read().decode('utf-8')
    return result

data = {}
with open(os.path.join(os.path.dirname(__file__), '.env'), 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        data[key] = value
key = data['SENDKEY']

if config == "所有" and status == "0":
    status_send = f'{server_name} - 所有训练已停止'
elif status == "1":
    status_send = f'{server_name} - 训练开始'
    config = config.split("/")[-1]
else:
    status_send = f'{server_name} - 训练结束'    
    config = config.split("/")[-1]

ret = sc_send(status_send, config, key)
print(ret)
