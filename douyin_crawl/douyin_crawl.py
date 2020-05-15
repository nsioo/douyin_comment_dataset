# coding=utf-8
# Author:jiaxiang
import re
import requests
import time
import json


def get_message(item, strinfo):
    info = strinfo.split('.')
    try:
        for i in info:
            item = item.get(i)
    except:
        item = '获取信息出错'
    return item


params_index = {"version_code": "3.4.0",
                "pass-region": "1",
                "pass-route": "1",
                "js_sdk_version": "1.3.0.1",
                "app_name": "aweme",
                "vid": "9B26A672-3A7D-4B47-BD27-1DA5D8661F67",
                "app_version": "3.4.0",
                "device_id": "46343428354",
                "channel": "App%20Store",
                "aid": "1128",
                "screen_width": "1125",
                "openudid": "264245ce96bca574ff80597fe2bb3dc982739dc5",
                "os_api": "18",
                "ac": "WIFI",
                "os_version": "12.1.1",
                "device_platform": "iphone",
                "build_number": "34008",
                "iid": "53571850558",
                "device_type": "iPhone10,3",
                "idfa": "E368A825-2F49-4D9F-9A2E-4ED10BD9E686",
                # "min_cursor": "0",
                "user_id": "865166028726761",
                "count": "40",
                "max_cursor": str(int(round(time.time() * 1000))),
                "mas": "011c521e5b9b55669e4adcf236edebf3ba5c88d52834b9760c56a1",
                "as": "a1450500a6763ce93b6419",
                "ts": "1544247654"}
params_message = {'version_code': '3.4.0',
                  'pass-region': '1',
                  'pass-route': '1',
                  'js_sdk_version': '1.3.0.1',
                  'app_name': 'aweme',
                  'vid': '9B26A672-3A7D-4B47-BD27-1DA5D8661F67',
                  'app_version': '3.4.0',
                  'device_id': '46343428354',
                  'channel': 'App Store',
                  'aid': '1128',
                  'screen_width': '1125',
                  'openudid': '264245ce96bca574ff80597fe2bb3dc982739dc5',
                  'os_api': '18',
                  'ac': 'WIFI',
                  'os_version': '12.1.1',
                  'device_platform': 'iphone',
                  'build_number': '34008',
                  'iid': '53571850558',
                  'device_type': 'iPhone10,3',
                  'idfa': 'E368A825-2F49-4D9F-9A2E-4ED10BD9E686',
                  'cursor': '0',
                  'aweme_id': '6632470552571284739',  # 变化
                  'count': '20',
                  'insert_ids': '',
                  'mas': '01d53da165ba9f6281098e784d15e109913b6009c2fcadd39a64ee',
                  'as': 'a17584e0c6086c2b7b9455',
                  'ts': '1544244102'}
headers = {
    'cache-control': 'no-cache',
    'Postman-Token': '90cdf937-9447-47d5-865f-67b1a9cf8888',
    'User-Agent': 'PostmanRuntime/7.4.0',
    'Accept': '*/*',
    'Host': 'aweme.snssdk.com',
    'accept-encoding': 'gzip, deflate',
}
url_index = "https://aweme.snssdk.com/aweme/v1/aweme/post/"  # 获取全部信息
url_message = 'https://aweme.snssdk.com/aweme/v2/comment/list/'  # 获取回复信息


def write_to_json(filename, json_data):
    """
    将内容写入json
    :param filename: 要存储的文件名称
    :param json_data: 数据
    :return:
    """
    with open(filename, 'w', encoding='utf8') as f:
        content = json.dumps(dict(json_data), ensure_ascii=False)
        f.write(content)


def get_client_reply(aweme_id='6632470552571284739'):
    """
    获取视频下方的回复消息
    """
    params_message['aweme_id'] = aweme_id
    response = requests.get(url=url_message, headers=headers, params=params_message)
    json_data = response.json()
    client_items = json_data.get('comments')
    return client_items


def get_all_video_info(user_id='865166028726761'):
    """
    获取全部视频信息
    """
    params_index['user_id'] = user_id
    response = requests.get(url=url_index, headers=headers, params=params_index)
    json_data = response.json()
    aweme_list = json_data.get('aweme_list')
    return aweme_list





if __name__ == '__main__':
    print('1')
    aweme_list = get_all_video_info()

    print(aweme_list)

    for i, item in enumerate(aweme_list):
        print('-' * 20)
        print(f'第{i+1}个视频')
        print(f"标题：{get_message(item,'desc')}")
        aweme_id = get_message(item, 'aweme_id')
        if aweme_id:
            client_items = get_client_reply(aweme_id)
            if isinstance(client_items, list):
                for j, client_item in enumerate(client_items):
                    if re.search(r'买|多少|钱|微信|卖', get_message(client_item, 'text')):
                        print(f"第{j+1}条回复，"
                              f"{get_message(client_item,'user.nickname')}说:"
                              f"{get_message(client_item,'text')}")