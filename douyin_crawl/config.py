# coding=utf-8
import time
import pandas as pd
import json

def load_url_co():
    df = pd.read_excel('../mitmproxy/url_400.xls', encoding='utf-8-sig')
    for i, request in df.iterrows():
        # process cookies         
        c = json.loads(request['cookies'])
        cookie = [k+'='+c[k]+';' for k in c]
        # ok cookies
        cookie = ' '.join(cookie)[:-1]
        # process url        
        url = str(request['请求路径'])
        url_list = url.split('&')
        for p in range(0, 8000, 20):
            # time.sleep(0.5)
            ts = str(time.time()).split(".")[0]
            _rticket =str(time.time()*1000).split(".")[0]
            temp = ''
            for i in url_list:
                if 'cursor' in i:
                    i = i[:7]+str(p)
                elif 'count' in i:
                    i = i[:6]+str(p+20)
                elif 'ts=' in i:
                    i = i[:3]+str(ts)
                elif '_rticket' in i:
                    i = i[:9]+str(_rticket)
                temp += i+'&'
            # ok url
            url = temp[:-1]
            contin = yield (url, ts, _rticket, cookie)

            if contin:
                print('jump!!!!!!')
                break


if __name__ == "__main__":
    load_url_co()