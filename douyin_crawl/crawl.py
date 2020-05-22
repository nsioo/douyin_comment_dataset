#coding=utf-8

import hashlib
from urllib import request, parse
import time
from io import StringIO
import gzip
import json
import os


from config import load_url_co



byteTable1 ="D6 28 3B 71 70 76 BE 1B A4 FE 19 57 5E 6C BC 21 B2 14 37 7D 8C A2 FA 67 55 6A 95 E3 FA 67 78 ED 8E 55 33 89 A8 CE 36 B3 5C D6 B2 6F 96 C4 34 B9 6A EC 34 95 C4 FA 72 FF B8 42 8D FB EC 70 F0 85 46 D8 B2 A1 E0 CE AE 4B 7D AE A4 87 CE E3 AC 51 55 C4 36 AD FC C4 EA 97 70 6A 85 37 6A C8 68 FA FE B0 33 B9 67 7E CE E3 CC 86 D6 9F 76 74 89 E9 DA 9C 78 C5 95 AA B0 34 B3 F2 7D B2 A2 ED E0 B5 B6 88 95 D1 51 D6 9E 7D D1 C8 F9 B7 70 CC 9C B6 92 C5 FA DD 9F 28 DA C7 E0 CA 95 B2 DA 34 97 CE 74 FA 37 E9 7D C4 A2 37 FB FA F1 CF AA 89 7D 55 AE 87 BC F5 E9 6A C4 68 C7 FA 76 85 14 D0 D0 E5 CE FF 19 D6 E5 D6 CC F1 F4 6C E9 E7 89 B2 B7 AE 28 89 BE 5E DC 87 6C F7 51 F2 67 78 AE B3 4B A2 B3 21 3B 55 F8 B3 76 B2 CF B3 B3 FF B3 5E 71 7D FA FC FF A8 7D FE D8 9C 1B C4 6A F9 88 B5 E5"
def getXGon(url,stub,cookies):
    NULL_MD5_STRING = "00000000000000000000000000000000"
    sb=""
    if len(url)<1 :
        sb =NULL_MD5_STRING
    else:
        sb =encryption(url)
    if len(stub)<1:
        sb+=NULL_MD5_STRING
    else:
        sb+=stub
    if len(cookies)<1:
        sb+=NULL_MD5_STRING
    else:
        sb+=encryption(cookies)
    try:
        index = cookies.index("sessionid=")
    except:
        index = -1
    if index == -1:
        sb+=NULL_MD5_STRING
    else:
        sessionid = cookies[index+10:]
        if sessionid.__contains__(';'):
            endIndex = sessionid.index(';')
            sessionid = sessionid[:endIndex]
        sb+=encryption(sessionid)
    return sb

def encryption(url):
    obj = hashlib.md5()  # 先创建一个md5的对象
    # 写入要加密的字节
    obj.update(url.encode("UTF-8"))
    # 获取密文
    secret = obj.hexdigest()
    return secret.lower()


def initialize(data):
    myhex = 0
    byteTable2 = byteTable1.split(" ")
    for i in range(len(data)):
        hex1 = 0
        if i==0:
            hex1= int(byteTable2[int(byteTable2[0],16)-1],16)
            byteTable2[i]=hex(hex1)
            # byteTable2[i] = Integer.toHexString(hex1);
        elif i==1:
            temp=   int("D6",16)+int("28",16)
            if temp>256:
                temp-=256
            hex1 = int(byteTable2[temp-1],16)
            myhex = temp
            byteTable2[i] = hex(hex1)
        else:
            temp = myhex+int(byteTable2[i], 16)
            if temp > 256:
                temp -= 256
            hex1 = int(byteTable2[temp - 1], 16)
            myhex = temp
            byteTable2[i] = hex(hex1)
        if hex1*2>256:
            hex1 = hex1*2 - 256
        else:
            hex1 = hex1*2
        hex2 = byteTable2[hex1 - 1]
        result = int(hex2,16)^int(data[i],16)
        data[i] = hex(result)
    for i in range(len(data)):
        data[i] = data[i].replace("0x", "")
    return data


def handle(data):
    for i in range(len(data)):
        byte1 = data[i]
        if len(byte1)<2:
            byte1+='0'
        else:
            byte1 = data[i][1] +data[i][0]
        if i<len(data)-1:
            byte1 = hex(int(byte1,16)^int(data[i+1],16)).replace("0x","")
        else:
            byte1 = hex(int(byte1, 16) ^ int(data[0], 16)).replace("0x","")
        byte1 = byte1.replace("0x","")
        a =  (int(byte1, 16) & int("AA", 16)) / 2
        a = int(abs(a))
        byte2 =((int(byte1,16)&int("55",16))*2)|a
        byte2 = ((byte2&int("33",16))*4)|(int)((byte2&int("cc",16))/4)
        byte3 = hex(byte2).replace("0x","")
        if len(byte3)>1:
            byte3 = byte3[1] +byte3[0]
        else:
            byte3+="0"
        byte4 = int(byte3,16)^int("FF",16);
        byte4 = byte4 ^ int("14",16)
        data[i] = hex(byte4).replace("0x","")
    return data


def xGorgon(timeMillis,inputBytes):
    data1 = []
    data1.append("3")
    data1.append("61")
    data1.append("41")
    data1.append("10")
    data1.append("80")
    data1.append("0")
    data2 = input(timeMillis,inputBytes)
    data2 = initialize(data2)
    data2 = handle(data2)
    for i in range(len(data2)):
        data1.append(data2[i])

    xGorgonStr = ""
    for i in range(len(data1)):
        temp = data1[i]+""
        if len(temp)>1:
            xGorgonStr += temp
        else:
            xGorgonStr +="0"
            xGorgonStr+=temp
    return xGorgonStr

def input(timeMillis,inputBytes):
    result = []
    for i in range(4):
        if inputBytes[i]<0:
            temp = hex(inputBytes[i])+''
            temp = temp[6:]
            result.append(temp)
        else:
            temp = hex(inputBytes[i]) + ''
            result.append(temp)
    for i in range(4):
        result.append("0")
    for  i in range(4):
        if inputBytes[i+32]<0:
            result.append( hex(inputBytes[i+32])+'')[6:]
        else:
            result.append(hex(inputBytes[i + 32]) + '')
    for i in range(4):
        result.append("0")
    tempByte = hex(int(timeMillis))+""
    tempByte = tempByte.replace("0x","")
    for i in range(4):
        a = tempByte[i * 2:2 * i + 2]
        result.append(tempByte[i*2:2*i+2])
    for i in range(len(result)):
        result[i] = result[i].replace("0x","")
    return result
def strToByte(str):
    length = len(str)
    str2 = str
    bArr =[]
    i=0
    while i < length:
        # bArr[i/2] = b'\xff\xff\xff'+(str2hex(str2[i]) << 4+str2hex(str2[i+1])).to_bytes(1, "big")
        a = str2[i]
        b = str2[1+i]
        c = ((str2hex(a) << 4)+str2hex(b))
        bArr.append(c)
        i+=2
    return bArr

def str2hex(s):
    odata = 0;
    su =s.upper()
    for c in su:
        tmp=ord(c)
        if tmp <= ord('9') :
            odata = odata << 4
            odata += tmp - ord('0')
        elif ord('A') <= tmp <= ord('F'):
            odata = odata << 4
            odata += tmp - ord('A') + 10
    return odata
def doGetGzip(url,headers,charset):
    req = request.Request(url)
    for key in headers:
        req.add_header(key,headers[key])
    with request.urlopen(req) as f:
        
        data = f.read()
        # print(gzip.decompress(data))
        # return gzip.decompress(data)
        return data


def doPostGzip(url,headers,charset, params):
    data = parse.urlencode(params).encode(encoding='UTF8')
    req = request.Request(url)
    for key in headers:
        req.add_header(key,headers[key])
    with request.urlopen(req, data=data) as f:
        data = f.read()
        return gzip.decompress(data).decode()




if __name__=="__main__":

    save_to  = './result_v3.txt'
    if os.path.exists(save_to):
        # os.remove('./comment.txt') # 是否清除上次的文档
        open(save_to, 'w', encoding='utf-8-sig')
    
    good = 0
    error = 0
    generator = load_url_co()
    for url, ts, _rticket, cookies in generator:
        params = url[url.index('?')+1:]
        STUB = ""
        
        s = getXGon(params,STUB,cookies)
        gorgon = xGorgon(ts,strToByte(s))
        
        headers={
            "X-Gorgon":gorgon,
            # "X-SS-REQ-TICKET": "1585711173953",
            "X-Khronos": ts,
            "sdk-version":"1",
            "Accept-Encoding": "gzip",
            "X-SS-REQ-TICKET": _rticket,
            "User-Agent": "ttnet okhttp/3.10.0.2",
            "Host": "aweme.snssdk.com",
            "Cookie": cookies,
            "Connection": "Keep-Alive",
            # "x-tt-token":"00080ab789c0bf0519740314c59de87d8ace96d49d8ab2afd7a0f09cba0911612f99baf92acae289860e0f84ffd97fc2c344"
        }
    
        cet = []
        try:
            result =doGetGzip(url,headers,"UTF-8")
            result = gzip.decompress(result)
            # print(result.encode(''))
            result = json.loads(result)
            l1 = result['comments']
            for c in l1:
                if isinstance(c['reply_comment'], list):
                    cet.append(c['reply_comment']['text'])
                cet.append(c['text'])
                print('good!!!')
                good += 1
        except:
            print('error!!!')
            # print(result)
            error += 1
            if '"has_more":0' in str(result):
                generator.send( True)
        
        # save
        with open(save_to, 'a', encoding='utf-8-sig') as f:
            for c in cet:
                f.write(c+'\n')

    print('=='*100)
    print('end, success :{0}, fail:{1}'.format(str(good), str(error)))