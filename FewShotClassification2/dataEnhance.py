import numpy as np
import jieba
import random as rd
import http.client
import hashlib
import urllib
import json
import synonyms

Lang_List = ["ru", "ja", "fr", "it", "kr"]

# 概率为0.05
p = 0.05


# 随机删
def random_remove(abstract: list):
    for x in range(len(abstract)):
        if rd.uniform(0, 1) <= p:
            # 将该值删除
            x = ""


# 随机改
def random_replace(abstract: list):
    for x in range(len(abstract)):
        if rd.uniform(0, 1) <= p:
            ch_list = synonyms.nearby(abstract[x])[0]
            if len(ch_list) > 0:
                ch = rd.choice(ch_list)
                abstract[x] = ch


# 随机交换
def random_swap(abstract: list):
    for x in range(len(abstract)):
        if rd.uniform(0, 1) <= p:
            position = rd.randint(len(abstract) - 5, len(abstract) + 5)
            if position < 0 or position >= len(abstract):
                continue
            tmp = abstract[x]
            abstract[x] = abstract[position]
            abstract[position] = tmp


def work(abstract: str, i: int) -> str:
    # 随机改 随机删 随机交换，cn->["ru", "ja", "fr", "it", "kr"]->cn
    # 随机翻译
    if i & 1:
        lang = Lang_List[rd.randint(0, 4)]
        abstract = trans_lang(trans_lang(abstract, "cn", lang), lang, "cn")

    # 进行词分割
    seg_list = list(jieba.cut(abstract))
    random_swap(seg_list)
    random_replace(seg_list)
    random_remove(seg_list)
    return "".join(seg_list)


def data_Enhance(anns: list) -> list:
    count = 0
    result = anns.copy()
    # 期望总数为样本的10倍，然后样本均匀分布
    expect = len(anns) * 0.18
    # 统计各个样本出现次数
    sum = np.zeros(36, dtype=float)
    for ann in anns:
        sum[int(ann['label_id'])] += 1
    for ann in anns:
        iterator = 4
        print("process: {0},label: {1},needGenerate: {2}".format(count, ann['label_id'], iterator))
        count += 1
        for i in range(iterator):
            enhance_abstract = work(ann['abstract'], rd.randint(0, 1))
            ann_enhance = ann.copy()
            ann_enhance['abstract'] = enhance_abstract
            result.append(ann_enhance)
    return result


def trans_lang(q, source_Lang, destination_Lang):
    trans_result = q
    # 百度appid和密钥需要通过注册百度【翻译开放平台】账号后获得
    appid = '20221028001423163'  # 填写你的appid
    secretKey = 'PGVEv_4c81VbP7KVeN5k'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址

    fromLang = source_Lang  # 原文语种
    toLang = destination_Lang  # 译文语种
    salt = rd.randint(32768, 65536)
    # 手动录入翻译内容，q存放
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + \
            '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    # 建立会话，返回结果
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        trans_result = result['trans_result'][0]['dst']
    except Exception as e:
        pass
    finally:
        if httpClient:
            httpClient.close()
    return trans_result


if __name__ == '__main__':
    anns = list()
    with open('./data/train_div.json', 'r', encoding='utf8') as f:
        anns = json.load(f)
    result = data_Enhance(anns)
    # val_ratio = 0.1
    # rd.shuffle(anns)
    # val_anns = anns[:int(val_ratio * len(anns))]
    # train_anns = anns[int(val_ratio * len(anns)):]
    # with open('{}/train_div.json'.format('./data'), 'w', encoding='UTF-8') as fp:
    #     fp.write(json.dumps(train_anns, indent=2, ensure_ascii=False))
    # with open('{}/val_div.json'.format('./data'), 'w', encoding='UTF-8') as fp:
    #     fp.write(json.dumps(val_anns, indent=2, ensure_ascii=False))
    with open('{}/data_enhance.json'.format('./data'), 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(result, indent=2, ensure_ascii=False))
    print("成功写入文件。")
