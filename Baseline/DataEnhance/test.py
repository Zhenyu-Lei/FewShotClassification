import http.client
import hashlib
import urllib
import random
import json
# import jieba
# import synonyms

from data_helper import create_dataloaders
from config import parse_args
import os
import numpy as np

os.chdir("..")


def test_translate():
    a = '本发明公开了一种耐磨、抗粘钢复合涂层、制备方法及应用，包括基体和基体上由内到外依次设置的过渡层、高硬基层、超硬表层；\
    过渡层采用焊材堆焊而成，焊材成分按重量计包括：0.02～0.04％C，16～19％Cr，1.2～1.6％Mn，0.5～0.8％Ni，0.2～0.4％Si，\
    余量为Fe；高硬基层采用焊材堆焊而成，焊材成分按重量计包括：0.7～0.8％C，3.8～5.8％Cr，0.6～1.0％Mn，0.7～1.5％Mo，\
    0.2～0.6％Si，0.4～0.8％V，8.1～9.1％W，余量为Fe；超硬表层采用金属基陶瓷材料喷涂而成，金属基陶瓷材料成分包括碳化钨和Co\
    ，本发明采用堆焊和喷涂的复合强化工艺，焊接层和喷涂层结合良好、硬度高，提升了矫直辊表面的硬化效果，改善了矫直辊辊面的耐磨性、\
    抗划伤性和抗粘钢性。'
    Lang_List = ["ru", "ja", "fr", "it", "kr"]
    print(trans_lang(trans_lang(a, "zh", "en"), "en", "zh"))
    print(trans_lang(trans_lang(a, "zh", "ja"), "ja", "zh"))
    print(trans_lang(trans_lang(a, "zh", "fr"), "fr", "zh"))


def trans_lang(q, source_Lang, destination_Lang):
    trans_result = q
    # 百度appid和密钥需要通过注册百度【翻译开放平台】账号后获得
    appid = '20221028001423163'  # 填写你的appid
    secretKey = 'PGVEv_4c81VbP7KVeN5k'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址

    fromLang = source_Lang  # 原文语种
    toLang = destination_Lang  # 译文语种
    salt = random.randint(32768, 65536)
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
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    return trans_result


def test_splitTest():
    seg_list = jieba.cut("中国上海是一座美丽的国际性大都市")
    replace_list = []
    # for x in seg_list:
    #     replace_list.append(sss := synonyms.nearby(x)[0][1])
    # print(replace_list)
    # print("Full Mode: " + "/ ".join(seg_list))
    #
    # print("人脸: ", synonyms.nearby("大都市"))


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        # get_synonyms 获取某个单词的同义词列表
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words


def data_Enhance(anns: list):
    expect = len(anns) * 0.36
    sum = np.zeros(36, dtype=float)
    for ann in anns:
        sum[int(ann['label'])] += 1


def test_statistic():
    args = parse_args()
    # train_dataloader, text_dataloader = create_dataloaders(args)
    sum = np.zeros(36, dtype=float)
    # for i, x in enumerate(train_dataloader):
    #     for a in x['label'].squeeze().numpy():
    #         sum[a] += 1
    # for i, x in enumerate(text_dataloader):
    #     for a in x['label'].squeeze().numpy():
    #         sum[a] += 1
    sum = np.array([
        [0.03340292, 0.01983299, 0.18789144, 0.03340292, 0.04384134, 0.03653445,
         0.0480167, 0.04906054, 0.04070981, 0.02609603, 0.05323591, 0.05532359,
         0.00730689, 0.01565762, 0.01983299, 0.03444676, 0.0177453, 0.01356994,
         0.01670146, 0.01356994, 0.02505219, 0.0125261, 0.00521921, 0.02296451,
         0.0302714, 0.01670146, 0.0177453, 0.02296451, 0.00835073, 0.00730689,
         0.01356994, 0.00417537, 0.00521921, 0.00835073, 0.0125261, 0.00521921]
    ])
    '''
    1/36=0.0277777777777778
    # [0.03340292 0.01983299 0.18789144 0.03340292 0.04384134 0.03653445
    #  0.0480167  0.04906054 0.04070981 0.02609603 0.05323591 0.05532359
    #  0.00730689 0.01565762 0.01983299 0.03444676 0.0177453  0.01356994
    #  0.01670146 0.01356994 0.02505219 0.0125261  0.00521921 0.02296451
    #  0.0302714  0.01670146 0.0177453  0.02296451 0.00835073 0.00730689
    #  0.01356994 0.00417537 0.00521921 0.00835073 0.0125261  0.00521921]
    '''
    print(sum / 958, 0.277 / sum.min(), 0.277 / sum.max())


if __name__ == "__main__":
    # test_translate()
    # test_splitTest()
    test_statistic()
