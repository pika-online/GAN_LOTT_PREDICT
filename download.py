#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: taojinwnen

import requests
import collections
import json
import time
import pandas as pd

# 记录开始时间
start_time = time.time()

# 定义HTTP请求头
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
}

# 定义DataFrame的列
columns = ['开奖日期', '期号', '前区', '后区']
data = collections.OrderedDict()  # 使用有序字典存储数据
data['开奖日期'] = []
data['期号'] = []
data['前区'] = []
data['后区'] = []

# 循环爬取每一页的数据
for page_no in range(1, 86):
    url = f'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=30&isVerify=1&pageNo={page_no}'
    
    # 发送HTTP请求并获取响应
    response = requests.get(url, headers=header)
    
    # 检查响应状态码
    if response.status_code != 200:
        print(f"请求第 {page_no} 页数据失败，状态码：{response.status_code}")
        continue
    
    content = response.content.decode('utf-8')
    
    # 打印调试信息
    if not content:
        print(f"第 {page_no} 页数据为空。")
        continue
    
    try:
        json_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"解析第 {page_no} 页的 JSON 数据失败：{e}")
        continue
    
    print(f'正在处理第 {page_no} 页数据...')
    
    results = json_data.get('value').get('list')
    
    # 处理每一期的数据
    for entry in results:
        draw_date = entry.get('lotteryDrawTime')
        draw_num = entry.get('lotteryDrawNum')
        draw_result = entry.get('lotteryDrawResult')
        numbers = draw_result.split(' ')
        
        front_area = ' '.join(numbers[:5])
        back_area = ' '.join(numbers[-2:])
        
        # 将数据添加到对应的列表中
        data['开奖日期'].append(draw_date)
        data['期号'].append(draw_num)
        data['前区'].append(front_area)
        data['后区'].append(back_area)

# 创建DataFrame并导出到Excel
df = pd.DataFrame(data, columns=columns)
df.to_excel('dataset/dlt_results.xlsx', index=False)

# 记录结束时间并计算耗时
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时：{elapsed_time} 秒")
