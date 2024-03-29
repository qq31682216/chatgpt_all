## chatgpt_all 一站式本地GPT搭建工具

## 介绍
一个all_in_one合集，包含大模型微调、部署、langchain开发，方便大家使用
初步设想几个部分：
- [x] langchain例子：
    - [x] [简单的QA智能客服](/Simple_QA_customer_service)
    - [x] [简单的聊天机器人](/Simple_chatbot)
    - [x] [简单的Agent实现](/Simple_Agent)
    - [x] [简单的text2SQL](/text2SQL)
- [x] [统一微调](/yongyou_demo/train_qlora.py)
    - [x] [LLaMa2](/yongyou_demo/llama2-qlora.json)
    - [x] [Baichuan](/yongyou_demo/baichuan-13b-qlora.json)
    - [x] [chatglm2-6b](/yongyou_demo/yongyou_demo/chatglm2-6b-qlora.json)
- [ ] 统一api
    - [ ] chatglm2-6b
    - [ ] Baichuan
    - [ ] LLaMa2
- [x] 讲课demo快速入口，按时间排序
    - [x] [用友demo](/yongyou_demo)
    - [x] [华为demo](/huawei_demo)

## 说明
欢迎大家关注，后续持续更新，做方便你用的工具


## 其他说明:
如果要搭建一个本地的chatGPT模型，需要借助开源的力量，大体上要解决两个主要问题：基础大模型用什么，怎么用指令微调(instruction)。开源项目众多，鱼龙混杂，这么一个索引式记录可以让大家少走弯路，欢迎贡献。

## 数据
预训练数据：
指令数据：
## 热门项目汇总
| 项目名称 | github地址 | 基础大模型 | 训练方法 | 备注 |
| ---- | ----- | ------ | ---- | ---- |
| Alpaca | https://github.com/tatsu-lab/stanford_alpaca| LLaMA | Alpaca | 大家熟知的羊驼，入门推荐
| Alpaca | https://github.com/Facico/Chinese-Vicuna| LLaMA | Vicuna | 一个中文低资源的llama+lora方案

## 加微信交流群，备注: 公司/学校-方向
<img src="img/zhou759405.jpg" alt="微信号:zhou759405" width="300" height="300" />
