## About
This repo contains two parts:

1. A survey "多轮对话综述：过去、现在与应用" written by myself in 2019-12-21 as class proposal about multi turn dialogue system .
2. Multi turn dialogue models.

## Survey
"多轮对话综述：过去、现在与应用"成稿于2019-12-21，是我为国科大自然语言处理课程而完成的课堂论文。

Survey目录下包括一个PPT，是这篇论文作为优秀课堂论文在课上分享时所用。

### Why write this
为什么要写这样一篇：

- 一方面，最近的有关多轮对话的文章是京东在2017年发表的。最近几年是深度学习发展最快的几年，很多新方法论文没有覆盖到。
- 第二个方面是我们这次比赛只做了规则和端对端的模型，其实没有涉及到经典的管道模型，也是想借此机会更加了解。
- 第三是考虑到我在搭建JDDC比赛检索模型的时候，系统介绍多轮对话的中文文献比较少。然后，这样一篇综述可以帮助后来者知识的总结。

### Introduction
对于对话系统，根据不同的分类标准，有不同的划分方式。根据其应用领域，可以分为问答对话系统、闲聊对话系统和任务对话系统 (Chen et al. 2017a);根据其涉及的领域，可以分为开放领域对话系统和垂直领域对话系统;根据其是否使用历史对话信息，可以分为单轮对话系统和多轮对话系统 (Shang, Lu, and Li 2015; Mou et al. 2016);根据研究的发展历史和系统构建方式，可以分为第一代基于符号规则和模版的对话系统、第二代基于管道的对话系统和第三代大数据驱动的对话系统。

本文从上述划分方式出发，介绍了对话系统的发展。

## Models
### Data format
**Input**

For each sample:  
Input - list_history and query represented by ids  
Output - 2 class softmax result

One sample one line, and k utterances is split by "\t"

**Expected final output**

shape = (n_sample, n_label) where n_label = 0 (not related) or 1(related)
