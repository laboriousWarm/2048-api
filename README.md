# 2048-api  
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents  
#代码运行  
python myAgent2.py//训练模型  
python evaluate.py >> evaluate.log  
python generate_fingerprint.py  
在报告中注明的50次平均分为自己写的代码来判断50次的平均分  
在生成指纹和评估时要将myAgent2.py的代码最后的主体部分进行注释  
生成.log文件时评测次数为50次时，服务器和本机都会崩溃，因此只能采用原始的10次  
.h5文件，放至百度云  https://pan.baidu.com/s/1IwcMncP8vpiySilgHlwwHw  
4个.h5文件，分别对应0-128，128-512，512-1024，1024-2048 
# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).
