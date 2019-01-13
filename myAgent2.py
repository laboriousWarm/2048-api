'''Build the test version of the agent.
'''
from game2048 import agents, game
import os
import numpy as np
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, Concatenate, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
'''使用分层数据策略的改良版本。出于阅读性考虑将使用中文注释。'''


class my_agent(agents.Agent):
    def __init__(self, game, max_depth=17, modelConfig=[], display=None):
        '''初始化
        
        Arguments:
            game {game.Game} -- 棋盘对象。如果是需要进行训练，该game对象需要可以允许重写。
        
        Keyword Arguments:
            max_depth {int} -- 描述了最大输入棋盘支持到2的多少次方。对于一般的棋盘，最大可能数字应该是17。(default: {17})
            modelConfig {list} -- 描述了使用多少个网络来建立自己的Agent以及他们的起始分数。不同的网络处理不同分段的问题，因此每个网络都可以获得较好的拟合情况。该参数要求一个长度为n的列表。
                                第n个参数代表了第n个网络负责分数小于等于该值的决策。将默认在队尾添加负责分数低于+inf的一个网络，因此如果只使用1个网络，该项省略即可。
            display {display.display} -- 表明将如何展示你的棋盘。 (default: {None})
        '''
        # 初始化基本参数
        self.game = game
        self.display = display
        self.max_depth = max_depth

        # 初始化网络列表和网络负责最大分列表
        assert isinstance(modelConfig, (tuple, list))
        modelConfig.append(np.inf)
        modelConfig = np.array(modelConfig)
        self.model_list = list()
        self.model_max_score = modelConfig
        for max_score in modelConfig:
            self.model_list.append(self.build_model(max_depth=max_depth))

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        '''Agent的核心步骤：决策下一步如何进行。
        在本Agent当中，不同分数阶段的决策由不同网络完成。
        
        Returns:
            int -- 所决策的方向
            '''
        # 将board转换为需要的输入形式
        board_now = self.game.board
        board_input = np.expand_dims(
            board2input(board_now, self.max_depth), axis=0)
        # 使用当前的score选择对应的网络
        score_now = self.game.score
        model_index = np.sum(score_now > self.model_max_score)

        # 使用对应网络完成决策
        choice = self.model_list[model_index].predict(board_input)[0]
        direction = np.where(np.max(choice) == choice)[0][0]

        return direction

    def build_model(self, max_depth):
        '''建立该模型中使用的model。该模型部分借鉴了ResNeXt的结构，但在深度和类型上有所适应性调整。也删除了ResNet中直接连接的部分（由于网络足够的浅）
        该模型由三个blocks连接两个全连接层之后连接SoftMax作为输出。所有激活函数使用LeakyReLu。
        blocks包含了五种形态不同的卷积核（1*4，4*1，2*2，3*3，4*4）。
        作为对老版本agent中model的改进（任务变简单），降低了参数量（7346W-->106W）
        '''
        x = Input(shape=(self.game.size, self.game.size, max_depth))
        # Conv Blocks
        y = self.add_blocks(x, 128)

        # Flatten&Dense Blocks
        # y = AveragePooling2D(pool_size=(self.game.size, self.game.size))(y)
        y = Flatten()(y)
        for num in [512,128]:
            y = Dense(num, kernel_initializer='he_uniform')(y)
            y = BatchNormalization()(y)
            y = LeakyReLU(alpha=0.2)(y)
        # Output
        y = Dense(4, activation='softmax')(y)
        model = Model(x, y)
        model.summary()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model

    def add_blocks(self, inputs, num_filters):
        '''这一部分详细的描述了blocks的配置：blocks包含了五种形态不同的卷积核（1*4，4*1，2*2，3*3，4*4）。
        inputs代表输入层，num_filters表明的卷积核的数量。
        '''

        conv14 = Conv2D(
            num_filters,
            kernel_size=(1, 4),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv41 = Conv2D(
            num_filters,
            kernel_size=(4, 1),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv22 = Conv2D(
            num_filters,
            kernel_size=(2, 2),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv33 = Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv44 = Conv2D(
            num_filters,
            kernel_size=(4, 4),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        outputs = keras.layers.add([conv14, conv41, conv22, conv33, conv44])
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = LeakyReLU(alpha=0.2)(outputs)
        return outputs

    def train(self,
              expert,
              train_batch_size=32,
              batches_data_training=1000,
              epoch_each_train=10,
              max_suffer=30):
        '''训练agent的所有网络。采用了一种混合式的在线学习和批学习策略（并不知道是否靠谱）。
            每次批生成数据，进行一定程度的训练之后更换数据。当分数能够稳定到达当前阶段的分数要求时，进行下一部分的网络训练。
            如果达到次数设定的批量，网络仍然在综合表现上不佳，将会终止训练。相关信息将被输出。
        
        Arguments:
            expert {Agent} -- 专家，被模仿的对象。棋盘的标签由专家标注。
            train_batch_size {int} -- 每一批使用的数据量。训练时数据被分为数个batch进行训练。
            batches_data_training {int} -- 在线学习中，每一次生成的数据量能够满足的batch数量。
            epoch_each_train {int} -- 在线学习中，生成的数据训练的轮数。
            max_suffer {int} -- 最大接受多少轮的在线学习而没有提升到能够进行下一个网络训练的分数。

        '''

        # 分模型进行训练
        for i, max_score in enumerate(self.model_max_score):
            # 确定该网络需要的棋盘分数段，重置模型训练完成标志
            if i == 0:
                min_score = 0
            else:
                min_score = self.model_max_score[i - 1]
            this_model_is_over = False

            print(
                'Start training with network %d , which is used with score [ %d , %d ]'
                % (i, min_score, max_score))

            # 构造该分数段对应的数据生成器
            # 这要求模型能够稳定的达到目的分数以上
            # 如何使得模型能够做到这点？
            training_generator = training_gene(
                expert,
                board_min=min_score,
                board_max=max_score,
                batch_size=train_batch_size)

            # 最坏在同一个网络上给出max_suffer次数据但无法得到较好的效果。
            for big_epoch in range(max_suffer):

                print('Online Studying with Big Epoch %d (Max: %d)' %
                      (big_epoch, max_suffer))

                # 使用生成器构建训练用棋盘数据
                training_x = list()
                print('creating dataset with size of %d * %d' %
                      (train_batch_size, batches_data_training))
                for i_gene in range(1,batches_data_training+1):
                    if i_gene % 200 == 0:
                        print('dataset creating flag:%d batches created.' %
                              (i_gene))
                    training_x.extend(training_generator.__next__())

                # 规范化输入
                print('normalizing...')
                training_y = list()
                for i_0, each in enumerate(training_x):
                    # 棋盘标注：用专家给出棋盘正解，同时将输入数据转换为需要格式
                    expert.game.board = np.array(each)
                    training_y.append(expert.step())
                    training_x[i_0] = board2input(
                        training_x[i_0], max_depth=self.max_depth)
                # one-hot 编码和 numpy.array类型转换
                training_x = np.array(training_x)
                training_y = to_categorical(training_y, num_classes=4)

                # 将数据分批进行训练,并不显示
                print('training...')
                h = self.model_list[i].fit(
                    training_x,
                    training_y,
                    batch_size=train_batch_size,
                    epochs=epoch_each_train,
                    verbose=0,
                    validation_split=0.1)

                # 单独的输出训练结果
                loss, val_loss, accu, val_accu = h.history['loss'], h.history[
                    'val_loss'], h.history['acc'], h.history['val_acc']

                for ep in range(len(loss)):
                    print(
                        'batch_epoch %d: loss = %.2f; val_loss= %.2f; accu = %.2f; val_accu = %.2f'
                        % (ep,loss[ep], val_loss[ep], accu[ep], val_accu[ep]))
                # 用该棋盘进行run_time局(eg.10)游戏并统计平均得分。如果达到了valid_rate(eg,0.9)水平的最高分，认为该部分网络合格（暂定）
                run_time = 10
                valid_rate = 1.0
                score_sum = 0
                empty_board = np.zeros((self.game.size, self.game.size))
                for i_0 in range(run_time):
                    # 初始化棋盘
                    self.game.board = empty_board
                    self.game._maybe_new_entry()
                    self.game._maybe_new_entry()
                    self.game.__end = False
                    # 进行游戏
                    n_iter = 0
                    while not self.game.end:
                        direction = self.step()
                        self.game.move(direction)
                        n_iter += 1
                    score_sum += self.game.score
                # 若该层模型进行训练较好则跳出循环
                print('average score = %.1f' % (score_sum / run_time))
                if score_sum >= max_score * valid_rate * run_time:
                    print(
                        'at big_epoch= %d , model %d train finished with ave score= %.1f >= %.1f'
                        % (big_epoch, i, score_sum / run_time, max_score * valid_rate))
                    this_model_is_over = True
                    break
                else:
                    print(
                        'at big_epoch= %d , model %d train continued with ave score= %.1f.'
                        % (big_epoch, i, score_sum / run_time))

            if this_model_is_over:
                continue
            else:
                print('model training failed at model %d : accuracy not reach.'
                      % i)
                break

    def load_model(self, path):
        '''加载模型。当包含多个模型时，也应该按照顺序给出名称。
        '''
        model_list = list()
        for this in path:
            model_list.append(keras.models.load_model(this))
        return model_list

    def save_model(self, path):
        '''保存模型。当包含多个模型时，也应该按照顺序给出名称。
        '''
        assert len(path) == len(self.model_list)
        for i in range(len(self.model_list)):
            keras.models.save_model(self.model_list[i], path[i])


def board2input(board, max_depth):
    '''reshape the board into the one that the network used.
    
    Arguments:
        board {ndarray}
        max_depth {int} -- the depth.
    '''
    size = len(board)
    board_input = np.zeros((size, size, max_depth))
    meshx, meshy = np.meshgrid(range(size), range(size))
    meshx, meshy = meshx.flatten(), meshy.flatten()
    for (x, y) in zip(meshx, meshy):
        if board[x, y] != 0:
            pos = int(np.log2(board[x, y]) - 1)
            board_input[x, y, pos] = 1
    # print(board_input.shape)
    return board_input


def training_gene(weak_agent, board_min=2, board_max=np.inf, batch_size=32):
    '''训练中使用的棋盘生成器。

    Arguments:
        weak_agent {Agent.agent} -- 被训练的Agent。使用其走出的局面来进行训练，有利于应对更多的情况。
        score_min {int} -- 收集棋盘的最低分。
        score_max {int} -- 收集棋盘的最高峰。只有分数在区间内的棋盘会被记录到buffer中并生成。该设计适用于分段作用的模型网络。
        batch_size {int} -- 每次生成的数据量。
    '''

    buffer = list()
    empty_board = np.zeros((weak_agent.game.size, weak_agent.game.size))

    while True:
        # 初始化一张新棋盘
        weak_agent.game.board = empty_board
        weak_agent.game._maybe_new_entry()
        weak_agent.game._maybe_new_entry()
        weak_agent.game.__end = False

        # 进行一场超过最高分就停止的游戏
        while (not weak_agent.game.end) and (weak_agent.game.score <=
                                             board_max):
            # 记录在分数区间的步骤
            if weak_agent.game.score > board_min:
                buffer.append(weak_agent.game.board)
            direction = weak_agent.step()
            weak_agent.game.move(direction)

        # 输出buffer中的内容，作为生成的小批数据
        while len(buffer) > batch_size:
            batch = buffer[:batch_size]
            yield (batch)
            buffer = buffer[batch_size:]


new_game = game.Game(size=4, enable_rewrite_board=True)
expert_agent=agents.ExpectiMaxAgent(new_game)
ag = my_agent(new_game, max_depth=17, modelConfig=[128, 512, 1024])
ag.train(
    expert_agent,
    train_batch_size=32,
    batches_data_training=1000,
    epoch_each_train=25,
    max_suffer=40)
ag.save_model(['level1.h5','level2.h5','level3.h5','level4.h5'])
