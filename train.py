# from __future__ import absolute_import, division, print_function, unicode_literals
# 设置 tensorflow 禁止使用 GPU
# 在import tensorflow之前
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import VGGSPP
import pre

BATCH_SIZE = 16 #batch大小128
EPOCH_SIZE = 20 #epoch大小20
MOM_BETA = 0.9 #momentum算法的摩擦系数
LEARN_RATE = 0.01 #梯度下降的学学习率


#图片输入模型时的大小
#这里设置为 pre 中调整的大小
IMG_INPUT_WIDTH = pre.IMG_PRE_OUT_WIDTH
IMG_INPUT_HEIGHT = pre.IMG_PRE_OUT_HEIGHT

##tensorflow gpu 版本时 设置显存申请方式
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     print(gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)


def show_loss_plot(loss_results, accuracy_results, img_type):
    """代价(损失)与准确率的可视化"""
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss_results)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(accuracy_results)
    plt.savefig(str(img_type) + '-' + 'show_loss.jpg')
    plt.show()


def train_vggspp(batch_size, epoch, img_type, dataLoader):
    """训练"""
    # 记录损失和准确率,用于画图
    train_loss_results = []
    train_accuracy_results = []
    # 构建VGG网络,可选vgg11;vgg16;vgg19
    model = VGGSPP.build_vggspp('vgg16')
    # 设置优化器(小批量梯度下降及momentum)
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARN_RATE, momentum=MOM_BETA)
    # 设置代价函数(交叉熵)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # 开始训练 
    for e in range(epoch):
        train_loss.reset_states()
        train_accuracy.reset_states()
        images_all, labels_all, num_iter = dataLoader.get_batch_train(batch_size)
        #在每个epoch中以batch为单位遍历训练集
        for i in range(0,num_iter):
            images = images_all[i]
            labels = labels_all[i]
            with tf.GradientTape() as tape:
                #单尺寸输入时
                if img_type == pre.CHANGE_GRAY_SIN or img_type == pre.CHANGE_BGR_SIN:
                    preds = model(images, training=True)  # 获取预测值
                #多尺寸输入时
                else:
                    preds = []
                    for image in images:
                        image = np.expand_dims(image,0)
                        preds.append(model(image, training=True))
                    preds = tf.convert_to_tensor(preds)
                    preds = tf.squeeze(preds,1)
                loss = loss_object(labels, preds)     # 计算损失
            #更新
            gradients = tape.gradient(loss, model.trainable_variables)           # 更新参数梯度
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # 更新参数
            train_loss(loss)                          # 更新损失
            train_accuracy(labels, preds)             # 更新准确率
        train_loss_results.append(train_loss.result())
        train_accuracy_results.append(train_accuracy.result())
        model.save_weights(str(img_type) + "-" + str (e+1) + "_epoch_weight.h5") #保存参数
        print('Epoch {}, Loss: {}, Accuracy: {}%'.format(e+1,train_loss.result(),train_accuracy.result()*100))
    show_loss_plot(train_loss_results, train_accuracy_results, img_type)

    # 用model.fit()训练
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # for e in range(epoch):
    #     num_iter  = dataLoader.num_train//batch_size
    #     for i in range(num_iter):
    #         images, labels = dataLoader.get_batch_train(batch_size)            
    #         model.fit(images, labels, batch_size=batch_size)
    #     model.save_weights(str (e+1) + "_epoch_vgg11_weight.h5")


def test_vggspp(model_path, img_type, dataLoader):
    """测试"""
    #构建VGGSPP网络 vgg11;vgg16;vgg19
    model = VGGSPP.build_vggspp('vgg16')
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #模型初始化
    if img_type == pre.CHANGE_BGR_SIN:
        model.build((None,IMG_INPUT_WIDTH,IMG_INPUT_HEIGHT,3))
    elif img_type == pre.CHANGE_GRAY_SIN:
        model.build((None,IMG_INPUT_WIDTH,IMG_INPUT_HEIGHT,1))
    #由SPP可知,逻辑上,下两个if分支的图片的宽高都应该是None,表示混合尺寸
    #但tensorflow不支持初始化时宽和高为None,所以给予某两个非零值
    #同样由SPP可知,该两个值任意充分大即可
    elif img_type == pre.CHANGE_BGR_MUL:
        model.build((None,IMG_INPUT_WIDTH,IMG_INPUT_HEIGHT,3))
    else:
        model.build((None,IMG_INPUT_WIDTH,IMG_INPUT_HEIGHT,1))

    try:
        model.load_weights(model_path)
    except:
        print("weight_load_error")

    # 显示模型网络结构
    print(model.summary())
    # 评估模型
    test_images, test_labels,= dataLoader.get_batch_test()
    ##输出准确率与代价
    #单尺寸时
    if img_type == pre.CHANGE_GRAY_SIN or img_type == pre.CHANGE_BGR_SIN:
        #################################################################
        # 输出
        model.evaluate(test_images, test_labels)
    #混合尺寸时
    else:
        loss_fin = float(0)
        accuracy_fin = float(0)

        i = 0 #用于遍历测试集中的label
        test_images_sum = 0 #纪录真实测试的图片数
        for image in test_images:
            label = test_labels[i]
            i = i+1
            image = np.expand_dims(image,0)
            label = np.array([label])
            #不知为何，有的测试图片数据输入时会出错，因此有test_images_sum与try与后面输出的rate
            #此 bug 修复失败，只能用 try 舍弃出 bug 的数据
            try:
                loss,accuracy = model.evaluate(image, label,verbose=0)
            except:
                continue
            test_images_sum = test_images_sum + 1
            loss_fin = loss_fin + loss
            accuracy_fin = accuracy_fin + accuracy
        loss_fin = loss_fin / test_images_sum
        accuracy_fin = accuracy_fin / test_images_sum
        #################################################################
        # 输出
        # rate为真实利用测试图片比率
        print("[==============================]","loss:",loss_fin,"-accuracy:",accuracy_fin,"**rate",test_images_sum/i)

def train_test (dataLoader,img_type,train_batch,train_epoch,test_weight_file):
    """训练和测试模型"""
    #训练
    train_vggspp(train_batch, train_epoch, img_type, dataLoader)
    #测试
    test_vggspp(test_weight_file, img_type, dataLoader)

if __name__ == '__main__':
    #四种 input_shape 情形
    img_type_list = [pre.CHANGE_GRAY_SIN,pre.CHANGE_GRAY_MUL,pre.CHANGE_BGR_SIN,pre.CHANGE_BGR_MUL]
    #最开始划分数据集，保证各种情形下的数据集结构相同
    files_name_train,files_name_test = pre.set_group()
    #按四种 input_shape 情形开始训练
    for img_type in img_type_list:
        print(str(img_type) + "-type-----------------------------------------------------------")
        dataLoader = DataLoader(img_type,files_name_train, files_name_test)
        train_test(dataLoader, img_type, BATCH_SIZE, EPOCH_SIZE, str(img_type) + '-' + str(EPOCH_SIZE) + '_epoch_weight.h5')