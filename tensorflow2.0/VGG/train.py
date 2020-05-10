from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import vgg
import tensorflow as tf
import json
import os


"""
因为VGG网络是在ImageNet上预训练过的，而且ImageNet数据集图像的RGB三个通道的均值分别为[123.68, 116.78, 103.94]，VGG在预训练过程中将图像的三个通道减去了各自的均值，所以如果要使用迁移学习使用VGG，那么需要将你自己的图像通道减去上面的均值才可以。
但是在该例子中没有减去，因为这里是将VGG网络在自己的数据集上从头训练的。
"""
# 因为VGG网络参数量为一亿多，加载到内存是大约500MB，这种模型是很大的。所以当数据集只有3000张图像的时候，不足以对模型进行训练，建议使用迁移学习的方式来利用VGG

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

# data generator with data augmentation
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices  # 这里是通过ImageDataGenerator得到花类别和012编码之间的对应关系的字典

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')
total_val = val_data_gen.n

model = vgg("vgg16", 224, 224, 5)
model.summary()

# using keras high level api for training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
# 通过ImageDataGenerator类得到的生成器会自动将label转换成one hot编码的形式，所以在这里的loss使用的是CategoticalCrossentropy，但是如果没有对label进行one hot编码的话，就要用SparseCategoricalCrossentropy损失
# 如果网络最后经过了softmax处理，也就是网络输出的是每个样本属于每个类别的概率，那么from_logits设置为False。反之要设置为True，但是设置为True，计算会更加稳定，所以有些人不在网络最后加softmax层

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex_{epoch}.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]
# 在保存模型的参数的时候，保存路径字符串中epoch是直接放在字符串中的，不知道对不对

# tensorflow2.1 recommend to using fit，但是也可以用fit_generator方法
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=callbacks)
