import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np

from dataset import Graph_Regression_Dataset
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import  PredictModel,BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



def main(seed):
    # tasks = ['caco2', 'logD', 'logS', 'PPB', 'tox']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights','addH':True}
    medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights','addH':True}
    medium2 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2',
               'addH': True}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights','addH':True}
    medium_without_H = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H','addH':False}
    medium_without_pretrain = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256,'path': 'medium_without_pretraining_weights','addH':True}

    arch = medium ## small 3 4 128   medium: 6 6  256     large:  12 8 516

    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10
    task = 'PPB'
    print(task)
    seed = seed

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    dropout_rate = 0.1

    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset,val_dataset = Graph_Regression_Dataset('data/reg/{}.txt'.format(task), smiles_field='SMILES',
                                                           label_field='Label',addH=addH).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.15)
    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, total_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.total_step = total_steps
            self.warmup_steps = total_steps*0.10

        def __call__(self, step):
            arg1 = step/self.warmup_steps
            arg2 = 1-(step-self.warmup_steps)/(self.total_step-self.warmup_steps)

            return 10e-5* tf.math.minimum(arg1, arg2)

    steps_per_epoch = len(train_dataset)
    learning_rate = CustomSchedule(128,100*steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)

    r2 = -10
    stopping_monitor = 0
    for epoch in range(100):
        mse_object = tf.keras.metrics.MeanSquaredError()
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = tf.reduce_mean(tf.square(y-preds))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mse_object.update_state(y,preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'mse: {:.4f}'.format(mse_object.result().numpy().item()))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        r2_new = r2_score(y_true,y_preds)

        test_mse = keras.metrics.MSE(y_true, y_preds).numpy()
        print('val r2: {:.4f}'.format(r2_new), 'val mse:{:.4f}'.format(test_mse))
        if r2_new > r2:
            r2 = r2_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('regression_weights/{}.h5'.format(task))
        else:
            stopping_monitor +=1
        print('best r2: {:.4f}'.format(r2))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>20:
            break

    y_true = []
    y_preds = []
    model.load_weights('regression_weights/{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)

    test_auc = r2_score(y_true, y_preds)
    test_accuracy = keras.metrics.MSE(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    print('test r2:{:.4f}'.format(test_auc), 'test mse:{:.4f}'.format(test_accuracy))


    return r2

if __name__ == "__main__":

    r2_list = []
    for seed in [7,17,27,37,47]:
        print(seed)
        r2 = main(seed)
        r2_list.append(r2)
    print(r2_list)

