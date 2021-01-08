import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np

from dataset import Graph_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import  PredictModel,BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def main(seed):
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['H_HT', 'Pgp_inh', 'Pgp_sub']

    task = 'Ames'
    print(task)

    medium2 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2',
               'addH': True}
    small = {'name':'Small','num_layers': 3, 'num_heads': 4, 'd_model': 128,'path':'small_weights','addH':True}
    medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
    large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 516,'path':'large_weights','addH':True}
    medium_without_H = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_without_H','addH':False}
    medium_balanced = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_balanced',
                       'addH': True}
    medium_without_pretrain = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_without_pretraining_weights','addH':True}

    arch = medium ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 6

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset1, test_dataset1,val_dataset1 = Graph_Classification_Dataset('data\clf\Ames.txt', smiles_field='SMILES',
                                                               label_field='Label',addH=addH).get_data()
    train_dataset2, test_dataset2,val_dataset2 = Graph_Classification_Dataset('data\clf\BBB.txt', smiles_field='SMILES',
                                                                    label_field='Label', addH=addH).get_data()

    train_dataset3, test_dataset3, val_dataset3 = Graph_Classification_Dataset('data\clf\FDAMDD.txt',
                                                                               smiles_field='SMILES',
                                                                               label_field='Label',
                                                                               addH=addH).get_data()

    train_dataset4, test_dataset4, val_dataset4 = Graph_Classification_Dataset('data\clf\H_HT.txt',
                                                                               smiles_field='SMILES',
                                                                               label_field='Label',
                                                                               addH=addH).get_data()
    train_dataset5, test_dataset5, val_dataset5 = Graph_Classification_Dataset('data\clf\Pgp_inh.txt',
                                                                               smiles_field='SMILES',
                                                                               label_field='Label',
                                                                               addH=addH).get_data()

    train_dataset6, test_dataset6, val_dataset6 = Graph_Classification_Dataset('data\clf\Pgp_sub.txt',
                                                                               smiles_field='SMILES',
                                                                               label_field='Label',
                                                                               addH=addH).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset1.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.2)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, total_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.total_step = total_steps
            self.warmup_steps = total_steps*0.06

        def __call__(self, step):
            arg1 = step/self.warmup_steps
            arg2 = 1-(step-self.warmup_steps)/(self.total_step-self.warmup_steps)

            return 5e-5* tf.math.minimum(arg1, arg2)

    steps_per_epoch = len(train_dataset1)
    learning_rate = CustomSchedule(128,100*steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)

    auc= 0
    stopping_monitor = 0

    for epoch in range(100):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for  x1, adjoin_matrix1, y1 in train_dataset1:
            x2, adjoin_matrix2, y2 = next(iter(train_dataset2))
            x3, adjoin_matrix3, y3 = next(iter(train_dataset3))
            x4, adjoin_matrix4, y4 = next(iter(train_dataset4))
            x5, adjoin_matrix5, y5 = next(iter(train_dataset5))
            x6, adjoin_matrix6, y6 = next(iter(train_dataset6))

            with tf.GradientTape() as tape:
                seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
                mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
                preds1 = model(x1,mask=mask1,training=True,adjoin_matrix=adjoin_matrix1)
                # s1 = model.s[0]
                # s2 = model.s[1]
                # s3 = model.s[2]
                # s4 = model.s[3]
                # s5 = model.s[4]
                # s6 = model.s[5]
                loss1 = loss_object(y1,preds1[:,0])*10

                seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)
                mask2 = seq2[:, tf.newaxis, tf.newaxis, :]
                preds2 = model(x2,mask=mask2,training=True,adjoin_matrix=adjoin_matrix2)


                loss2 = loss_object(y2, preds2[:,1])
                seq3 = tf.cast(tf.math.equal(x3, 0), tf.float32)
                mask3 = seq3[:, tf.newaxis, tf.newaxis, :]
                preds3 = model(x3, mask=mask3, training=True, adjoin_matrix=adjoin_matrix3)

                loss3 = loss_object(y3, preds3[:,2])

                seq4 = tf.cast(tf.math.equal(x4, 0), tf.float32)
                mask4 = seq4[:, tf.newaxis, tf.newaxis, :]
                preds4 = model(x4, mask=mask4, training=True, adjoin_matrix=adjoin_matrix4)

                loss4 = loss_object(y4, preds4[:, 3])

                seq5 = tf.cast(tf.math.equal(x5, 0), tf.float32)
                mask5 = seq5[:, tf.newaxis, tf.newaxis, :]
                preds5 = model(x5, mask=mask5, training=True, adjoin_matrix=adjoin_matrix5)

                loss5 = loss_object(y5, preds5[:, 4])

                seq6 = tf.cast(tf.math.equal(x6, 0), tf.float32)
                mask6 = seq6[:, tf.newaxis, tf.newaxis, :]
                preds6 = model(x6, mask=mask6, training=True, adjoin_matrix=adjoin_matrix6)

                loss6 = loss_object(y6, preds6[:, 5])

                loss = loss1+loss2+loss3+loss4+loss5+loss6
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()))


        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset1:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:,0].numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)
        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc :{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset2:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:,1].numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true, y_preds)

        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc:{:.4f}'.format( auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset3:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:,2].numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true, y_preds)
        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc :{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset4:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:, 3].numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true, y_preds)
        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc :{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset5:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:, 4].numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true, y_preds)
        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc :{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in test_dataset6:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds[:, 5].numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true, y_preds)
        test_accuracy = keras.metrics.binary_accuracy(y_true, y_preds).numpy()
        print('test auc :{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))


    return auc

if __name__ == '__main__':

    auc_list = []
    for seed in [7,17,27,37,47]:
        print(seed)
        auc = main(seed)
        auc_list.append(auc)
    print(auc_list)



