import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
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
    # tasks = ['BBB', 'FDAMDD',  'Pgp_sub']

    task = 'FDAMDD'
    print(task)

    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
    medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
    large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights','addH':True}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10

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
    train_dataset, test_dataset , val_dataset = Graph_Classification_Dataset('data/clf/Ames.csv', smiles_field='SMILES',
                                                               label_field='Label',addH=True).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.5)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    auc= -10
    stopping_monitor = 0
    for epoch in range(100):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy_object.update_state(y,preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))

        y_true = []
        y_preds = []

        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)

        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_{}.h5'.format(task,seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>20:
            break

    y_true = []
    y_preds = []
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_preds = tf.sigmoid(y_preds).numpy()
    test_auc = roc_auc_score(y_true, y_preds)
    test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy))

    return test_auc

if __name__ == '__main__':

    auc_list = []
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        print(seed)
        auc = main(seed)
        auc_list.append(auc)
    print(auc_list)



