import io
import os
import unittest
from magpie import Magpie
DATA_DIR1='data/Hamshahri.labels'
def test_cnn_train():
    with io.open(DATA_DIR1, 'r') as f:
        labels = {line.rstrip('\n') for line in f}


    model = Magpie()
    model.init_word_vectors('data/Hamshahri', vec_dim=100)
    history = model.train('data/Hamshahri', labels, nn_model='rnn', test_ratio=0.2, epochs=10)
    model.save_word2vec_model(filepath='save/embeddings/here')
    model.save_scaler(filepath='save/scaler/here', overwrite=True)
    model.save_model(filepath='save/model/here.h5')
    assert history is not None
    # predictions = model.predict_from_text("Black holes are cool!")
    # assert len(predictions) == len(labels)
    # print(predictions)
    # for lab, val in predictions:
    #     assert lab in labels
    #     assert 0 <= val <= 1
def test_cnn_test():
    with io.open(DATA_DIR1, 'r') as f:
        labels = {line.rstrip('\n') for line in f}
    model = Magpie(
        keras_model='save/model/here.h5',
        word2vec_model='save/embeddings/here',
        scaler='save/scaler/here',
        labels=labels
    )
    res=model.predict_from_file('data/Hamshahri/11.txt')
    print(res)

#test_cnn_train()
test_cnn_test()