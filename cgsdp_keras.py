import math
import random
import re

import tensorflow as tf
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from keras.models import Model
from keras.utils import to_categorical
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn import model_selection
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from stellargraph.layer import GAT
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
import keras_nlp

from utils.MyLabelBinarizer import *
from IPython.display import display, HTML

# Set random seed
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

######################################################################################################################
baseURL = "./data/code/"
######################################################################################################################


def CNN_transformer(main_input, embedding_matrix, project):
    class Squeeze(Layer):
        def __init__(self, axis):
            self.axis = axis
            super(Squeeze, self).__init__()

        def build(self, input_shape):
            pass

        def build(self, input_shape):
            pass

        def call(self, x):
            return tf.squeeze(x, self.axis)

        def compute_output_shape(self, input_shape):
            return input_shape[:-1]

    Squeeze_input = Squeeze(0)(main_input)
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                          output_dim=embedding_matrix.shape[1],
                          input_length=get_avg_len(project),
                          weights=[embedding_matrix],
                          mask_zero=True,
                          trainable=False)
    embed = embedding(Squeeze_input)
    X = Conv1D(32, kernel_size=5, padding='same', strides=1, activation='relu')(embed)
    X = Dropout(0.1)(X)
    X = MaxPooling1D(pool_size=math.floor(get_avg_len(project)/64))(X)
    X = X + keras_nlp.layers.PositionEmbedding(get_avg_len(project))(X)
    X = keras_nlp.layers.TransformerEncoder(32, 5)(X)
    X = Dropout(0.1)(X)
    X = MaxPooling1D(pool_size=int(X.shape[1]))(X)
    flat = Flatten()(X)
    hidden = Dense(32, activation='relu')(flat)
    output = tf.expand_dims(hidden, axis=0)
    model = Model(inputs=main_input, outputs=output)
    return model


def load_embedding_matrix(path):
    embed_matrix_file = pd.read_csv(path, header=0, index_col=False)
    embed_matrix = np.array(embed_matrix_file.iloc[:, 1:])
    return embed_matrix


def get_avg_len(project):
    tokens_integer = []
    len_list = []
    tokens_integer_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\token.txt", 'r')
    # tokens_integer_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\token_cross.txt", 'r')
    lines = tokens_integer_file.readlines()
    for each_line in lines:
        integer = each_line[each_line.index(' ') + 1:].strip('\n')
        integer_list = integer.split(',')
        tokens_integer.append(integer_list[:-1])
        len_list.append(len(integer_list[:-1]))
    max_len = max(len_list)
    return max_len


def sequence(tokens_seq, project):
    return pad_sequences(tokens_seq, maxlen=get_avg_len(project), padding='post', truncating='post')


def load_code_data(project, train_index, test_index):
    tokens_map = []
    tokens_map_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\token.txt", 'r')
    # tokens_map_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\token_cross.txt", 'r')
    lines = tokens_map_file.readlines()
    for each_line in lines:
        integer = each_line[each_line.index(' ') + 1:].strip('\n')
        integer_list = integer.split(',')
        tokens_map.append(integer_list[:-1])

    process = lambda data, label: (sequence(data, project), to_categorical(np.int64(label > 0)))
    origin_data = pd.read_csv(baseURL + project.split('-')[0] + "\\" + project + "\\" + project + ".csv", header=0,
                              index_col=False)
    Alldata = process(tokens_map, origin_data['bug'])

    X = Alldata[0]
    y = np.argmax(Alldata[1], axis=1)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test, Alldata


def graph_data(project):
    dataset = eval("sg.datasets." + re.sub("[-.]", "_", project) + "()")

    display(HTML(dataset.description))
    G, node_subjects = dataset.load()
    print(G.info())
    print(node_subjects.value_counts().to_frame())

    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.8, test_size=None, stratify=node_subjects, random_state=0
    )

    generator = FullBatchNodeGenerator(G, method="gat")

    target_encoding = MyLabelBinarizer()
    node_targets = target_encoding.fit_transform(node_subjects)
    train_targets = target_encoding.fit_transform(train_subjects)
    test_targets = target_encoding.fit_transform(test_subjects)

    gen = generator.flow(node_subjects.index, node_targets)
    train_gen = generator.flow(train_subjects.index, train_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    gat = GAT(
        layer_sizes=[64, 32], activations=["relu", "relu"], attn_heads=5, generator=generator,
        in_dropout=0.1, attn_dropout=0.1, normalize=None,
    )
    x_inp, x_out = gat.in_out_tensors()

    return x_inp, x_out, gen, train_gen, test_gen, train_subjects, test_subjects


def mix(inp1, inp2, r1, r2, Alldata, X_train, X_test, gen, train_gen, test_gen, project):
    alpha = 0.5
    concat = tf.keras.layers.Add()([alpha * r1, (1-alpha) * r2])

    dense1 = Dense(32, activation='relu')(concat)
    prediction = Dense(2, activation='softmax', name="pred_Layer")(dense1)

    merged_model = Model(inputs=[inp1, inp2], outputs=prediction)

    merged_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=['binary_crossentropy'],
                         metrics=['BinaryAccuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=50,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    data_in = np.expand_dims(Alldata[0], axis=0)
    train_in = np.expand_dims(X_train, axis=0)
    test_in = np.expand_dims(X_test, axis=0)
    history = merged_model.fit(x=[train_in, train_gen.inputs], y=train_gen.targets,
                               epochs=300, verbose=1,
                               validation_data=([test_in, test_gen.inputs], test_gen.targets),
                               callbacks=[learning_rate_reduction]
                               )

    merged_model.evaluate(x=[test_in, test_gen.inputs], y=test_gen.targets, verbose=2)

    embedding_model = tf.keras.Model(inputs=[inp1, inp2], outputs=dense1)
    emb = embedding_model.predict([data_in, gen.inputs])

    X = emb.squeeze(0)

    # save embeddings
    df = pd.DataFrame(X, columns=[('emb_' + str(i)) for i in range(X.shape[1])])
    df.to_csv(baseURL + project.split('-')[0] + "\\" + project + '\\cgsdp_emb_add.csv', index=False)
    df.to_csv(baseURL + project.split('-')[0] + "\\" + project + '\\cgsdp_cross_emb_add.csv', index=False)


if __name__ == '__main__':
    projects = [
        "ant-1.4","ant-1.6","ant-1.7",
    #     "camel-1.2","camel-1.4", "camel-1.6",
    #     "jedit-4.1", "jedit-4.2",
    #     "lucene-2.0","lucene-2.2", "lucene-2.4",
    #     "poi-2.0", "poi-2.5", "poi-3.0",
    #     "velocity-1.4","velocity-1.5","velocity-1.6",
    ]

    # cross project
    # projects = ["ant-1.7", "camel-1.6", "jedit-4.2", "lucene-2.4", "poi-3.0", "velocity-1.6"]

    for i in range(len(projects)):
        # load graph data
        x_inp, x_out, gen, train_gen, test_gen, train_subjects, test_subjects = graph_data(projects[i])

        embedding_matrix = load_embedding_matrix(
            baseURL + projects[i].split('-')[0] + "\\" + projects[i] + "\\output.csv")
        # embedding_matrix = load_embedding_matrix(
        #     baseURL + projects[i].split('-')[0] + "\\" + projects[i] + "\\output_cross.csv")

        # load code data
        X_train, y_train, X_test, y_test, Alldata = load_code_data(projects[i], train_subjects.index,
                                                                  test_subjects.index)
        inpCNN = Input(batch_shape=(1, None, get_avg_len(projects[i])), dtype='float64', name='cnn_input')
        modelCNN = CNN_transformer(inpCNN, embedding_matrix, projects[i])
        r1 = modelCNN.output
        r2 = x_out
        mix(inpCNN, x_inp, r1, r2, Alldata, X_train, X_test, gen, train_gen, test_gen, projects[i])
