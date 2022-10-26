#  Import Dependencies
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import ModelCheckpoint


path = "./"

LABELS = ["pessoa", "capacete", "luvas", "colete", "mascara"]

IMG_WIDTH, IMG_HEIGHT = 135, 240 #54, 96 #108, 192 

n_epoca = 1000

n_pasta = '2'

BATCH_SIZE = 128 #64

#Uma coisa importante sobre o Mini-Batch é que, é melhor escolher o tamanho do Mini-Batch como múltiplo de 2 e os valores comuns são: 64, 128, 256 e 512.
# https://www.deeplearningbook.com.br/definindo-o-tamanho-do-mini-batch/

"""### Convert multi-hot labels to string labels"""
def covert_onehot_string_labels(label_string, label_onehot):
  labels = []
  for i, label in enumerate(label_string):
     if label_onehot[i]: #se for 1, adiciona o nome do label
       labels.append(label)
     if len(labels) == 0: #se o vetor tiver vazio, então não todos os labels = 0, logo tem ausencia NONE
       labels.append("NONE")
  return labels

ds_train = tf.data.Dataset.load(path + "split_dataset/train/")
ds_validation = tf.data.Dataset.load(path + "split_dataset/validation/")
ds_test = tf.data.Dataset.load(path + "split_dataset/test/")

#https://www.tensorflow.org/api_docs/python/tf/data/Dataset - todas as funções tf.data.dataset

ds_train_batched = ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE) #tf.data.experimental.AUTOTUNE
ds_validation_batched = ds_validation.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE) 
ds_test_batched = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("Number of batches in train: ", ds_train_batched.cardinality().numpy())
print("Number of batches in validation: ", ds_validation_batched.cardinality().numpy())
print("Number of batches in test: ", ds_test_batched.cardinality().numpy())

"""
prefetch:
Ele pode ser usado para desacoplar o momento em que os dados são produzidos do momento em que os dados são consumidos. Em particular, a transformação usa um thread em segundo plano e um buffer interno para pré-buscar elementos do conjunto de dados de entrada antes do momento em que são solicitados. O número de elementos para pré-busca deve ser igual (ou possivelmente maior que) ao número de lotes consumidos por uma única etapa de treinamento. Você pode ajustar manualmente esse valor ou defini-lo como tf.data.AUTOTUNE , que solicitará que o tempo de execução tf.data ajuste o valor dinamicamente em tempo de execução.

cache:
pode armazenar em cache um conjunto de dados, na memória ou no armazenamento local. Isso evitará que algumas operações (como abertura de arquivos e leitura de dados) sejam executadas durante cada época.
"""

"""
# 2. Create a Keras CNN model by using Transfer learning
"""

base_model = keras.applications.ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.
base_model.trainable = False


base_model.summary() #show_trainable=True


"""## Create the classification model"""
number_of_classes = len(LABELS)

inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)) #adiciona uma camada de entrada com a tamanho das imagens

x = base_model(inputs, training=False) #adiciona a rede pre treinada como não treinavel

x = keras.layers.GlobalAveragePooling2D()(x) #realiza o pooling

initializer = tf.keras.initializers.GlorotUniform(seed=42) #inicialização dos pesos
activation =  tf.keras.activations.sigmoid #função de ativação -> sigmoid(x) = 1 / (1 + exp(-x))
outputs = keras.layers.Dense(number_of_classes,
                             kernel_initializer=initializer,
                             activation=activation)(x) #adiciona uma camada densa

model = keras.Model(inputs, outputs)

model.summary() #show_trainable=True

"""**GlobalAveragePooling2D()**
entrada: tensor 4D com forma (batch_size, rows, cols, channels). [None, 7, 7, 512] - saida do model
saida: 2D tensor with shape (batch_size, channels). [None, 512]

https://keras.io/api/models/model/

**tf.keras.initializers.GlorotUniform**
seed: Um inteiro Python. Usado para tornar o comportamento do inicializador determinístico. Observe que um inicializador semeado não produzirá os mesmos valores aleatórios em várias chamadas, mas vários inicializadores produzirão a mesma sequência quando construídos com o mesmo valor de semente.

**camada densa e argumentos:**
https://keras.io/api/layers/core_layers/dense/
"""

"""
# 3. Compile & Train
"""

model.compile(optimizer=keras.optimizers.Adam(), #otimizador que implementa o algoritmo Adam
              loss=keras.losses.BinaryCrossentropy(), #calcula a perda de entropia cruzada binária
              metrics=[keras.metrics.BinaryAccuracy()]) #calcula a frequência com que as previsões correspondem aos rótulos binários

"""
**IMPORTANT:** We need to use **keras.metrics.BinaryAccuracy()** for **measuring** the **accuracy** since it calculates how often predictions matches **binary labels**. 
As we are dealing with **multi-label** classification and true lables are encoded **multi-hot**, we need to compare ***pairwise (binary!)***:  each element of prediction with the corresponding element of true lables.
"""
start_time = time.perf_counter()
       
history = model.fit(ds_train_batched, validation_data=ds_validation_batched, epochs=n_epoca, callbacks=ModelCheckpoint(path+'models/'+n_pasta+'/best_epoch_model.hdf5', save_best_only=True, monitor='val_binary_accuracy', mode='max'))

end_time = time.perf_counter()
train_time = end_time - start_time
#print("Finished in {}".format(time.strftime("%H:%M:%S", time.gmtime(train_time))))
train_time = time.strftime("%H:%M:%S", time.gmtime(train_time))

"""
**model.compile / model.fit / model.evaluate / model.predict**
https://keras.io/api/models/model_training_apis/
"""

#salvar modelo e pesos
model.save(path + "models/"+n_pasta+"/model.h5")

#salvar history de treinamento e validação
hist_df = pd.DataFrame(history.history) 
with open("csv_files/"+n_pasta+"/history.csv", mode='w') as f:
    hist_df.to_csv(f)


start_time = time.perf_counter()
"""## all samples predictions"""
predictions = model.predict(ds_test.batch(batch_size=10))
end_time = time.perf_counter()
test_time = end_time - start_time
test_time = time.strftime("%H:%M:%S", time.gmtime(test_time))

# salva as previsões reais - valores floats da saida da rede
file = open("csv_files/"+n_pasta+"/predict.csv", "w")
writer = csv.writer(file)
for pred in zip(predictions):
  #print("predicted: " ,pred)
  writer.writerow([pred])
file.close()

# salva as previsões com as condições aplicadas e as reais (onehot e string)
file = open("csv_files/"+n_pasta+"/predict_true.csv", "w")
writer = csv.writer(file)
list_true = []
list_predict = []
for (pred,(a,b)) in zip(predictions, ds_test):
  pred[pred>0.5] = 1
  pred[pred<=0.5] = 0
  # print("predicted: ", pred, str(covert_onehot_string_labels(LABELS, pred)),  
  #       "Actual Label: ("+str(covert_onehot_string_labels(LABELS, b.numpy())) +")")
  writer.writerow([pred, b.numpy(), str(covert_onehot_string_labels(LABELS, pred)), str(covert_onehot_string_labels(LABELS,b.numpy()))])
  list_predict.append(pred)
  list_true.append(b.numpy())
file.close()

#apresenta e salva o relatorio de classificação
print("Relatório de Classificação:\n", classification_report(list_true, list_predict, target_names=LABELS))
report = pd.DataFrame(classification_report(list_true, list_predict, target_names=LABELS, output_dict=True)).transpose()
report.to_csv('csv_files/'+n_pasta+'/classification_report.csv', index= True)


# 4. Evaulate the model
eval_model = model.evaluate(ds_test_batched) #ou val?
#print(eval_model)
accuracy = accuracy_score(list_true, list_predict)
#print("Acurácia: {:.4f}\n".format(accuracy)) #considera o conjunto

file = open("csv_files/"+n_pasta+"/evaluate_accuracy.csv", "w")
writer = csv.writer(file)
writer.writerow([eval_model, accuracy])
writer.writerow([train_time])
writer.writerow([test_time])
file.close()

# Utility function for plotting of the model results
def visualize_results(history, n_pasta):
    # Plot the accuracy and loss curves
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path + "plot_figures/"+n_pasta+"/acc.png")
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + "plot_figures/"+n_pasta+"/loss.png")
    #plt.show()

# Run the function to illustrate accuracy and loss
visualize_results(history, n_pasta)

