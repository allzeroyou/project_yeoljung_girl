import pickle
from keras.utils import np_utils


with open('./test_data/testdata3D.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_test, y_test = datasetL

x_test = x_test.reshape(429, 28, 28).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test)


from keras.models import load_model
model = load_model('deep_model_final2.h5')

result = model.evaluate(x_test, y_test, verbose=2)
print("최종 예측 성공률(%): ", result[1]*100)