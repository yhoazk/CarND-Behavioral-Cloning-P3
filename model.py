#!/usr/bin/env python


from augment import *
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import gc
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.visualize_util import plot
#define constants


#SIMULATOR_DIR = "/home/porko/Descargas/simulator-linux/"
SIMULATOR_DIR ="/home/porko/Descargas/simulator-linux/"
#SIMULATOR_DIR = "../simulator-linux/"
IMG_DIR       = SIMULATOR_DIR# + "IMG/"
LABEL_CSV     = SIMULATOR_DIR + "driving_log.csv"

PARK_REC_IMG  = "/home/porko/Descargas/simulator-linux/prk/"#SIMULATOR_DIR + "prk/"#IMG/"
PARK_REC_CSV  = PARK_REC_IMG + "/driving_log.csv"

LEFT_REC_IMG  = "/home/porko/Descargas/simulator-linux/lft/"#SIMULATOR_DIR + "lft/"#IMG/"
LEFT_REC_CSV  = LEFT_REC_IMG + "/driving_log.csv"

RIGHT_REC_IMG  = "/home/porko/Descargas/simulator-linux/" + "rgt/"#IMG/"
RIGHT_REC_CSV  = RIGHT_REC_IMG + "/driving_log.csv"

BOTTOM_MARGIN = 50
TOP_MARGIN = 140


# Get data sets



X_all,y_all = get_dataset(IMG_DIR, LABEL_CSV )
n_all, bins_all = analyze_dataset(y_all)

X_park,y_park = get_dataset(PARK_REC_IMG, PARK_REC_CSV )
# Make all the samples more to the right
for n,y in enumerate(y_park):
    y_park[n] = (-1.*abs(y)) + (random.randint(1, 10) / -100)

#n_park, bins_park = analyze_dataset(y_park)



X_lft,y_lft = get_dataset(LEFT_REC_IMG, LEFT_REC_CSV )
# Make all the samples more to the right
for n,y in enumerate(y_lft):
    y_lft[n] = (-1.*abs(y)) + (random.randint(1, 10) / -100)

#n_lft, bins_lft = analyze_dataset(y_lft)

X_rgt, y_rgt = get_dataset(RIGHT_REC_IMG, RIGHT_REC_CSV)
# Make all the samples more to the right
for n, y in enumerate(y_rgt):
    y_rgt[n] = abs(y) + (random.randint(1, 10) / 100)

#n_rgt, bins_rgt = analyze_dataset(y_rgt)

# create list for data
X = [[] for _ in range(len(n_all))]
# create list for labels]
y = [[] for _ in range(len(n_all))]
random.seed(1010)
# Transformtations to augment the data set
X_all = X_lft + X_park + 2 * X_rgt + X_all
y_all = y_lft + y_park + 2 * y_rgt + y_all
for i, (lbl, img) in enumerate(zip(y_all, X_all)):
    if -0.1 <= lbl <= 0.038 and random.random() >= 0.4:
        y_all.pop(i)
        X_all.pop(i)

        # Separate the images in bins for augmentation
for label, abs_path in zip(y_all, X_all):
    # Increase the data set.
    for b in range(len(n_all) - 1):
        if bins_all[b] < label <= bins_all[b + 1]:
            X[b].append(preprocessrgb2gray(abs_path, (0, BOTTOM_MARGIN, 320, TOP_MARGIN)))
            y[b].append(label)

for x in range(len(n_all)):
    print(len(X[x]))

# remove 40% of images with steering close to zero
for i, (lbl, img) in enumerate(zip(y[10], X[10])):
    if random.random() >= 0.2:
        y[10].pop(i)
        X[10].pop(i)

        # Add more data by augmentation in each bin, first try to balance
X[8], y[8] = augmentBin(X[8], y[8], 0.5)

X[11], y[11] = augmentBin(X[11], y[11], 0.5)

X[13], y[13] = augmentBin(X[13], y[13], 0.5)
X[7], y[7] = augmentBin(X[7], y[7], 0.5)
X[14], y[14] = augmentBin(X[14], y[14], 0.5)
X[6], y[6] = augmentBin(X[6], y[6], 0.5)

X[15], y[15] = augmentBin(X[15], y[15], 0.5)
X[5], y[5] = augmentBin(X[5], y[5], 0.5)
X[4], y[4] = augmentBin(X[4], y[4], 0.5)

X[17], y[17] = augmentBin(X[17], y[17], 0.5)
X[18], y[18] = augmentBin(X[18], y[18], 0.5)

# flat list of lists

X = sum(X, [])
y = sum(y, [])
gc.collect()
print(len(X))
print(len(y))

### Re print the graph

n, bins, _ = plt.hist(y, facecolor='green', bins=20)
print("New negative images count: " + str(sum(n[0:9])))
print("New postivie images count: " + str(sum(n[11:19])))

plt.xlabel('Steering angle')
plt.ylabel('Number of images in the range')
plt.grid(True)
plt.show()


X = np.asarray(X)
X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))
y = np.asarray(y).astype('float32')
y = y.reshape((y.shape[0],1))
##
X,y = shuffle(X,y,random_state=1)
## Split the dataset to get the validation data, the test will be with the model.
X, X_val, y, y_val = train_test_split(X, y, random_state=0, test_size=0.2)



# Instantiate a Sequential model
model = Sequential()
# Add first convolution layer

#

model.add(Convolution2D(1,1,1, border_mode='same', input_shape=(X.shape[1],X.shape[2],X.shape[3]), dim_ordering='tf'))
# shape ()
model.add(Convolution2D(24, 8, 8, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2),(2,2),'valid'))
model.add(Dropout(0.2))

model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2),(2,2),'valid'))
model.add(Dropout(0.2))

model.add(Convolution2D(48, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2),(2,2),'valid'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2),(2,2),'valid'))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))
model.add(Dense(1))
model.summary()

plot(model, to_file="model.png")

adam_opt = Adam(lr=0.0001 )
model.compile(loss='mean_squared_error',optimizer=adam_opt)
checkpointer = ModelCheckpoint(filepath="./weights-{epoch:02d}.h5", verbose=1, save_best_only=False)
log = model.fit(X, y,batch_size=40, nb_epoch=25, verbose=1, validation_data=(X_val, y_val), callbacks=[checkpointer])

open("model.json", "w").write(model.to_json())
model.save_weights("model.h5")
gc.collect()
K.clear_session()

print("--- Done ---")
