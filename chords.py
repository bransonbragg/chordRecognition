import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import gda12
import csv
import util
from gda12 import GDA
import math

def main():
    # Load in audio
    file_dir = os.listdir('audio/jim2012Chords/Guitar_Only')

    for i in range(1, len(file_dir)):
        files = os.listdir('audio/jim2012Chords/Guitar_Only/' + file_dir[i])
        chroma_array = np.ndarray(shape=(len(files), 12))
        for j in range(len(files)):
            file = files[j]
            rb, sr = librosa.load('audio/jim2012Chords/Guitar_Only/' + file_dir[i] + '/' + file)
            audio1_stft = librosa.feature.chroma_stft(y=rb)
            for k in range(len(audio1_stft)):
                key = audio1_stft[k]
                chroma_array[j][k] = np.mean(key)
            # Normalization
            row_sum = np.sum(chroma_array[j])
            for k in range(len(chroma_array[j])):
                chroma_array[j][k] = chroma_array[j][k] / row_sum
        # Append each group of chords to file
        with open('trainDat.csv', 'a') as f:
            for l in range(len(chroma_array)):
                str1 = ','.join(str(e) for e in chroma_array[l])
                f.write("%s,%s\n" % (str1, i))


    # Test
    file_dir = os.listdir('audio/jim2012Chords/Other_Instruments')
    for i in range(1, len(file_dir)):
        files1 = os.listdir('audio/jim2012Chords/Other_Instruments/' + file_dir[i])
        for o in range(len(files1)):
            files = os.listdir('audio/jim2012Chords/Other_Instruments/' + file_dir[i] + '/' + files1[o])
            chroma_array = np.ndarray(shape=(len(files), 12))
            for j in range(len(files)):
                file = files[j]
                rb, sr = librosa.load('audio/jim2012Chords/Other_Instruments/' + file_dir[i] + '/' + files1[o] + '/' + file)
                audio1_stft = librosa.feature.chroma_stft(y=rb)
                for k in range(len(audio1_stft)):
                    key = audio1_stft[k]
                    chroma_array[j][k] = np.mean(key)
                # Normalization
                row_sum = np.sum(chroma_array[j])
                for k in range(len(chroma_array[j])):
                    chroma_array[j][k] = chroma_array[j][k] / row_sum
            # Append each group of chords to file
            with open('testDat.csv', 'a') as f:
                for l in range(len(chroma_array)):
                    str1 = ','.join(str(e) for e in chroma_array[l])
                    f.write("%s,%s\n" % (str1, i))
    

    train_path = 'trainDat.csv'
    test_path = 'testDat.csv'
    """
    clf = GDA()
    phi, sigma, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10 = clf.fit(x_train, y_train)

    predictions = clf.predict(x_train, phi, sigma, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10)
    num_cor = 0
    for i in range(len(predictions)):
        if predictions[i] == y_train[i]:
            num_cor += 1
    print("Train accuracy is: " + str(num_cor / len(predictions)))

    predictions = clf.predict(x_test, phi, sigma, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10)
    num_cor = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            num_cor += 1
    print("Test accuracy is: " + str(num_cor / len(predictions)))
    """

    """
    #with open('trainBran.csv', 'w') as f:
    #    for key in master_dict.keys():
    #        f.write("%s, %s\n" % (key, master_dict[key]))
    print("debug")
    
    Pass the inputs in R^12 as x
    Pass the labels in R as y
    ^^ Format to match prior GDA implementation ^^
    """




    """
    rb, sr = librosa.load('audio/jim2012Chords/Guitar_Only/a/a1.wav')
    rap, _ = librosa.load('audio/jim2012Chords/Guitar_Only/a/a2.wav')
    rock, _ = librosa.load('audio/jim2012Chords/Guitar_Only/a/a3.wav')

    # Create chromagrams
    audio1_stft = librosa.feature.chroma_stft(y=rb)
    for key in audio1_stft:
        print(mean(key))
    print("beal")
    """

    # Look into this later
    # audio1_cqt = librosa.feature.chroma_cqt(y=rb, sr=sb)
    # audio1_sens = librosa.feature.chroma_cens(y=rb, sr=sb)
    # rap_chroma = librosa.feature.chroma_stft(rap, sr=sr)
    # rock_chroma = librosa.feature.chroma_stft(rock, sr=sr)

if __name__ == '__main__':
    main()