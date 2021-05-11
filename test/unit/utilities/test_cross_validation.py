from sklearn.model_selection import KFold

def test():
    array = list('qwerasdfzxcvghkj')
    kf = KFold(n_splits=3)
    for train, test in kf.split(array):

        for i in train:
            print(array[i], end='')
        print()
        for j in test:
            print(array[j], end='')
        print()



test()
