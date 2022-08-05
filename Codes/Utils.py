#%% Import modules
import os, tensorflow
import numpy as np

from sklearn import metrics
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


#%% Data managements
def split_XY(data, term):
    if len(data.shape)==2:
        dataX = data[:,:term]
        dataY = data[:,term:]
    elif len(data.shape)==3:
        dataX = data[:,:term,:]
        dataY = data[:,term:,:]
    else:
        print('data shape is not 2 or 3')
    return dataX, dataY

def categorical(trainY, valY, testY):
    return to_categorical(trainY), to_categorical(valY), to_categorical(testY), to_categorical(testY).shape[-1]

def get_lookup_list(steps_per_pat):
    lookup_list = []
    for idx, elements in enumerate(steps_per_pat):
        for i in range(elements):
            lookup_list.append(idx)
    return lookup_list

def get_PAT_CA_list(pat_data, lookup_list, lookup_pd):
    PAT_CA_list = []
    for pat_idx in lookup_list:
        temp_pat_num = pat_data[pat_idx]
        temp_pat_series = lookup_pd[lookup_pd['환자번호']==temp_pat_num]
        temp_pat_CA  = temp_pat_series['두번째 분류'].values
        whole_period = np.timedelta64((temp_pat_series['퇴원일시'] - temp_pat_series['입원일시']).values[0], 'h')
        frequency = temp_pat_series['측정횟수'].values[0]/whole_period.astype(int)
        PAT_CA_list.append([temp_pat_num, temp_pat_CA[0], whole_period, frequency])
    return PAT_CA_list

def get_morethan0_with_idx(data, num_morethan0):
    morethan0 = []
    morethan0_idx = []
    """9.csv 기준 222 224 229 287 1790 3498 3686 4827 행은 0점을 넘는 NRS 점수가 없다. row 자체를 날린다."""
    for i in  range(len(data)):
        pat = data.loc[i]
        pat_num = int(pat.iloc[0])
        num_NRS = pat.iloc[1]
        record = pat.iloc[2:].dropna()
        record_idx = np.where(record>0)
        # for idx in record_idx:
        morethan0.append(len(record_idx[0]))
        if len(record_idx[0])>num_morethan0 :
            morethan0_idx.append(i)
    return morethan0, morethan0_idx

def get_array_list(data_pd, lookup_pd, hours, type): # Get data array list from pandas
    data_list = []
    pat_list  = []
    start_list=[]
    if type != 'ALL':
        for i in  range(len(data_pd)):
            pat = data_pd.loc[i]
            pat_num = int(pat.iloc[0])
            pat_start = lookup_pd[lookup_pd['환자번호'] == pat_num]['\t1\t.시행일'].dt.hour.values[0]
            record = pat.iloc[2:].dropna()
            if len(record.values) > hours and lookup_pd[lookup_pd['환자번호']==pat_num]['첫번째 분류'].values[0]==type:
                data_list.append(record.values)
                pat_list.append(pat_num)
                start_list.append(pat_start)
    else :
        for i in  range(len(data_pd)):
            pat = data_pd.loc[i]
            pat_num = int(pat.iloc[0])
            pat_start = lookup_pd[lookup_pd['환자번호'] == pat_num]['\t1\t.시행일'].dt.hour.values[0]
            record = pat.iloc[2:].dropna()
            if len(record.values) > hours:
                data_list.append(record.values)
                pat_list.append(pat_num)
                start_list.append(pat_start)
    return data_list, pat_list, start_list

def get_array_list2(data_pd, lookup_pd, hours, type, ratio=1/4): # Get data array list from pandas
    data_list = []
    pat_list  = []
    start_list=[]
    if type != 'ALL':
        for i in  range(len(data_pd)):
            pat = data_pd.loc[i]
            pat_num = int(pat.iloc[0])
            pat_start = lookup_pd[lookup_pd['환자번호'] == pat_num]['\t1\t.시행일'].dt.hour.values[0]
            record = pat.iloc[2:].dropna()
            if len(record.values) > hours and lookup_pd[lookup_pd['환자번호']==pat_num]['첫번째 분류'].values[0]==type and ((record!=0).sum()/len(record)>ratio):
                data_list.append(record.values)
                pat_list.append(pat_num)
                start_list.append(pat_start)
    else :
        for i in  range(len(data_pd)):
            pat = data_pd.loc[i]
            pat_num = int(pat.iloc[0])
            pat_start = lookup_pd[lookup_pd['환자번호'] == pat_num]['\t1\t.시행일'].dt.hour.values[0]
            record = pat.iloc[2:].dropna()
            if len(record.values) > hours and ((record!=0).sum()/len(record)>ratio):
                data_list.append(record.values)
                pat_list.append(pat_num)
                start_list.append(pat_start)
    return data_list, pat_list, start_list

def get_records_by_hours(data_list, hours, start_list, term):
    records_hours_list = []
    steps_per_pat = []
    for idx, temp_data in enumerate(data_list):
        step_pat = 0
        if len(temp_data) <= hours: # for check
            print(idx)
        else:
            temp_data = np.pad(temp_data, (start_list[idx],0), mode='constant', constant_values=0) # 입원시간 기준 앞 시간대에 대한 zero 패딩
            steps = (len(temp_data)-hours+term) // term
            for step in range(steps):
                temp = temp_data[step * term: step * term + hours]
                if temp.sum() !=0:
                    records_hours_list.append(temp)
                    step_pat +=1
            if step_pat != 0:
                steps_per_pat.append(step_pat)
    return records_hours_list, steps_per_pat


#%% Preprocessings
def make_binary_score(data):
    new_data = np.copy(data)
    new_data[new_data<4]=0
    new_data[new_data>0]=1
    return new_data

def make_binary_score_for_all(trainY, valY, testY):
    return(make_binary_score(trainY), make_binary_score(valY), make_binary_score(testY))

def compute_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

def cvt_categorical(data):
    int_data = data.astype(np.int8)
    reshaped_data = tensorflow.keras.utils.to_categorical(int_data, np.max(int_data)+1)
    return reshaped_data

def cvt_shape(data, resolution):
    reshaped_data = np.reshape(data, (data.shape[0],data.shape[1]//(24//resolution),-1))
    return np.transpose(reshaped_data,(0,2,1))

def transformation(trainX, valX, testX, resolution=1):
    return(cvt_shape(trainX, resolution), cvt_shape(valX, resolution), cvt_shape(testX, resolution))

def time_binning(data, resolution):
    data_ = np.squeeze(data)
    data = data_.reshape((data_.shape[0], data_.shape[1] // resolution, resolution))
    data = np.max(data, axis=-1)
    data = np.expand_dims(data, axis=-1)
    return data

def cvt_value(data):
    cvt_data = np.copy(data)
    cvt_data[cvt_data<4]=0
    cvt_data[cvt_data>1]=1
    return cvt_data

def cvt_all_outputs(trainY, valY, testY):
    return(cvt_value(trainY), cvt_value(valY), cvt_value(testY))

def normalization(trainX, valX, testX):
    return trainX/10, valX/10, testX/10

def standardization(trainX, valX, testX, std_mode = 'Patwise-mean'):
    if std_mode == 'Patwise-mean':
        shape = trainX.shape
        tr = np.reshape(trainX, (shape[0], shape[1]//24, -1))
        tr_mean = np.expand_dims(np.mean(tr, axis=1), 2)
        tr_mean_reshape = np.reshape(np.swapaxes(np.repeat(tr_mean, shape[1]//24, axis=2), 1, 2), shape)
        trainX = trainX - tr_mean_reshape

        shape = valX.shape
        val = np.reshape(valX, (shape[0], shape[1]//24, -1))
        val_mean = np.expand_dims(np.mean(val, axis=1), 2)
        val_mean_reshape = np.reshape(np.swapaxes(np.repeat(val_mean, shape[1]//24, axis=2), 1, 2), shape)
        valX = valX - val_mean_reshape

        shape = testX.shape
        test = np.reshape(testX, (shape[0], shape[1]//24, -1))
        test_mean = np.expand_dims(np.mean(test, axis=1), 2)
        test_mean_reshape = np.reshape(np.swapaxes(np.repeat(test_mean, shape[1]//24, axis=2), 1, 2), shape)
        testX = testX - test_mean_reshape
        return trainX, valX, testX, tr_mean, val_mean, test_mean

    elif std_mode == 'All-mean':
        shape = trainX.shape
        tr = np.reshape(trainX, (shape[0], shape[1]//24, -1))
        tr_mean = np.expand_dims(np.mean(np.mean(tr, axis=1), axis=0),1)
        trainX = trainX - np.repeat(tr_mean, shape[1]//24, axis=0)

        shape = valX.shape
        val = np.reshape(valX, (shape[0], shape[1]//24, -1))
        val_mean = np.expand_dims(np.mean(np.mean(val, axis=1), axis=0),1)
        valX = valX - np.repeat(val_mean, shape[1]//24, axis=0)

        shape = testX.shape
        test = np.reshape(testX, (shape[0], shape[1]//24, -1))
        test_mean = np.expand_dims(np.mean(np.mean(test, axis=1), axis=0),1)
        testX = testX - np.repeat(test_mean, shape[1]//24, axis=0)
        return trainX, valX, testX, tr_mean, val_mean, test_mean

    else :
        return trainX, valX, testX, 0, 0, 0

def standardization_with_mean(trainY, valY, testY, tr_mean, val_mean, test_mean):
    return trainY-tr_mean, valY-val_mean, testY-test_mean

def value_modification(x, y, input_value_range, output_value_range):
    X = np.copy(x)
    Y = np.copy(y)
    if input_value_range == '[0,10]':
        pass
    elif input_value_range == '[0,1]':
        X[X<4]=0
        X[X>1]=1
    elif input_value_range == '[0,2]':
        zero_idx = X==0
        X[X<4]=1
        X[X>=4]=2
        X[zero_idx]=0
    else:
        print('!!!!!!!!!!! Wrong input value range !!!!!!!!!!!')
    if output_value_range == '[0,10]':
        pass
    elif output_value_range == '[0,1]':
        Y[Y<4]=0
        Y[Y>1]=1
    elif output_value_range == '[0,2]':
        zero_idx = Y==0
        Y[Y<4]=1
        Y[Y>=4]=2
        Y[zero_idx]=0
    else:
        print('!!!!!!!!!!! Wrong input value range !!!!!!!!!!!')
    return X, Y

def value_modification_for_all(x, input_value_range, output_value_range):
    newX = []
    for X in x:
        if input_value_range == '[0,10]':
            newX.append(X)
            pass
        elif input_value_range == '[0,1]':
            X[X<4]=0
            X[X>1]=1
            newX.append(X)
        elif input_value_range == '[0,2]':
            zero_idx = X==0
            X[X<4]=1
            X[X>=4]=2
            X[zero_idx]=0
            newX.append(X)
        else:
            print('!!!!!!!!!!! Wrong input value range !!!!!!!!!!!')
    return newX


#%% Metrics
def NRMSE():
    def nrmse(y_true, y_pred):
        rmse = K.sqrt(K.mean(K.square(y_true-y_pred), axis=1))
        means= K.mean(y_true, axis=1)
        return K.sum(rmse/(means + K.epsilon()))
    return nrmse

def Recall():
    def recall(y_true, y_pred):
        # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
        # round : 반올림한다
        y_true_yn = K.round(K.clip(y_true, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
        y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_true_yn * y_pred_yn)
        # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
        count_true_positive_false_negative = K.sum(y_true_yn)
        # Recall =  (True Positive) / (True Positive + False Negative)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
        # return a single tensor value
        return recall
    return recall

def R2score():
    def r2score(y_true, y_pred):
        SS_res = K.sum(K.square(y_true-y_pred))
        SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
        return (SS_res/(SS_tot + K.epsilon()))
    return r2score

def evaluation(testY, pred):
    evs = metrics.explained_variance_score(testY, pred)
    mae = metrics.mean_absolute_error(testY, pred)
    mse = metrics.mean_squared_error(testY, pred)
    median_ae = metrics.median_absolute_error(testY, pred)
    r2 = metrics.r2_score(testY, pred)
    f1 = metrics.f1_score(testY.reshape(-1), pred.reshape(-1))
    acc = metrics.accuracy_score(testY.reshape(-1), pred.reshape(-1))
    balanced_acc = metrics.balanced_accuracy_score(testY.reshape(-1), pred.reshape(-1))
    mcc = metrics.matthews_corrcoef(testY.reshape(-1), pred.reshape(-1))
    return evs, mae, mse, median_ae, r2, f1, acc, balanced_acc, mcc

def evaluation_essential(testY, pred):
    f1 = metrics.f1_score(testY.reshape(-1), pred.reshape(-1))
    acc = metrics.accuracy_score(testY.reshape(-1), pred.reshape(-1))
    balanced_acc = metrics.balanced_accuracy_score(testY.reshape(-1), pred.reshape(-1))
    mcc = metrics.matthews_corrcoef(testY.reshape(-1), pred.reshape(-1))
    return f1, acc, balanced_acc, mcc

#%% Visualization
def plot(hist, savedir):
    train_loss = hist['loss']
    val_loss = hist['val_loss']

    plt.figure()
    epochs = np.arange(1, len(train_loss) + 1, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(savedir) # <- When we run this before 'savefig', blank image files are generated.

def plot_comparison(testY, pred, PAT_CA_list, save_dir):
    plt.figure()#figsize=(30,10))
    plt.plot(testY, label="actual")
    plt.plot(pred, label="prediction")
    plt.legend()#fontsize=10)
    plt.grid(axis="both")
    plt.title("Pain prediction - Pat:{}, CA:{}".format(PAT_CA_list[0], PAT_CA_list[1]))#,fontsize=25)
    plt.savefig(save_dir)

def draw_histogram(morethan0, data_dir):
    unique_morethan0 = np.unique(morethan0)
    morethan0 = np.asarray(morethan0)
    count_morethan0 = {}
    for unique_num in unique_morethan0:
        count_morethan0[unique_num] = len(np.where(morethan0 == unique_num)[0])
    # {0: 8, 1: 4, 2: 2, 3: 2, 5: 1, 6: 2, 7: 3, 8: 3, 9: 5, 10: 5, 11: 6, 12: 9, 13: 2, 14: 6, 15: 5, 16: 17, 17: 28, 18: 63, 19: 104, 20: 229, 21: 191, 22: 186, 23: 158, 24: 153, 25: 154, 26: 149, 27: 135, 28: 128, 29: 116, 30: 112, 31: 109, 32: 101, 33: 96, 34: 102, 35: 99, 36: 94, 37: 81, 38: 72, 39: 80, 40: 65, 41: 74, 42: 49, 43: 58, 44: 55, 45: 60, 46: 61, 47: 56, 48: 52, 49: 63, 50: 60, 51: 48, 52: 56, 53: 39, 54: 39, 55: 23, 56: 43, 57: 39, 58: 33, 59: 30, 60: 29, 61: 25, 62: 26, 63: 29, 64: 31, 65: 22, 66: 21, 67: 26, 68: 16, 69: 30, 70: 25, 71: 25, 72: 23, 73: 21, 74: 18, 75: 19, 76: 11, 77: 16, 78: 16, 79: 13, 80: 22, 81: 10, 82: 9, 83: 13, 84: 13, 85: 13, 86: 16, 87: 14, 88: 14, 89: 10, 90: 14, 91: 13, 92: 13, 93: 12, 94: 12, 95: 7, 96: 6, 97: 8, 98: 8, 99: 15, 100: 8, 101: 11, 102: 9, 103: 7, 104: 8, 105: 4, 106: 6, 107: 9, 108: 5, 109: 5, 110: 7, 111: 6, 112: 10, 113: 8, 114: 6, 115: 9, 116: 7, 117: 5, 118: 10, 119: 5, 120: 4, 121: 1, 122: 3, 123: 1, 124: 1, 125: 6, 126: 6, 12...}
    plt.bar(list(count_morethan0.keys()), count_morethan0.values(), color='b')
    plt.savefig(os.path.join(data_dir, 'NRS_morethan0_hist.png'))

def AUC_ROC(testY, pred_prob, temp_full_name, save_dir):
    plt.clf()
    plt.figure(figsize=(4.5, 4.5))
    fpr, tpr, thresh = metrics.roc_curve(testY.reshape(-1), pred_prob.reshape(-1))
    auc = metrics.roc_auc_score(testY.reshape(-1), pred_prob.reshape(-1))
    plt.plot(fpr, tpr, lw=2, label="%s = %0.4f" % ('Ours', auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, temp_full_name+'_AUC-ROC.png'))
    plt.clf()
    return auc

def AUC_PR(testY, pred_prob, pred, temp_full_name, save_dir):
    plt.clf()
    plt.figure(figsize=(4.5, 4.5))
    lr_precision, lr_recall, _ = metrics.precision_recall_curve(testY.reshape(-1), pred_prob.reshape(-1))
    lr_f1 = metrics.f1_score(testY.reshape(-1), pred.reshape(-1))
    lr_auc = metrics.auc(lr_recall, lr_precision)
    plt.plot(lr_recall, lr_precision, lw=2, label="%s = %0.4f" % ('Ours', lr_auc))
    no_skill = testY.sum()/np.prod(list(testY.shape))
    plt.plot([1, 0], [no_skill, no_skill], linestyle='--', color='k', label="%s = %0.4f" % ('Random', no_skill))
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc="lower right")
    plt.rc('font', size=10)
    plt.savefig(os.path.join(save_dir, temp_full_name+'_AUC-PR.png'))
    plt.clf()
    return lr_auc

def modified_AUC_PR(testY, pred_prob, pred, temp_full_name, save_dir):
    one_idx = np.where(testY.reshape(-1) == 1)
    zero_idx = np.where(testY.reshape(-1) == 0)
    sampled_idx = zero_idx[0][:len(one_idx[0])]
    k = np.concatenate([one_idx[0], sampled_idx])
    plt.clf()
    plt.figure(figsize=(4.5, 4.5))
    lr_precision, lr_recall, _ = metrics.precision_recall_curve(testY.reshape(-1)[k], pred_prob.reshape(-1)[k])
    lr_f1 = metrics.f1_score(testY.reshape(-1)[k], pred.reshape(-1)[k])
    lr_auc = metrics.auc(lr_recall, lr_precision)
    plt.plot(lr_recall, lr_precision, label="%s = %0.4f" % ('Ours', lr_auc))
    plt.plot([1, 0], [0, 1], linestyle='--', color='k', label='Random')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, temp_full_name+'_modified PR-AUC.png'))
    plt.clf()
    return lr_auc

def scatter_plot_comparison(testY, pred, PAT_CA_list, save_dir):
    plt.figure()#figsize=(30,10))
    # axes = plt.axes()
    # axes.set_ylim([-0.5, 10.5])
    # axes.set_yticks(np.arange(11))
    plt.scatter(np.arange(len(testY)), testY, label="actual", edgecolors='k', c='#2ca02c', s=64)
    plt.plot(testY, c='#2ca02c')
    plt.scatter(np.arange(len(pred)),pred, label="prediction", edgecolors='k', c='#ff7f0e', s=64)
    plt.plot(pred, c='#ff7f0e')
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(axis="both")
    plt.title("Pain prediction - Pat:{}, CA:{}".format(PAT_CA_list[0], PAT_CA_list[1]),fontsize=15)
    plt.rc('font', size=15)
    plt.savefig(save_dir)

def scatter_plot_comparison_with_inputs(testX, testY, pred, input_length, save_dir):
    testX = np.squeeze(testX)
    testY = np.squeeze(testY)
    test_sequence = np.concatenate([testX, testY])
    plt.figure(figsize=(15,4))
    plt.plot(np.arange(test_sequence.shape[-1])[:input_length+1], np.append(testX, testY[0]), label='Inputs', marker='.')
    plt.scatter(np.arange(test_sequence.shape[-1])[input_length:], testY, label="actual", edgecolors='k', c='#2ca02c', s=32)
    plt.plot(np.arange(test_sequence.shape[-1])[input_length:], testY, c='#2ca02c')
    plt.scatter(np.arange(test_sequence.shape[-1])[input_length:],pred, label="prediction", edgecolors='k', c='#ff7f0e', s=32)
    plt.plot(np.arange(test_sequence.shape[-1])[input_length:], pred, c='#ff7f0e')
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(axis="both")
    plt.title("BTcP prediction with inputs",fontsize=15)
    plt.rc('font', size=15)
    plt.savefig(save_dir)


#%% Dirs
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise




