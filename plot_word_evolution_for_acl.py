import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

beta = scipy.io.loadmat('./results/detm_acl_K_10_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.0001_Bsz_100_RhoSize_300_L_3_minDF_2_trainEmbeddings_1_beta.mat')['values'] ## K x T x V
print('beta: ', beta.shape)

with open('data_acl_largev/min_df_2/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

## get vocab
data_file = 'data_acl_largev/min_df_2'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics 
num_words = 10
times = [0, 10, 30]
num_topics = 10
for k in range(num_topics):
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])
        topic_words = [vocab[a] for a in top_words]
        print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

print('Topic Climate Change...')
num_words = 10
for t in range(31):
    gamma = beta[2, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Time: {} ===> {}'.format(t, topic_words)) 

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
ticks = [str(x) for x in timelist]
#plt.xticks(np.arange(T)[0::10], timelist[0::10])

#words_1 = ['article', 'baseline', 'function', 'artificial', 'machine']
words_1 = ['superterm', 'uncommonly', 'approxim', 'doctorale', 'chairmen']
tokens_1 = [vocab.index(w) for w in words_1 if w in vocab]
#print(tokens_1)
betas_1 = [beta[1, :, x] for x in tokens_1 if x < beta.shape[2]]
#print(betas_1)
for i, comp in enumerate(betas_1):
    ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax1.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax1.set_xticks(np.arange(T)[0::10])
ax1.set_xticklabels(timelist[0::10])
ax1.set_title('Topic One', fontsize=12)

#words_2 = ['korean', 'japan', 'thought', 'generator', 'ambiguity']
words_2 = ['security', 'nyu', 'operationnelle', 'transportability', 'fbis']
words_2 = ['feedback', 'phrase', 'purchase', 'knowledge', 'symboles']
tokens_2 = [vocab.index(w) for w in words_2 if w in vocab]
#print(tokens_1)
betas_2 = [beta[2, :, x] for x in tokens_2 if x < beta.shape[2]]
#print(betas_1)
for i, comp in enumerate(betas_2):
    ax2.plot(range(T), comp, label=words_2[i], lw=2, linestyle='--', marker='o', markersize=4)
ax2.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax2.set_xticks(np.arange(T)[0::10])
ax2.set_xticklabels(timelist[0::10])
ax2.set_title('Topic Two', fontsize=12)

#words_3 = ['matlab', 'pickle', 'occupons', 'superterm', 'approxim']
words_3 = ['management', 'users', 'capabilities', 'asked', 'response']
words_3 = ['translation', 'evaluation', 'dicussion', 'errors', 'weida']
tokens_3 = [vocab.index(w) for w in words_3 if w in vocab]
#print(tokens_1)
betas_3 = [beta[3, :, x] for x in tokens_3 if x < beta.shape[2]]
#print(betas_1)
for i, comp in enumerate(betas_3):
    ax3.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax3.set_xticks(np.arange(T)[0::10])
ax3.set_xticklabels(timelist[0::10])
ax3.set_title('Topic Third', fontsize=12)

#words_4 = ['matlab', 'pickle', 'occupons', 'superterm', 'approxim']
words_4 = ['debt', 'inalf', 'miscarriage', 'saver', 'shavlik']
words_4 = ['personl', 'action', 'coordinate', 'knowledge', 'operators']
tokens_4 = [vocab.index(w) for w in words_4 if w in vocab]
#print(tokens_1)
betas_4 = [beta[4, :, x] for x in tokens_4 if x < beta.shape[2]]
#print(betas_1)
for i, comp in enumerate(betas_4):
    ax4.plot(range(T), comp, label=words_4[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax4.set_xticks(np.arange(T)[0::10])
ax4.set_xticklabels(timelist[0::10])
ax4.set_title('Topic Four', fontsize=12)

#words_5 = ['database', 'education', 'hunger', 'programme', 'water']
words_5 = ['rights', 'vegas', 'chopped', 'dismay', 'logl']
words_5 = ['database', 'education', 'hunger', 'programme', 'water']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[5, :, x] for x in tokens_5 if x < beta.shape[2]]
for i, comp in enumerate(betas_5):
    ax5.plot(comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)
ax5.legend(frameon=False)
ax5.set_xticks(np.arange(T)[0::10])
ax5.set_xticklabels(timelist[0::10])
ax5.set_title('Topic Five', fontsize=12)


words_6 = ['africa', 'summ', 'overthrow', 'endangered', 'namibia']
words_6 = ['switching', 'methode', 'selector', 'mentalism', 'frquent']
tokens_6 = [vocab.index(w) for w in words_6]
betas_6 = [beta[0, :, x] for x in tokens_6 if x < beta.shape[2]]
for i, comp in enumerate(betas_6):
    ax6.plot(comp, label=words_6[i], lw=2, linestyle='--', marker='o', markersize=4)
ax6.legend(frameon=False)
ax6.set_xticks(np.arange(T)[0::10])
ax6.set_xticklabels(timelist[0::10])
ax6.set_title('Topic Six', fontsize=12)

plt.savefig('word_evolution_acl.png')
plt.show()
