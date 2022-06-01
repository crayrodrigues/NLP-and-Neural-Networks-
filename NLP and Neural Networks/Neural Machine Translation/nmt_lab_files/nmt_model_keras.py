from keras.layers import Embedding,LSTM,Dropout,Dense,Layer
from keras import Model,Input
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import collections
import numpy as np
import time
from nltk.translate.bleu_score import corpus_bleu


class LanguageDict():
  def __init__(self, sents):
    word_counter = collections.Counter(tok.lower() for sent in sents for tok in sent)

    self.vocab = []
    self.vocab.append('<pad>') #zero paddings
    self.vocab.append('<unk>')
    # add only words that appear at least 10 times in the corpus
    self.vocab.extend([t for t,c in word_counter.items() if c > 10])

    self.word2ids = {w:id for id, w in enumerate(self.vocab)}
    self.UNK = self.word2ids['<unk>']
    self.PAD = self.word2ids['<pad>']



def load_dataset(source_path,target_path, max_num_examples=30000):
  ''' This helper method reads from the source and target files to load max_num_examples 
  sentences, split them into train, development and testing and return relevant data.
  Inputs:
    source_path (string): the full path to the source data, SOURCE_PATH
    target_path (string): the full path to the target data, TARGET_PATH
  Returns:
    train_data (list): a list of 3 elements: source_words, target words, target word labels
    dev_data (list): a list of 2 elements - source words, target word labels
    test_data (list): a list of 2 elements - source words, target word labels
    source_dict (LanguageDict): a LanguageDict object for the source language, Vietnamese.
    target_dict (LanguageDict): a LanguageDict object for the target language, English.
  ''' 
  # source_lines/target lines are list of strings such that each string is a sentence in the
  # corresponding file. len(source/target_lines) <= max_num_examples
  source_lines = open(source_path).readlines()
  target_lines = open(target_path).readlines()
  assert len(source_lines) == len(target_lines)
  if max_num_examples > 0:
    max_num_examples = min(len(source_lines), max_num_examples)
    source_lines = source_lines[:max_num_examples]
    target_lines = target_lines[:max_num_examples]

  # strip trailing/leading whitespaces and tokenize each sentence. 
  source_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in source_lines]
  target_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in target_lines]
    # for the target sentences, add <start> and <end> tokens to each sentence 
  for sent in target_sents:
    sent.append('<end>')
    sent.insert(0,'<start>')

  # create the LanguageDict objects for each file
  source_lang_dict = LanguageDict(source_sents)
  target_lang_dict = LanguageDict(target_sents)


  # for the source sentences.
  # we'll use this to split into train/dev/test 
  unit = len(source_sents)//10
  # get the sents-as-ids for each sentence
  source_words = [[source_lang_dict.word2ids.get(tok,source_lang_dict.UNK) for tok in sent] for sent in source_sents]
  # 8 parts (80%) of the sentences go to the training data. pad upto maximum sentence length
  source_words_train = pad_sequences(source_words[:8*unit],padding='post')
  # 1 parts (10%) of the sentences go to the dev data. pad upto maximum sentence length
  source_words_dev = pad_sequences(source_words[8*unit:9*unit],padding='post')
  # 1 parts (10%) of the sentences go to the test data. pad upto maximum sentence length
  source_words_test = pad_sequences(source_words[9*unit:],padding='post')


  eos = target_lang_dict.word2ids['<end>']
  # for each sentence, get the word index for the tokens from <start> to up to but not including <end>,
  target_words = [[target_lang_dict.word2ids.get(tok,target_lang_dict.UNK) for tok in sent[:-1]] for sent in target_sents]
  # select the training set and pad the sentences
  target_words_train = pad_sequences(target_words[:8*unit],padding='post')
  # the label for each target word is the next word after it
  target_words_train_labels = [sent[1:]+[eos] for sent in target_words[:8*unit]]
  # pad the labels. Dim = [num_sents, max_sent_lenght]
  target_words_train_labels = pad_sequences(target_words_train_labels,padding='post')
  # expand dimensions Dim = [num_sents, max_sent_lenght, 1]. 
  target_words_train_labels = np.expand_dims(target_words_train_labels,axis=2)

  # get the labels for the dev and test data. No need for inputs here. no need to expand dimensions
  target_words_dev_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[8 * unit:9 * unit]], padding='post')
  target_words_test_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[9 * unit:]], padding='post')

  # we have our data.
  train_data = [source_words_train,target_words_train,target_words_train_labels]
  dev_data = [source_words_dev,target_words_dev_labels]
  test_data = [source_words_test,target_words_test_labels]

  return train_data,dev_data,test_data,source_lang_dict,target_lang_dict





class AttentionLayer(Layer):
  def compute_mask(self, inputs, mask=None):
    if mask == None:
      return None
    return mask[1]

  def compute_output_shape(self, input_shape):
    return (input_shape[1][0],input_shape[1][1],input_shape[1][2]*2)


  def call(self, inputs, mask=None):
    encoder_outputs, decoder_outputs = inputs

    """
    Task 3 attention
    
    Start
    """
    #Change shape: from [batch_size,max_target_sent_len, hidden_size] to [batch_size, hidden_size, max_target_sent_len]
    decoder_outputs_1 = K.permute_dimensions(decoder_outputs, (0, 2, 1))
    #Atention scores: performing dot product and then apply softmax.
    luong_score = K.batch_dot(encoder_outputs, decoder_outputs_1)
    luong_score = K.softmax(luong_score, axis=1)
    # Expand the last dimension of luong_score [batch_size, max_source_sent_len, max_target_sent_len, 1].
    luong_score = K.expand_dims(luong_score, axis=-1)
    # Expand dimension encoder_ouputs: [batch_size, max_source_sent_len, 1, hidden_size]
    encoder_outputs = K.expand_dims(encoder_outputs, axis=2)
    #create encoder_vector by doing element-wise multiplication between the encoder_outputs and their attention scores
    #and summing max_source_sent_len dimension.
    encoder_vector = luong_score * encoder_outputs
    encoder_vector = K.sum(encoder_vector, axis=1)
                               
    """
    End Task 3
    """
    # [batch,max_dec,2*emb]
    new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])

    return new_decoder_outputs




class NmtModel(object):
  def __init__(self,source_dict,target_dict,use_attention):
    ''' The model initialization function initializes network parameters.
    Inputs:
      source_dict (LanguageDict): a LanguageDict object for the source language, Vietnamese.
      target_dict (LanguageDict): a LanguageDict object for the target language, English.
      use_attention (bool): if True, use attention.
    Returns:
      None.
    '''
    # the number of hidden units used by the LSTM
    self.hidden_size = 200
    # the size of the word embeddings being used
    self.embedding_size = 100
    # the dropout rate for the hidden layers
    self.hidden_dropout_rate=0.2
    # the dropout rate for the word embeddings
    self.embedding_dropout_rate = 0.2
    # batch size
    self.batch_size = 100

    # the maximum length of the target sentences
    self.max_target_step = 30

    # vocab size for source and target; we'll use everything we receive
    self.vocab_target_size = len(target_dict.vocab)
    self.vocab_source_size = len(source_dict.vocab)

    # intances of the dictionaries
    self.target_dict = target_dict
    self.source_dict = source_dict

    # special tokens to indicate sentence starts and ends.
    self.SOS = target_dict.word2ids['<start>']
    self.EOS = target_dict.word2ids['<end>']

    # use attention or no
    self.use_attention = use_attention

    print("number of tokens in source: %d, number of tokens in target:%d" % (self.vocab_source_size,self.vocab_target_size))



  def build(self):
    #-------------------------Train Models------------------------------
    source_words = Input(shape=(None,),dtype='int32')
    target_words = Input(shape=(None,), dtype='int32')

    """
    Task 1 encoder
    
    Start
    """
    # The train encoder
    # (a.) Create two randomly initialized embedding lookups, one for the source, another for the target. 
    print('Task 1(a): Creating the embedding lookups...')
    embeddings_source = Embedding(input_dim=self.vocab_source_size, embeddings_initializer='random_uniform', mask_zero=True, output_dim=self.embedding_size,input_length=source_words.shape[1])
    embeddings_target = Embedding(input_dim=self.vocab_target_size, embeddings_initializer='random_uniform', mask_zero=True, output_dim=self.embedding_size,input_length=target_words.shape[1])
    
    # (b.) Look up the embeddings for source words and for target words. Apply dropout to each encoded input
    print('\nTask 1(b): Looking up source and target words...')
    source_word_embeddings =  Dropout(self.embedding_dropout_rate)(embeddings_source(source_words))
    target_words_embeddings =  Dropout(self.embedding_dropout_rate)(embeddings_target(target_words))

    # (c.) An encoder LSTM() with return sequences set to True
    print('\nTask 1(c): Creating an encoder')
    encoder_outputs, encoder_state_h, encoder_state_c = LSTM(self.hidden_size, return_sequences=True, return_state=True)(source_word_embeddings)
    """
    End Task 1
    """
    encoder_states = [encoder_state_h,encoder_state_c]

    # The train decoder
    decoder_lstm = LSTM(self.hidden_size,recurrent_dropout=self.hidden_dropout_rate,return_sequences=True,return_state=True)
    decoder_outputs_train,_,_ = decoder_lstm(target_words_embeddings,initial_state=encoder_states)

    if self.use_attention:
      decoder_attention = AttentionLayer()
      decoder_outputs_train = decoder_attention([encoder_outputs,decoder_outputs_train])

    decoder_dense = Dense(self.vocab_target_size,activation='softmax')
    decoder_outputs_train = decoder_dense(decoder_outputs_train)

    # compiling the train model.
    adam = Adam(lr=0.01,clipnorm=5.0)
    self.train_model = Model([source_words,target_words], decoder_outputs_train)
    self.train_model.compile(optimizer=adam,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # at this point you can print model summary for the train model
    print('\t\t\t\t\t\t Train Model Summary.')
    self.train_model.summary()



    #-------------------------Inference Models------------------------------
    # The inference encoder 
    self.encoder_model = Model(source_words,[encoder_outputs,encoder_state_h,encoder_state_c])
    # at this point you can print the summary for the encoder model.
    print('\t\t\t\t\t\t Inference Time Encoder Model Summary.')
    self.encoder_model.summary()

    # The decoder model
    # specifying the inputs to the decoder
    decoder_state_input_h = Input(shape=(self.hidden_size,))
    decoder_state_input_c = Input(shape=(self.hidden_size,))
    encoder_outputs_input = Input(shape=(None,self.hidden_size,))

    """
    Task 2 decoder for inference
    
    Start
    """
    # Task 1 (a.) Get the decoded outputs
    print('\n Putting together the decoder states')
    # get the inititial states for the decoder, decoder_states
    # decoder states are the hidden and cell states from the training stage
    decoder_states = [decoder_state_input_h, decoder_state_input_c]
    # use decoder states as input to the decoder lstm to get the decoder outputs, h, and c for test time inference
    decoder_outputs_test,decoder_state_output_h, decoder_state_output_c = decoder_lstm(target_words_embeddings,initial_state=decoder_states)


    # Task 1 (b.) Add attention if attention
    if self.use_attention:
      decoder_attention = AttentionLayer()
      decoder_outputs_test = decoder_attention([encoder_outputs_input,decoder_outputs_test])
      


    # Task 1 (c.) pass the decoder_outputs_test (with or without attention) to the decoder dense layer
    decoder_outputs_test = decoder_dense(decoder_outputs_test)
    
    """
    End Task 2 
    """
    # put the model together
    self.decoder_model = Model([target_words,decoder_state_input_h,decoder_state_input_c,encoder_outputs_input],
                               [decoder_outputs_test,decoder_state_output_h,decoder_state_output_c])
    # you can now view the model summary
    print('\t\t\t\t\t\t Decoder Inference Model summary')
    print(self.decoder_model.summary())



  def time_used(self, start_time):
    curr_time = time.time()
    used_time = curr_time-start_time
    m = used_time // 60
    s = used_time - 60 * m
    return "%d m %d s" % (m, s)



  def train(self,train_data,dev_data,test_data, epochs):
    start_time = time.time()
    for epoch in range(epochs):
      print("Starting training epoch {}/{}".format(epoch + 1, epochs))
      epoch_time = time.time()
      source_words_train, target_words_train, target_words_train_labels = train_data

      self.train_model.fit([source_words_train,target_words_train],target_words_train_labels,batch_size=self.batch_size)

      print("Time used for epoch {}: {}".format(epoch + 1, self.time_used(epoch_time)))
      dev_time = time.time()
      print("Evaluating on dev set after epoch {}/{}:".format(epoch + 1, epochs))
      self.eval(dev_data)
      print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

    print("Training finished!")
    print("Time used for training: {}".format(self.time_used(start_time)))

    print("Evaluating on test set:")
    test_time = time.time()
    self.eval(test_data)
    print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))



  def get_target_sentences(self, sents,vocab,reference=False):
    str_sents = []
    num_sent, max_len = sents.shape
    for i in range(num_sent):
      str_sent = []
      for j in range(max_len):
        t = sents[i,j].item()
        if t == self.SOS:
          continue
        if t == self.EOS:
          break

        str_sent.append(vocab[t])
      if reference:
        str_sents.append([str_sent])
      else:
        str_sents.append(str_sent)
    return str_sents



  def eval(self, dataset):
    # get the source words and target_word_labels for the eval dataset
    source_words, target_words_labels = dataset
    vocab = self.target_dict.vocab

    # using the same encoding network used during training time, encode the training
    encoder_outputs, state_h,state_c = self.encoder_model.predict(source_words,batch_size=self.batch_size)
    # for max_target_step steps, feed the step target words into the decoder.
    predictions = []
    step_target_words = np.ones([source_words.shape[0],1]) * self.SOS
    for _ in range(self.max_target_step):
      
      step_decoder_outputs, state_h,state_c = self.decoder_model.predict([step_target_words,state_h,state_c,encoder_outputs],batch_size=self.batch_size)
      step_target_words = np.argmax(step_decoder_outputs,axis=2)
      predictions.append(step_target_words)

    # predictions is a [time_step x batch_size x 1] array. We use get_target_sentence() to recover the batch_size sentences
    candidates = self.get_target_sentences(np.concatenate(predictions,axis=1),vocab)
    references = self.get_target_sentences(target_words_labels,vocab,reference=True)

    # score using nltk bleu scorer
    score = corpus_bleu(references,candidates)
    print("Model BLEU score: %.2f" % (score*100.0))
    


def main(source_path, target_path, use_attention):
  max_example = 30000
  print('loading dictionaries')
  train_data, dev_data, test_data, source_dict, target_dict = load_dataset(source_path,target_path,max_num_examples=max_example)
  print("read %d/%d/%d train/dev/test batches" % (len(train_data[0]),len(dev_data[0]), len(test_data[0])))

  model = NmtModel(source_dict,target_dict,use_attention)
  model.build()
  model.train(train_data,dev_data,test_data,10)

