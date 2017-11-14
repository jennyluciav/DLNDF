# Udacity Deep Learning Nanodegree Foundations

* [General](#general)
* [Tensorflow](#tensorflow)
* [Neural Networks](#neural-networks)
* [Convolutional Neural Networks](#convolutional-neural-networks)
* [Transfer Learning](#transfer-learning)
* [Recurrent Neural Networks](#recurrent-neural-networks)
* [Generative Adversarial Networks](#generative-adversarial-networks)

## General

### Data Processing

* StratifiedShuffleSplit: Shuffles and splits data into training, validation and test sets ensuring the distribution of classes in each set is identical. (scikit-learn)

### Definitions/Key Terms

* Logits: Values used as an input to the softmax layer. 

### Hyperparameters

* A variable that needs to be set before you can train your algorithm
* There are two categories: 
	* Optimizer hyperparameters: learning rate, mini-batch size, training iterations/epochs etc.
	* Model hyperparameters: # of layers and hidden units etc.
* Learning Rate
	* Good starting point is 0.01
	* If the learning rate is too large (more than two times the ideal rate), your model will overshoot the optimal point and continue to diverge as it hops back and forth at greater magnitudes
	* If the learnign rate is too small, your model will take a long time to reach the optimal point
	* The learning rate is the multiplier used to decide the magnitude of change in the weights by the calculated gradient
	* Learning rate decay: a method where the learning rate is decreased as you reach the ideal point in order to prevent the weights from oscillating back and forth between two points. Some common methods are to use a linearly decaying or exponentially decaying learning rate. 
	* Adaptive Learning Optimizers:
		* [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
		* [AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
	* Scenarios: 
		* If your training error is decreasing very slowly, you can try to increase the learning rate
		* If your trainging error decreases and begins to oscillate, you can either try decreasing your learning rate or try implementing an adaptive learning rate
* Mini-batch size
	* Effects the training speed, resources required. 
	* Online (stochastic) Training: You feed in a single example, calculate the error, update weights and then feed in the next
	* Recommended mini-batch size: b/w one and a few hundred (32 - 256)
	* Could run into an out of memory error if your batch size is too large
	* Benefits of using mini-batches:
		* Less computationally expensive to run on smaller batches
		* Small batches introduce noise on the error calculations which help prevent the optimizer from stopping on local minima
	* You need to change the learning rate as you increase the batch size to compensate for an increased error as the batch size increases as shown in [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)
* Number of training iterations/epochs
	* Generally you would continue to train your model as long as the validation error continues to decreases. Once the validation error plateaus or begins to increase, you use early stopping to preven the model from overfitting.
	* You can use tensorflow's ValidationMonitor to monitor the progress your training and stop the process once your conditions have been met. 
* Number of hidden units/layers
	* Generally, set a number of units that is 'large enough'. The more complex, the more capacity the model will have to learn. 
		* If the model is too complex, it could overfit and try to memorize the data. In order to prevent this, you could implement regularization techniques to prevent overfitting. 
		* In contrast, if the model is too simple, it simple won't be able to learn the underlying patterns of your data. Hence, it is better to increase the complexity of your model by adding units/layers until the model starts to noticeable overfit. Then, you can implement regularization techniques to alleviate the overfitting. 
	* According to Andrej Karpathy, you generally want to stay within 2-3 layers for neural networks, adding more doesn't have much of an effect with the exception of cnn's which perform much better with deeper layers. 

### Tensorflow Functions

* [tf.nn.embedding_lookup(parameters, ids)](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup): Creates a word2vec embedding lookup layer

## Tensorflow

* TODO: Add notes on creating network layers  

## Neural Networks

* This section was completed before this notes document was compiled and will be updated. (I'm more of a handwritten notes guy but I need to practice getting my thoughts out there)

## Convolutional Neural Networks

* This section was completed before this notes document was compiled and will be updated. 

## Transfer Learning

* In many cases you can build off of huge networks that have already been trained such as VGGnet
	* Would save you a lot of training time!
	* Keep the convolutional layers as they've already been trained to detect features and replace the fully connected layers to learn your classification
* Need to scale your images to the image size the pretrained network was initially trained on
* [Repo]( https://github.com/machrisaa/tensorflow-vgg) for pretrained VGGnet

## Recurrent Neural Networks

### General

* Have more flexibility than a vanilla neural network in terms of not being constrained to a fixed input/output size
* RNN's allow you to operate over sequences

### General Architecture and Code

* TODO: Add a diagram of sample architecture here

### RNN Hyperparameters

* Two main choices: cell type (LSTM, Vanilla RNN or Gated Recurrent Unit) and model depth/layers
* In practice, LSTM's and GRU's perform better than Vanilla RNN's (choice is task dependant) (LSTM's more popular)
* Some research suggests LSTM's converge faster and produce better results
* Embedding sizes, for text, are generally between 50-200 but could even be anywhere up to 1000
* Number of LSTM cells per layer: more is better, common numbers are 128, 256, 512, etc. 
* Number of LSTM layers: start with 1 and increase if underfitting
* Batch size (number of reviews being fed into each training iteration): As large as you can hold in memory

### Embeddings and Word2Vec

* A method which represents data with lower dimensional vectors to improve the networks learning ability by allowing the network to more efficiently process data with a large number of classes. 
* With word embeddings, the network is able to learn semantic relationships between words. Word embeddings are learned with Word2Vec
* Word2Vec is a computationally efficient predictive model to learn word embeddings from raw text. 
	* Two types: Continuous Bag-of-Words (CBOW) or Skip-gram
	* CBOW predicts target words from source context-words and is good on small datasets
	* Skip-gram predicts source context-words from the target words and is good for larger data sets as it treats each context-target pair as a new observation in contrast to CBOW which treats the entire context as one observation. (You pass in a word and it predicts the words surrounding it)
* Vanilla definition of 'context': a window of words surrounding the target word.
* Negative Sampling: Instead of updating all the weights after showing the network a single example, only a subset of the weights is updated to make it more efficient. Weights are updated for the correct label but only some of the weights are updated for incorrect labels. 
	* This sounds a lot like the stochastic localized linear updating method that I was brainstorming the other day... 

### Sentiment Analysis with an RNN

* Trained an RNN for binary sentiment analysis (positive/negative) on labelled movie reviews.
* The network architecture consists of a word embedding layer followed by a layer of LSTM cells and then finally an output sigmoid layer.
	* We only care about the final sigmoid output and can ignore the rest. 
	* The ouput of the sigmoid cell will represent the probability of the classification being positive
* Tips:
	* Remove reviews of zero length and truncate reviews to a maximum size to speed upu training
	* For reviews shorter than your maximum length, pad the left with zeros












