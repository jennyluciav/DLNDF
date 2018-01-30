# Udacity Deep Learning Nanodegree Foundations

Projects and notes from Udacity's Deep Learning Nanodegree Foundations [course](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) and [Github repo](https://github.com/udacity/deep-learning). 

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

* [tf.nn.embedding_lookup(parameters, ids)](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup): Returns a tensor with the embedded vectors
* [tf.contrib.rnn.BasicLSTMCell(num_units)](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell): Creates a layer of LSTM cells
* [tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob)](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper): Wraps lstm cells in another cell and applies dropout to the input and output
* [tf.contrib.rnn.MultiRNNCell([prev_layer] * num_layers)](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell): Creates multiple LSTM layers

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
	* Additional layers will allow capturing of more complex sequences. Overfitting can be prevented by wrapping cells in a dropout layer.
* Batch size (number of reviews being fed into each training iteration): As large as you can hold in memory
* Sequence Length: Should try values close to the length of sentences you're trying to generate. 
	* Look at the average number of words per line in the training data to get an estimate. 

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
* Can use this tf function to create the embedding layer: tf.contrib.layers.embed_sequence(input_data, vocab_size, embed_dim)

### Sentiment Analysis with an RNN

* Trained an RNN for binary sentiment analysis (positive/negative) on labelled movie reviews.
* The network architecture consists of a word embedding layer followed by a layer of LSTM cells and then finally an output sigmoid layer.
	* We only care about the final sigmoid output and can ignore the rest. 
	* The ouput of the sigmoid cell will represent the probability of the classification being positive
* Tips:
	* Remove reviews of zero length and truncate reviews to a maximum size to speed upu training
	* For reviews shorter than your maximum length, pad the left with zeros

### Sequence to Sequence

* Can handle sequential data to output
* A 'many to many' Rnn architecture that can be trained for chatbots and translation models
* Applications: 
	* Translations 
	* Summarization bot ()
	* Q & A bot
	* Chatbot
* Types of chatbot conversation frameworks:
	* Retrieval-based models (closed domain, eg. customer service requests)
	* Generative-based models 
		* generate new responses - technology isn't there yet, can suffer in quality (grammer, spelling, context, etc.)

## Generative Adversarial Networks

### General Notes

* Invented by Ian Goodfellow
* Applications:
	* Used for generating realistic data
		* Stackgan model: taking a description of a bird and generating an image to match it. You can generate an endless amount of novel images (iGAN - a tool where an artist can sketch an image and the GAN will generate similar images)
		* Pix2Pix: Can be used for image to image translation, drawing of cats to images of cats, blueprints to models
	* Photos of day scences to night scenes
	* Photos of a person to a sketch of a person
	* Can even apply it to videos! A Stanford tool called CycleGAN was used to change a horse to a zebra in a video!
		* Background was changed too because of the different environments you'd find each in
	* Imitation Learning: Can learn to imitate actions 
* A form of semi-supervised learning
* GANs allow you to generate an entire image in parallel in contrast to generating each pixel one at a time.
* Training a GAN is different from training a supervised learning model since there is no output to associate each image with. They're trained by adjusting the parameters that maximize the probability that the generator will produce the training data set. 
	* This can be computationally expensive.  
	* This can be approximated by including a second network, the discriminator network. This network is shown real images from the training set and fake images produced by the generator network and tries to assign a high probability to real images(1) and low probability to the generated images. 
	* Generator is trained to output images that the discriminator would assign a high probability. 
	* Ideally they converge to a point where there is a uniform probability of the image being either real or fake.

### Game Theory

* The two networks are adversarially in conflict with one another and can be understood mathematically with game theory
* Founded by John Von Neumann and extended by John Nash
* Can be used to model cooperation and conflict between two rational agents in any situation where each agent can choose from a set of possible actions, which leads to a well-defined pay-off
* Most machine learning models so far have been based on optimization of a cost-function for chosen parameters
	* With GANs, there are two networks you're training adversarially each with their own cost. The cost of the second network being the inverse cost function of the first. 
	* From game theory, equilibrium is reached once neither player can improve thier situation without changing the other player's strategy. For GANs this happens when you have the maximum value for the discriminator and the minimum point for the generator. A saddle point on the optimization graph. 
* From game theory, we can show that if both networks are big enough, there is a point where the generator density equals the true density and the discriminator outputs 1/2 everywhere. 
	* May not find it in practice because the optimization algorithms may not find the true minimum. (The generator could learn to produce clusters of representative density that the discriminator eventually isn't tricked by leading the the generator then producing a new convincing cluster)
* Minimax: a strategy of always minimizing the maximum possible loss resulting from the choice a player makes.
* Nash equilibrium: The equilibrium of a game where no player has any incentive to deviate from their strategy after considering their opponents choice

### Training Tips and Tricks

* For simple tasks, you can use a fully connected architecture with no convolutions or recurrence
* Both generator and discriminator should have at least one hidden layer 
* Leaky ReLU is a popular activation function (many others will work)
	* Helps to ensure gradients flow through entire architecture (The discriminator network being at the end of the architecture needs to be able to see the gradient)
* For the output layer of the generator, a tanh activation function which allows your data to be scaled between -1 and 1
* Define individual loss functions for the generator and the discriminator and then assign an optimizer to minimize loss for discriminator while simultaneously assigning an optimizer to minimize loss for teh discriminator
	* Training the discriminator to output a 1 for real images and 0 for fake
	* AdamOptimizer is good
	* Can use a sigmoid_cross_entropy loss for the discriminator
	* Make sure you use numerically stable cross-entropy which uses the logits (before sigmoid - can introduce rounding error if you use the probabilities)
	* Generator loss withh also be cross_entropy but with the labels flipped
* For larger images you can replace inner layers with convolution layers in the generator network 
	* Instead of decreasing the size of the feature maps per layer you would increase it in this case (The stride in the output map is greater than the input match)
	* Use batch normalization on most layers (DCGAN authors recommend using batch normalization on every layer except the output layer of the generator and the input layer of the discriminator)
		* Necessary for deep GANs
	


















