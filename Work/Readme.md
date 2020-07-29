#### Note:
For abstract model description, please refer parent directory.

Download (Upload) the files to Google Drive folder Colab Notebooks:

The directories should be:
'My Drive/Colab Notebooks/Work/'
Copy the files extract_data.ipynb and LAN_v3.ipynb in this folder.

In that two directories get created after running extract_data.ipynb (you can create yourself if you prefer):

> 'My Drive/Colab Notebooks/Work/Amazon reviews/'

>> 'My Drive/Colab Notebooks/Work/Amazon reviews/train'

>> 'My Drive/Colab Notebooks/Work/Amazon reviews/dev'

>> 'My Drive/Colab Notebooks/Work/Amazon reviews/test'

> 'My Drive/Colab Notebooks/Work/bwe/'

>> 'My Drive/Colab Notebooks/Work/bwe/vectors'

Next run the LAN_v3.ipynb.

If the files are copied to some other folder instead, you may have to modify the top cells in both the files accordingly.
After running the first cells, click the link generated and enter the authorization code for same drive from which the files are being run.

In extract_data.ipynb, you can select which language datas and word-embeddings to download.

# About LAN:
1. The basic architecture involves Embeddings layer, Averaging layer, Feature Extractor Model, Sentiment Classifier Model and Language Detector Model.
2. The overall model has two inputs, which are the padded reviews converted to sequences, and the corresponding lengths of the reviews just before padding. The lengths are required for averaging the embeddings throught the averaging layer.

## Parameters:
1. Input1 : Padded sequences
2. Input2 : Actual sequence lengths
3. Output1 : Predicted labels/star ratings
4. Output2 : Predicted language
5. Labels : Actual star ratings (for source language training)

## Embedding Averager model:
1. The Embeddings and Averaging Layers are put as separate model for training purposes.
2. The inputs are first passed through this model to convert the padded sequences to embeddings and then average them to produce averaged embeddings (like BOW approach) as outputs.

## Feature Extractor:
1. This is the base model for feature extraction; during training this base remains the same for both branches. Thus while during sentiment analysis, the weights are updated accoring to averaged labeled reviews, during language classification, it happens with language-labeled data, for which reviews are not a necessity.
2. The motive is to somehow take advantage of language detection in sentiment classification through feature extraction.
3. Averaged embeddings are given as inputs and the outputs of the final Dense layer are the features we're refering to.

## Sentiment Classifier:
1. This is the main objective model; classification/labeling of reviews. The training is done in this repository through source language (English) reviews labeled data.
2. The features are given as inputs and the outputs are the label predictions.
3. Loss is evaluated as Sparse Categorical Crossentropy loss.

## Language Detector:
1. This model is for the adversarial training. It has been assumed that the model can't be trained on target language due to lack of labeled target reviews data. Thus the features of target language are learned through this model. Therefore, training of this branch of LAN updates the already trained feature extractor weights to adjust for target language features.
2. The inputs are same as sentiment classifier, i.e., the features extracted through feature extractor. The outputs are binary labels with +1 representing source and -1 representing target language data, from the output of a tanh layer.
3. The loss is evaluated as Hinge loss.


# Running the notebook
The notebook has been divided in several subsections, so that it can be easily converted for local machine running as well.

## Mounting of google drive:
(Only while running the notebook through drive)

## Options:
Options here refers to the CLI arguments that are passed through sys.argv variable in python. Since The execution through notebook doesn't include CLI arguments, the sys.argv list needs to be explicitly updated.
This section also creates necessary folders if they're not already existing.

## Utils:
Some common use and debugging functions are defined here. In higher versions, the saving and loading of models is also defined, which is needed to create checkpoints, as the all the models can't be directly saved and imported; LAN as a whole is saved and loaded and the component models have to be reconfigured from the same only. (This is also another reason for division of feature extractor in two parts).

## Data:
The Amazon Reviews class definition which reads the downloaded data from the datapath mentioned in Options section.

## Layers:
This section defines the Averaging layer in this version.

## Models:
This section creates the models which are described above, with their inputs and outputs, and their combinations for training.
The fundamental are names as EA, F, P and Q respectively. These are combined to produce models named EAF, EAFP, EAFQ and EAFPQ/LAN models.

## Training:
The training was divided in 2 major parts:
### > Without trainable embeddings
The embedding layer weights (or weights of EA model) are kept untrainable or constant for this, so that only the F, P and Q weights get updated.
1. The EAFP or sentiment classifier branch is trained on source language reviews data, thus updating F and P model weights.
2. The EAFQ or language detetor branch is trained on source and target data, updating F and Q models.
### > With trainable embeddings
The embedding layer weights are also made trainable, so that weights of EA are also updated during training.
1. The EAFP or sentiment classifier branch is trained on source language reviews data, thus updating EA, F and P model weights.
2. The EAFQ or language detetor branch is trained on source and target data, updating EA, F and Q models.
3. The overall LAN is trained, updating weights of EA, F, P as well as Q at the same time. For this, the labels need to be zipped with the language-labels and shuffled so that loss can include both Sparse Categorical Crossentropy loss for sentiment classification, as well as Hinge loss for language detection.

## Evaluation:
The model is evaluated on (unseen) target data.

## Saving and Reloading of at different checkpoints:
The model may need to be saved and reloaded at many places to cater to unexplained crashes while running.
### Saving:
Saving LAN will save all the models in a common model (just like usual saves as in TF documentation).
### Loading:
1. Load the model as single LAN model.
2. Look at LAN structure with LAN.summary() function.
3. The summary should show three sequential models in the bottom. These are F, P and Q (in top-to-bottom order in v3 implementation).
4. These layers and models can be accessed with LAN.layers and stored in (_, E, _, A, F, P, Q) in this order.
5. There would be two inputs for padded review sequences and corresponding actual lengths. These have to be used for inputting to the other complex models, i.e., EA, EAF, EAFP, EAFQ. The inputs can be accessed with LAN.inputs and stored in (input1, input2).

The Utils section in v5 has these two functions implemented as load_models and save_models.
