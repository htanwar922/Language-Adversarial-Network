
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
