# EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
This is the code for the ICLR Workshop paper [EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks.](https://arxiv.org/abs/1901.11196) A blog post that explains EDA is [here](https://medium.com/@jason.20/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610). 

By [Jason Wei](https://jasonwei20.github.io/research/) and Kai Zou, with Protago Labs AI Research.

We present **EDA**: **e**asy **d**ata **a**ugmentation techniques for boosting performance on text classification tasks. These are a generalized set of data augmentation techniques that are easy to implement and have shown improvements on five NLP classification tasks, with substantial improvements on datasets of size *N<500*. While other techniques require you to train a language model on an external dataset just to get a small boost, we found that simple text editing operations using EDA result in substantial performance gains. Given a sentence in the training set, we perform the following operations:

- **Synonym Replacement (SR):** Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
- **Random Insertion (RI):** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
- **Random Swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
- **Random Deletion (RD):** For each word in the sentence, randomly remove it with probability *p*.

<p align="center"> <img src="eda_figure.png" alt="drawing" width="400" class="center"> </p>
Average performance on 5 datasets with and without EDA, with respect to percent of training data used.

# Usage

You can run EDA any text classification dataset in less than 5 minutes. Just two steps:

### Install NLTK (if you don't have it already):

Pip install it.

```
pip install -U nltk
```

Download WordNet.
```
python
>>> import nltk; nltk.download('wordnet')
```

### Run EDA

You can easily write your own implementation, but this one takes input files in the format `label\tsentence` (note the `\t`). So for instance, your input file should look like this (example from stanford sentiment treebank):

```
1   neil burger here succeeded in making the mystery of four decades back the springboard for a more immediate mystery in the present 
0   it is a visual rorschach test and i must have failed 
0   the only way to tolerate this insipid brutally clueless film might be with a large dose of painkillers
...
```

Now place this input file into the `data` folder. Run 

```
python code/augment.py --input=<insert input filename>
```

The default output filename will append `eda_` to the front of the input filename, but you can specify your own with `--output`. You can also specify the number of generated augmented sentences per original sentence using `--num_aug` (default is 9). Furthermore, you can specify the alpha parameter, which approximately means the percent of words in the sentence that will be changed (default is `0.1` or `10%`). So for example, if your input file is `sst2_train.txt` and you want to output to `sst2_augmented.txt` with `16` augmented sentences per original sentence and `alpha=0.05`, you would do:

```
python code/augment.py --input=sst2_train.txt --output=sst2_augmented.txt --num_aug=16 --alpha=0.05
```

Note that at least one augmentation operation is applied per augmented sentence regardless of alpha. So if you do alpha=0.001 and your sentence only has four words, one augmentation operation will still be performed. Best of luck!

# Experiments (Coming soon)

### Word embeddings
Download [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) and place in a folder named `word2vec`.
数据增强技术已经是图像领域的标配，通过对图像的翻转、旋转、镜像、高斯白噪声等技巧实现数据增强。然而，在NLP领域，情况有所不同：改变某个词汇可能会改变整个句子的含义，那么在NLP领域，如何使用数据增强技术呢？

ICLR 2019 workshop 论文《EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks》介绍了几种NLP数据增强技术，并推出了[EDA github代码](jasonwei20/eda_nlp)。EDA github repo提出了四种简单的操作来进行数据增强，以防止过拟合，并提高模型的泛化能力。下面进行简单的介绍:


1. 同义词替换（SR: Synonyms Replace）：不考虑stopwords，在句子中随机抽取n个词，然后从同义词词典中随机抽取同义词，并进行替换。

2. 随机插入(RI: Randomly Insert)：不考虑stopwords，随机抽取一个词，然后在该词的同义词集合中随机选择一个，插入原句子中的随机位置。该过程可以重复n次。

3. 随机交换(RS: Randomly Swap)：句子中，随机选择两个词，位置交换。该过程可以重复n次。

4. 随机删除(RD: Randomly Delete)：句子中的每个词，以概率p随机删除。


上述四种数据增强技术效果如何呢？（这里以文本分类实验为例）

在英文的数据上出奇的好！经过上述四种操作，数据增强后的句子可能不易理解，但作者们发现模型变得更加鲁棒了，尤其是在一些小数据集上。实验结果如下：




我们可以发现，仅仅使用50%的训练数据，使用EDA就能够达到原始模型使用100%训练数据的效果。


可能有人会问：使用EDA技术后，改变后的数据和原数据的特征空间分布与其标签是否一致？

作者同样做了一个实验，在产品评论数据集上，使用RNN结合EDA数据增强进行训练，将最后一层的输出作为文本特征，使用tSNE进行可视化，得到如下结果：




可以发现，数据增强后的数据分布与原始数据的分布非常吻合。这也意味着EDA对于句子的修改，并没有改变数据分布以及标签的分布。


上述四种数据增强技术，每一个都很有用吗？

实验结果如下：



对于四种技术，数据集很小时，每一种技术都能够有2-3%的提升，当数据集很大时，每一种技术也能够有1%的提升。根据作者的经验来看，不要改变超过1/4的词汇的前提下，模型的鲁棒性都能得到很大的提升。


既然EDA很有用，大家可能有一个问题：我们要产生多少个句子来进行数据增强呢？

答案是取决于训练数据的大小。当训练数据很小时，模型更容易过拟合，这时建议多生成一些数据增强的样本。当训练数据很大时，大量增加数据增强样本可能没有帮助，因为模型本身可能已经能够泛化。实验结果如下：




其中，横轴是一个句子平均产生的增强样本数目，纵轴是模型增益。我们可以看到，当一个句子平均产生4-8个新句子时，模型提升就能达到很好的效果。训练数据越少，提升效果效果越明显。过多的数据增强数据实际上对模型的提升有限。

EDA中的四种简单的NLP数据增强技术能够有效提升文本分类的模型表现。
