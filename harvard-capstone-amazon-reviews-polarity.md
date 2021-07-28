---
title: "Amazon Reviews  \n Sentiment Analysis/Text Classification  \n Choose Your Own Project  \n A Harvard Capstone Project"
author: "Manoj Bijoor"
email: manoj.bijoor@gmail.com
date: "July 28, 2021"
output: 
  pdf_document: 
    latex_engine: xelatex
    number_sections: yes
    keep_tex: yes
    keep_md: yes
    df_print: kable
    highlight: pygments
    extra_dependencies: "subfig"
  md_document:
    variant: markdown_github 
    # check https://bookdown.org/yihui/rmarkdown/markdown-document.html#markdown-variants
  github_document:
    toc: true
    toc_depth: 5
    pandoc_args: --webtex
    # pandoc_args: ['--lua-filter', 'math-github.lua']
  html_document:
    keep_md: true
    code_folding: hide
# urlcolor: blue
# linkcolor: blue
#citecolor: blue
#geometry: margin=1in
always_allow_html: true
links-as-notes: true
header-includes:
  \usepackage[utf8]{inputenc}
  \usepackage[english]{babel}
  \usepackage{bookmark}
  \usepackage[]{hyperref}
  \hypersetup{
    backref,
    pdftitle={"Amazon Review Polarity Harvard Capstone"},
    bookmarks=true,
    bookmarksnumbered=true,
    bookmarksopen=true,
    bookmarksopenlevel=3,
    pdfpagemode=FullScreen,
    pdfstartpage=1,
    hyperindex=true,
    pageanchor=true,
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan
    }
  \usepackage{amsmath}
  \usepackage{pdflscape}
  \usepackage[titles]{tocloft}
  \usepackage{tocloft}
  \usepackage{titlesec}
  \usepackage{longtable}
  \usepackage{xpatch}
  \usepackage[T1]{fontenc}
  \usepackage{imakeidx}
  \makeindex[columns=3, title=Alphabetical Index, intoc]

  # \usepackage{amssymb}
  # \usepackage{mathtools}
  # \usepackage{unicode-math}
  # \usepackage{fontspec}
  # \usepackage{letltxmacro}%
  # \usepackage{float}
  # \usepackage{flafter}
  # \usepackage[titles]{tocloft}
---













<!-- ------------------------------ -->

\bookmark[dest=TitlePage]{Title Page}

\pagenumbering{roman}     <!-- first page with Roman numbering -->

\newpage                  <!-- new page -->

<!-- ------------------------------ -->

\newpage 

\begin{center}

\hypertarget{Abstract}{}
\large{Abstract}
\bookmark[dest=Abstract]{Abstract}

\end{center}

\bigskip

Deriving truth and insight from a pile of data is a powerful but error-prone job. 

This project offers an empirical exploration on the use of Neural networks for text classification using the Amazon Reviews Polarity dataset. 

Text classification algorithms are at the heart of a variety of software systems that process text data at scale.

One common type of text classification is sentiment analysis, whose goal is to identify the polarity of text content: the type of opinion it expresses. This can take the form of a binary like/dislike rating, or a more granular set of options, such as a star rating from 1 to 5. Examples of sentiment analysis include analyzing Twitter posts to determine if people liked the Black Panther movie, or extrapolating the general public’s opinion of a new brand of Nike shoes from Walmart reviews.

Algorithms such as regularized linear models, support vector machines, and naive Bayes models are used to predict outcomes from predictors including text data. These algorithms use a shallow (single) mapping. In contrast, Deep learning models approach the same tasks and have the same goals, but the algorithms involved are different. Deep learning models are "deep" in the sense that they use multiple layers to learn how to map from input features to output outcomes.

Deep learning models can be effective for text prediction problems because they use these multiple layers to capture complex relationships in language.

The layers in a deep learning model are connected in a network and these models are called Neural Networks.

Neural language models (or continuous space language models) use continuous representations or embeddings of words to make their predictions. These models make use of Neural networks.

Continuous space embeddings help to alleviate the curse of dimensionality in language modeling: as language models are trained on larger and larger texts, the number of unique words (the vocabulary) increases. The number of possible sequences of words increases exponentially with the size of the vocabulary, causing a data sparsity problem because of the exponentially many sequences. Thus, statistics are needed to properly estimate probabilities. Neural networks avoid this problem by representing words in a distributed way, as non-linear combinations of weights in a neural net. 

Instead of using neural net language models to produce actual probabilities, it is common to instead use the distributed representation encoded in the networks' "hidden" layers as representations of words; each word is then mapped onto an n-dimensional real vector called the word embedding, where n is the size of the layer just before the output layer. 
An alternate description is that a neural net approximates the language function and models semantic relations between words as linear combinations, capturing a form of compositionality. 

In this project we will cover four network architectures, namely DNN, CNN, sepCNN and BERT. We will also first implement a Baseline linear classifier model which serves the purpose of comparison with the deep learning techniques.  

For metrics we will use the default performance parameters for binary classification which are Accuracy, Loss and ROC AUC (area under the receiver operator characteristic curve).

<!-- ------------------------------ -->

\newpage 
\clearpage
\phantomsection
\setcounter{secnumdepth}{5}
\setcounter{tocdepth}{5}

\cleardoublepage <!-- ensure that the hypertarget is on the same page as the TOC heading -->
\hypertarget{toc}{} <!-- set the hypertarget -->
\bookmark[dest=toc,level=chapter]{\contentsname}
\tableofcontents

\clearpage

<!-- ------------------------------ -->
<!-- \renewcommand{\theHsection}{\thepart.section.\thesection} -->

\newpage
\clearpage
\phantomsection
# List of tables{-}
\renewcommand{\listtablename}{} <!-- removes default section name -->

\listoftables
\clearpage

\newpage
\clearpage
\phantomsection
# List of figures{-}
\renewcommand{\listfigurename}{}

\listoffigures
\clearpage

\newpage
\clearpage
\phantomsection
\newcommand{\listequationsname}{List of Equations}
\newlistof{equations}{equ}{\listequationsname}
\newcommand{\equations}[1]{%
\refstepcounter{equations}
\addcontentsline{equ}{equations}{ \protect\numberline{\theequations}#1}\par}
\xpretocmd{\listofequations}{\addcontentsline{toc}{section}{\listequationsname}}{}{}

\renewcommand{\listequationsname}{}

\listofequations
\clearpage

<!-- ------------------------------ -->

\newpage

\pagenumbering{arabic} 

<!-- ------------------------------ -->

\newpage
# Project Overview: Amazon Reviews Polarity

## Introduction

Deriving truth and insight from a pile of data is a powerful but error-prone job. 

Text classification algorithms are at the heart of a variety of software systems that process text data at scale.

One common type of text classification is sentiment analysis, whose goal is to identify the polarity of text content: the type of opinion it expresses. This can take the form of a binary like/dislike rating, or a more granular set of options, such as a star rating from 1 to 5. Examples of sentiment analysis include analyzing Twitter posts to determine if people liked the Black Panther movie, or extrapolating the general public’s opinion of a new brand of Nike shoes from Walmart reviews.

Algorithms such as regularized linear models, support vector machines, and naive Bayes models are used to predict outcomes from predictors including text data. These algorithms use a shallow (single) mapping. In contrast, Deep learning models approach the same tasks and have the same goals, but the algorithms involved are different. Deep learning models are "deep" in the sense that they use multiple layers to learn how to map from input features to output outcomes.

Deep learning models can be effective for text prediction problems because they use these multiple layers to capture complex relationships in language.

The layers in a deep learning model are connected in a network and these models are called neural networks.   

### Neural networks  

Neural language models (or continuous space language models) use continuous representations or [embeddings of words](https://en.wikipedia.org/wiki/Word_embedding) to make their predictions.[ Karpathy, Andrej. "The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) These models make use of [Neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network).

Continuous space embeddings help to alleviate the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) in language modeling: as language models are trained on larger and larger texts, the number of unique words (the vocabulary) increases.[Heaps' law](https://en.wikipedia.org/wiki/Heaps%27_law). The number of possible sequences of words increases exponentially with the size of the vocabulary, causing a data sparsity problem because of the exponentially many sequences. Thus, statistics are needed to properly estimate probabilities. Neural networks avoid this problem by representing words in a distributed way, as non-linear combinations of weights in a neural net.[Bengio, Yoshua (2008). "Neural net language models". Scholarpedia. 3. p. 3881. Bibcode:2008SchpJ...3.3881B. doi:10.4249/scholarpedia.3881](https://ui.adsabs.harvard.edu/abs/2008SchpJ...3.3881B/abstract) An alternate description is that a neural net approximates the language function.

Instead of using neural net language models to produce actual probabilities, it is common to instead use the distributed representation encoded in the networks' "hidden" layers as representations of words;  
A hidden layer is a synthetic layer in a neural network between the input layer (that is, the features) and the output layer (the prediction). Hidden layers typically contain an activation function such as [ReLU](https://developers.google.com/machine-learning/glossary?utm_source=DevSite&utm_campaign=Text-Class-Guide&utm_medium=referral&utm_content=glossary&utm_term=sepCNN#rectified-linear-unit-relu) for training. A deep neural network contains more than one hidden layer. Each word is then mapped onto an n-dimensional real vector called the word embedding, where n is the size of the layer just before the output layer. The representations in skip-gram models for example have the distinct characteristic that they model semantic relations between words as [linear combinations](https://en.wikipedia.org/wiki/Linear_combination), capturing a form of [compositionality](https://en.wikipedia.org/wiki/Principle_of_compositionality). 


In this project we will cover four network architectures, namely:

1. DNN - Dense Neural Network - a bridge between the "shallow" learning approaches and the other 3 - CNN, sepCNN, BERT.  

2. CNN - Convolutional Neural Network - advanced architecture appropriate for text data because they can capture specific local patterns.  

3. sepCNN - Depthwise Separable Convolutional Neural Network.  

4. BERT - Bidirectional Encoder Representations from Transformers.  

We will also first implement a Baseline linear classifier model which serves the purpose of comparison with the deep learning techniques we will implement later on, and also as a succinct summary of a basic supervised machine learning analysis for text.  

This linear baseline is a regularized linear model trained on the same data set, using tf-idf weights and 5000 tokens.  

For metrics we will use the default performance parameters for binary classification which are Accuracy, Loss and ROC AUC (area under the receiver operator characteristic curve).

We will also use the confusion matrix to get an overview of our model performance, as it includes rich information.

We will use tidymodels packages along with Tensorflow, the R interface to Keras. See [Allaire, JJ, and François Chollet. 2021. keras: R Interface to ’Keras’](https://CRAN.R-project.org/package=keras) for preprocessing, modeling, and evaluation, and [Silge, Julia, and David Robinson. 2017. Text Mining with R: A Tidy Approach. 1st ed. O’Reilly Media, Inc.](https://www.tidytextmining.com/), [Supervised Machine Learning for Text Analysis in R, by Emil Hvitfeldt and Julia Silge.](https://smltar.com/) and [Tidy Modeling with R, Max Kuhn and Julia Silge, Version 0.0.1.9010, 2021-07-19](https://www.tmwr.org/) and how can we forget [Introduction to Data Science, Data Analysis and Prediction Algorithms with R - Rafael A. Irizarry, 2021-07-03](https://rafalab.github.io/dsbook/).  

The keras R package provides an interface for R users to Keras, a high-level API for building neural networks.

This project will use some key machine learning best practices for solving text classification problems.  
Here’s what you’ll learn:

1. The high-level, end-to-end workflow for solving text classification problems using machine learning
2. How to choose the right model for your text classification problem
3. How to implement your model of choice using TensorFlow with Keras acting as an interface for the TensorFlow library

I have used/mentioned several references throughout the project.  

This project depends on python and R software for Tensorflow and Keras that needs to be installed both inside and outside of R. As each individual's environment may be different, I cannot automate this part in my code.

R side:  
https://cran.r-project.org/  
https://tensorflow.rstudio.com/installation/  
https://tensorflow.rstudio.com/installation/gpu/local_gpu/  

Python side:  
https://www.tensorflow.org/install  
https://www.anaconda.com/products/individual  
https://keras.io/  

Instead of cluttering code with comments, I ask you to please use these references and the rstudio help (?cmd/??cmd) if you are not very familiar with any specific command.  Most commands are pretty self explanatory if you are even a little familiar with R.

Here are some more references:  

## References

[Tensorflow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.  

The [TensorFlow Hub](https://tfhub.dev/) lets you search and discover hundreds of trained, ready-to-deploy machine learning models in one place.

[Tensorflow for R](https://tensorflow.rstudio.com/) provides an R interface for Tensorflow.  

[Tidy Modeling with R](https://www.tmwr.org/)  

[Tinytex](https://yihui.org/tinytex/)  
I have used tinytex in code chunks.  

[Latex](https://www.overleaf.com/learn/latex)  
I have used Latex beyond the very basic provided by default templates in RStudio. Too numerous to explain. Though that much is not needed, I have used it to learn and make better pdf docs.  

[Rmarkdown](https://bookdown.org/yihui/rmarkdown)  


\newpage
## Text Classification Workflow

Here’s a high-level overview of the workflow used to solve machine learning problems:

Step 1: Gather Data  
Step 2: Explore Your Data  
Step 2.5: Choose a Model*  
Step 3: Prepare Your Data  
Step 4: Build, Train, and Evaluate Your Model  
Step 5: Tune Hyperparameters  
Step 6: Deploy Your Model  

The following sections explain each step in detail, and how to implement them for text data.

### Gather Data
Gathering data is the most important step in solving any supervised machine learning problem. Your text classifier can only be as good as the dataset it is built from.

Here are some important things to remember when collecting data:

1. If you are using a public API, understand the limitations of the API before using them. For example, some APIs set a limit on the rate at which you can make queries.  

2. The more training examples/samples you have, the better. This will help your model generalize better.  

3. Make sure the number of samples for every class or topic is not overly imbalanced. That is, you should have comparable number of samples in each class.  

4. Make sure that your samples adequately cover the space of possible inputs, not only the common cases.  

This dataset contains amazon reviews posted by people on the Amazon website, and is a classic example of a sentiment analysis problem.

Amazon Review Polarity Dataset - Version 3, Updated 09/09/2015


ORIGIN

The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review. For more information, please refer to the following paper: [J. McAuley and J. Leskovec. Hidden factors and hidden topics: Understanding rating dimensions with review text. In Proceedings of the 7th ACM Conference on Recommender Systems, RecSys ’13, pages 165–172, New York, NY, USA, 2013. ACM](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf).

The Amazon reviews polarity dataset was constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).  Advances in Neural Information Processing Systems 28 (NIPS 2015).

Here is an Abstract of that paper:  

This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.

Coming back to our project: As Google has changed it's API, I had to download the dataset manually from the following URL:    

Please select file named "amazon_review_polarity_csv.tar.gz" and download it to the project directory.

Download Location URL : [Xiang Zhang Google Drive](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ)


DESCRIPTION

The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. Each class has 1,800,000 training samples and 200,000 testing samples.

The files train.csv and test.csv contain all the training samples as comma-separated values. There are 3 columns in them, corresponding to label/class index (1 or 2), review title and review text. The review title and text are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\textbackslash n".


\newpage
### Explore Your Data
Building and training a model is only one part of the workflow. Understanding the characteristics of your data beforehand will enable you to build a better model. This could simply mean obtaining a higher accuracy. It could also mean requiring less data for training, or fewer computational resources.

#### Load the Dataset
First up, let’s load the dataset into R.

In the dataset, class 1 is the negative and class 2 is the positive review. We will change these to 0 and 1.

columns = (0, 1, 2) \# 0 - label/class index, 1 - title/subject, 2 -
text body/review.

In this project we will NOT be using the "title" data. We will use only "label" and "text".
Also note that I have more comments in the code file/s than in the pdf document.


```r
untar("amazon_review_polarity_csv.tar.gz", list = TRUE)  ## check contents
[1] "amazon_review_polarity_csv/"          
[2] "amazon_review_polarity_csv/test.csv"  
[3] "amazon_review_polarity_csv/train.csv" 
[4] "amazon_review_polarity_csv/readme.txt"
untar("amazon_review_polarity_csv.tar.gz")
```








\newpage
#### Check the Data
After loading the data, it’s good practice to run some checks on it: pick a few samples and manually check if they are consistent with your expectations. For example see Table \ref{tbl:amazon_train}


```r
glimpse(amazon_train)
Rows: 2,879,960
Columns: 3
$ label <dbl> 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0~
$ title <chr> "For Older Mac Operating Systems Only", "greener today than yest~
$ text  <chr> "Does not work on Mac OSX Per Scholastic tech support If you hav~
```

\begin{table}[H]

\caption{\label{tab:chk_data_2}Amazon Train  data\label{tbl:amazon_train}}
\centering
\fontsize{6}{8}\selectfont
\begin{tabular}[t]{rll}
\toprule
label & title & text\\
\midrule
0 & For Older Mac Operating Systems Only & Does not work on Mac OSX Per Scholastic tech support If you have a newer Mac machine with one of the later OS versions ( or later) the game will not work as it was meant to run on an older system\\
1 & greener today than yesterday & I was a skeptic however I nowhave three newspapers the bible six books and two magazines on my kindle I ve wowed my church reading group with skipping back and forth between our book we are reading and the bible I ve eliminated several paper subscriptions and actually am reading more because i can easily carry it all with me The page is easy on the eyes I really like it\\
0 & Overrated very overrated & This film is a disappointment Based on the reviews I read here at Amazon and in other media I expected an exciting and visually engrossing experience This is a nineteen nineties touchy feely melodrama and furthermore it s poorly done Can I get my money back\\
1 & well really & dmst are my favourite band and have been for a while now just mind blowing i wanted to make one correction to the previous review though the do makes are from toronto not quebec\\
1 & dynomax super turbo exhaust & it fit good took only about mins to put on little quiter than i wanted ( but for the price can t beat it ) it is getting a littler louder everyday i drive it starting to burn some of the glass packing out\\
1 & East LA Marine the Guy Gabaldon Story & This movie puts history in perspective for those who know the story of Guy Gabaldon I think that he is an unsung hero that should have been awarded for the lives that he saved on both sides I think that you will be glad that you watched this movie Not the Hollywood version but through this you get to meet the actual hero not Hollywood s version of who they thought he should be\\
0 & World of Bad Support & Before getting this game be sure to check out the Support forums WoW (World of warcraft) is suffering from things like Peoples accounts expiring with no way to get a CC or game card updated High graphics glitches make game unplayableHigh rate of computer lock ups freezing and Blue screening Blizzards support staff ask the users to update drivers There latest patch has caused a large amount of players to be unable to play at all So make sure that your computer won t have these issues Even though systems with gig of ram and the best video cards have issues maybe yours won t I recommended waiting for Blizzard to finish the stress test they call GOLD Instead get any other MMORPG none are having the issues this one has If you do buy it and can t play please note that for the last days Blizzards support line has been ringing fast busy hehe\\
0 & disapointing & Only two songs are great Desire All I want is you There are some good live performaces but Helter Skelter and All along the watch tower covers were very bad moves If you re a die hard fan buy this for the two songs I mentioned because they re classics but otherwise this is hardly essential\\
1 & SAITEK X FLIGHT CONTROL SYSTEM BLOWS YOU AWAY & When I purchased my Flight Simulator Deluxe Edition I chose to purchase these controls as well I wanted as real a feel as I could get with my gaming system Well at the time they were the best on the shelf that I could find Nothing else came close to these The first few reviewers have explained the controls already They are right on the money You will want to purchase these along with your game\\
1 & the secret of science & The best kept secret of science is how strongly it points towards a creator and dovetails with Christianity In this marvelously lucid book the eminent physical chemist Henry Schaefer unfolds the secret\\
\bottomrule
\end{tabular}
\end{table}

Labels : Negative reviews = 0, Positive reviews = 1

```r
unique(amazon_train$label)
[1] 0 1
```

\newpage
#### Collect Key Metrics

Once you've verified the data, collect the following
important metrics that can help characterize your text classification
problem:

1.Number of samples: Total number of examples you have in the data.

2.Number of classes: Total number of topics or categories in the data.

3.Number of samples per class: Number of samples per class
(topic/category). In a balanced dataset, all classes will have a similar
number of samples; in an imbalanced dataset, the number of samples in
each class will vary widely.

4.Number of words per sample: Median number of words in one sample.

5.Frequency distribution of words: Distribution showing the frequency
(number of occurrences) of each word in the dataset.

6.Distribution of sample length: Distribution showing the number of words
per sample in the dataset.
  

Number of samples

```r
(num_samples <- nrow(amazon_train))
[1] 2879960
```


Number of classes

```r
(num_classes <- length(unique(amazon_train$label)))
[1] 2
```


Number of samples per class

```r
# Pretty Balanced classes
(num_samples_per_class <- amazon_train %>%
    count(label))
```



\begin{tabular}{r|r}
\hline
label & n\\
\hline
0 & 1439405\\
\hline
1 & 1440555\\
\hline
\end{tabular}


Number of words per sample

```r
amazon_train_text_wordCount <- sapply(temp, length)

(mean_num_words_per_sample <- mean(amazon_train_text_wordCount))
[1] 75.89602

(median_num_words_per_sample <- median(amazon_train_text_wordCount))
[1] 67
```


\newpage
#### Tokenization

To build features for supervised machine learning from natural language, we need some way of representing raw text as numbers so we can perform computation on them. Typically, one of the first steps in this transformation from natural language to feature, or any of kind of text analysis, is tokenization. Knowing what tokenization and tokens are, along with the related concept of an n-gram, is important for almost any natural language processing task.  

Tokenization in NLP/Text Classification is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens.

For Frequency distribution of words(nrams) and for Top 25 words see Table \ref{tbl:train_words} and Figure \ref{fig:model_1}

\begin{table}

\caption{\label{tab:freq_dist_ngrams}Frequency distribution of words\label{tbl:train_words}}
\centering
\begin{tabular}[t]{lrrrr}
\toprule
word & n & total & rank & term frequency\\
\midrule
the & 11106073 & 218378699 & 1 & 0.0508569\\
i & 6544247 & 218378699 & 2 & 0.0299674\\
and & 6018678 & 218378699 & 3 & 0.0275607\\
a & 5452771 & 218378699 & 4 & 0.0249693\\
to & 5398243 & 218378699 & 5 & 0.0247196\\
it & 5028997 & 218378699 & 6 & 0.0230288\\
of & 4325341 & 218378699 & 7 & 0.0198066\\
this & 4083354 & 218378699 & 8 & 0.0186985\\
is & 3850538 & 218378699 & 9 & 0.0176324\\
in & 2594242 & 218378699 & 10 & 0.0118796\\
\bottomrule
\end{tabular}
\end{table}



```
Warning: Ignoring unknown parameters: binwidth
```

![Frequency distribution of words(nrams) for Top 25 words\label{fig:model_1}](figures/plot_freq_dist_ngrams-1.pdf) 


\newpage
#### Stopwords

Once we have split text into tokens, it often becomes clear that not all words carry the same amount of information, if any information at all, for a predictive modeling task. Common words that carry little (or perhaps no) meaningful information are called stop words. It is common advice and practice to remove stop words for various NLP tasks.  

The concept of stop words has a long history with Hans Peter Luhn credited with coining the term in 1960. [Luhn, H. P. 1960. “Key Word-in-Context Index for Technical Literature (kwic Index).” American Documentation 11 (4): 288–295. doi:10.1002/asi.5090110403](https://doi.org/10.1002/asi.5090110403). Examples of these words in English are “a,” “the,” “of,” and “didn’t.” These words are very common and typically don’t add much to the meaning of a text but instead ensure the structure of a sentence is sound.  

Historically, one of the main reasons for removing stop words was to decrease the computational time for text mining; it can be regarded as a dimensionality reduction of text data and was commonly used in search engines to give better results [Huston, Samuel, and W. Bruce Croft. 2010. “Evaluating Verbose Query Processing Techniques.” In Proceedings of the 33rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 291–298. SIGIR ’10. New York, NY, USA: ACM. doi:10.1145/1835449.1835499](https://doi.org/10.1145/1835449.1835499).

For Frequency distribution of words(ngrams) and for Top 25 words excluding stopwords see Table \ref{tbl:train_words_sw} and Figure \ref{fig:model_2}


Using Pre-made stopwords

```r
length(stopwords(source = "smart"))
[1] 571
length(stopwords(source = "snowball"))
[1] 175
length(stopwords(source = "stopwords-iso"))
[1] 1298
```


Frequency distribution of words with stopwords removed

We will use the "stopwords-iso" Pre-made stopwords along with a few unique to our case

```r
mystopwords <- c("s", "t", "m", "ve", "re", "d", "ll")
```

\begin{table}

\caption{\label{tab:freq_dist_ngrams_stopwords}Frequency distribution of words excluding stopwords\label{tbl:train_words_sw}}
\centering
\begin{tabular}[t]{lrrrr}
\toprule
word & n & total & rank & term frequency\\
\midrule
book & 1441251 & 71253939 & 1 & 0.0202270\\
read & 513081 & 71253939 & 2 & 0.0072007\\
time & 506943 & 71253939 & 3 & 0.0071146\\
movie & 431317 & 71253939 & 4 & 0.0060532\\
love & 332012 & 71253939 & 5 & 0.0046596\\
product & 306813 & 71253939 & 6 & 0.0043059\\
bought & 292201 & 71253939 & 7 & 0.0041008\\
album & 265835 & 71253939 & 8 & 0.0037308\\
story & 264836 & 71253939 & 9 & 0.0037168\\
music & 235009 & 71253939 & 10 & 0.0032982\\
\bottomrule
\end{tabular}
\end{table}



![Frequency distribution of words(nrams) for Top 25 words excluding stopwords\label{fig:model_2}](figures/plot_freq_dist_ngrams_stopwords-1.pdf) 


\newpage
Here are Google's recommendations after decades of research:

Algorithm for Data Preparation and Model Building

1. Calculate the number of samples/number of words per sample ratio.
2. If this ratio is less than 1500, tokenize the text as n-grams and use a
simple multi-layer perceptron (MLP) model to classify them (left branch in the
flowchart below):
  a. Split the samples into word n-grams; convert the n-grams into vectors.
  b. Score the importance of the vectors and then select the top 20K using the scores.
  c. Build an MLP model.
3. If the ratio is greater than 1500, tokenize the text as sequences and use a
   [sepCNN](https://developers.google.com/machine-learning/glossary?utm_source=DevSite&utm_campaign=Text-Class-Guide&utm_medium=referral&utm_content=glossary&utm_term=sepCNN#depthwise-separable-convolutional-neural-network-sepcnn) model to classify them (right branch in the flowchart below):
  a. Split the samples into words; select the top 20K words based on their frequency.
  b. Convert the samples into word sequence vectors.
  c. If the original number of samples/number of words per sample ratio is less
     than 15K, using a fine-tuned pre-trained embedding with the sepCNN
     model will likely provide the best results.
4. Measure the model performance with different hyperparameter values to find
   the best model configuration for the dataset.


```r
# 3. If the ratio is greater than 1500, tokenize the text as sequences and use
# a sepCNN model see above

(S_W_ratio <- num_samples/median_num_words_per_sample)
[1] 42984.48
```


\newpage
## Preprocessing for deep learning continued with more exploration

For "Number of words per review text" see Figure \ref{fig:model_3}  

For "Number of words per review title" see Figure \ref{fig:model_4}  

For "Number of words per review text by label" see Figure \ref{fig:model_5}  

For "Number of words per review title by label" see Figure \ref{fig:model_6}  

For "Sample/Subset of our training dataset" see Table \ref{tbl:amazon_subset_train}  


![Number of words per review text\label{fig:model_3}](figures/preproc_1-1.pdf) 

![Number of words per review title\label{fig:model_4}](figures/preproc_2-1.pdf) 

![Number of words per review text by label\label{fig:model_5}](figures/preproc_3-1.pdf) 

![Number of words per review title by label\label{fig:model_6}](figures/preproc_4-1.pdf) 


Let's trim down our training dataset due to computing resource limitations.

```r
amazon_subset_train <- amazon_train %>%
    select(-title) %>%
    mutate(n_words = tokenizers::count_words(text)) %>%
    filter((n_words < 35) & (n_words > 5)) %>%
    select(-n_words)
dim(amazon_subset_train)
[1] 579545      2
# head(amazon_subset_train)
```

\begin{table}

\caption{\label{tab:subset_train}Sample/Subset of our training dataset\label{tbl:amazon_subset_train}}
\centering
\begin{tabular}[t]{rl}
\toprule
label & text\\
\midrule
1 & dmst are my favourite band and have been for a while now just mind blowing i wanted to make one correction to the previous review though the do makes are from toronto not quebec\\
1 & The best kept secret of science is how strongly it points towards a creator and dovetails with Christianity In this marvelously lucid book the eminent physical chemist Henry Schaefer unfolds the secret\\
1 & Our children ( age ) love these DVD s They are somewhat educational and seem to be much better than the cartoons you find on regular TV\\
1 & I have enjoyed using these picks I am a beginning guitar player and using a thin gauge pick like this makes strumming much easier\\
1 & This book is very concise and useful I found it very easy to accurately translate quickly\\
1 & Please don t deny your selfof a most irresistable chocolate This biography Am confident to sayForget what s current now Just Get this caviar of a biographyand drool on with pleasures un expected\\
0 & These are not clear Not even close They are opaque (and even closer to white) and not advertised as such To me they were completely useless Buyer beware\\
1 & Comfortable and classy but they scratch really easy Other than that a good buy if you don t plan on wearing them everyday\\
0 & I read about of the book then finally dropped it because I found it rather tiring (I had read Five Children and It before and thought I might like more book of the author\\
0 & I bought to look up the rules of Euchre and it had hardly anything here I thought the book sucked\\
\bottomrule
\end{tabular}
\end{table}



\newpage
# Model Baseline linear classifier

This model serves the purpose of comparison with the deep learning techniques we will implement later on, and also as a succinct summary of a basic supervised machine learning analysis for text.

This linear baseline is a regularized linear model trained on the same data set, using tf-idf weights and 5000 tokens.

## Modify label column to factor


```r
# Free computer resources
rm(amazon_train, amazon_val, amazon_train_text_wordCount, num_samples_per_class,
    temp, total_words, train_words)
rm(mean_num_words_per_sample, median_num_words_per_sample, num_classes, num_samples,
    S_W_ratio)
gc()
           used  (Mb) gc trigger   (Mb)   max used    (Mb)
Ncells  3372585 180.2   16004112  854.8   20005140  1068.4
Vcells 24396993 186.2 1240182133 9461.9 1550227666 11827.3

# save(amazon_subset_train)
write_csv(amazon_subset_train, "amazon_review_polarity_csv/amazon_subset_train.csv",
    col_names = TRUE)

amazon_train <- amazon_subset_train

amazon_train <- amazon_train %>%
    mutate(label = as.factor(label))

# amazon_val <- amazon_train %>% mutate(label = as.factor(label))
```


## Split into test/train and create resampling folds


```r
set.seed(1234)
amazon_split <- amazon_train %>%
    initial_split()
amazon_train <- training(amazon_split)
amazon_test <- testing(amazon_split)
set.seed(123)
amazon_folds <- vfold_cv(amazon_train)
# amazon_folds
```


## Recipe for data preprocessing

"step_tfidf" creates a specification of a recipe step that will convert a tokenlist into multiple variables containing the [term frequency-inverse document frequency](https://www.tidytextmining.com/tfidf.html) of tokens.(check it out in the console by typing ?textrecipes::step_tfidf)  


```r
# library(textrecipes)

amazon_rec <- recipe(label ~ text, data = amazon_train) %>%
    step_tokenize(text) %>%
    step_tokenfilter(text, max_tokens = 5000) %>%
    step_tfidf(text)

amazon_rec
Data Recipe

Inputs:

      role #variables
   outcome          1
 predictor          1

Operations:

Tokenization for text
Text filtering for text
Term frequency-inverse document frequency with text
```

## Lasso regularized classification model and tuning

Linear models are not considered cutting edge in NLP research, but are a workhorse in real-world practice. Here we will use a lasso regularized model [Tibshirani, Robert. 1996. "Regression Shrinkage and Selection via the Lasso." Journal of the Royal Statistical Society. Series B (Methodological) 58 (1). Royal Statistical Society, Wiley: 267–288.]( http://www.jstor.org/stable/2346178).

Let’s create a specification of lasso regularized model.

"penalty" is a model hyperparameter and we cannot learn its best value during model training, but we can estimate the best value by training many models on resampled data sets and exploring how well all these models perform. Let’s build a new model specification for model tuning.


```r
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
    set_mode("classification") %>%
    set_engine("glmnet")

lasso_spec
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = tune()
  mixture = 1

Computational engine: glmnet 
```

## A model workflow

We need a few more components before we can tune our workflow. Let's use
a sparse data encoding.

We can change how our text data is represented to take advantage of its sparsity, especially for models like lasso regularized models. The regularized regression model we trained above used set_engine("glmnet"); this computational engine can be more efficient when text data is transformed to a sparse matrix, rather than a dense data frame or tibble representation.

To keep our text data sparse throughout modeling and use the sparse capabilities of set_engine("glmnet"), we need to explicitly set a non-default preprocessing blueprint, using the package hardhat [Vaughan, Davis, and Max Kuhn. 2020. hardhat: Construct Modeling Packages.](https://CRAN.R-project.org/package=hardhat).


```r
library(hardhat)
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")
```

Let's create a grid of possible regularization penalties to try, using a convenience function for penalty() called grid_regular() from the dials package.


```r
lambda_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 20)
lambda_grid
```



\begin{tabular}{r}
\hline
penalty\\
\hline
0.0000100\\
\hline
0.0000183\\
\hline
0.0000336\\
\hline
0.0000616\\
\hline
0.0001129\\
\hline
0.0002069\\
\hline
0.0003793\\
\hline
0.0006952\\
\hline
0.0012743\\
\hline
0.0023357\\
\hline
0.0042813\\
\hline
0.0078476\\
\hline
0.0143845\\
\hline
0.0263665\\
\hline
0.0483293\\
\hline
0.0885867\\
\hline
0.1623777\\
\hline
0.2976351\\
\hline
0.5455595\\
\hline
1.0000000\\
\hline
\end{tabular}

Now these can be combined in a tuneable workflow()


```r
amazon_wf <- workflow() %>%
    add_recipe(amazon_rec, blueprint = sparse_bp) %>%
    add_model(lasso_spec)

amazon_wf
== Workflow ====================================================================
Preprocessor: Recipe
Model: logistic_reg()

-- Preprocessor ----------------------------------------------------------------
3 Recipe Steps

* step_tokenize()
* step_tokenfilter()
* step_tfidf()

-- Model -----------------------------------------------------------------------
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = tune()
  mixture = 1

Computational engine: glmnet 
```


\newpage
## Tune the workflow

Let’s use tune_grid() to fit a model at each of the values for the regularization penalty in our regular grid and every resample in amazon_folds.


```r
set.seed(2020)
lasso_rs <- tune_grid(amazon_wf, amazon_folds, grid = lambda_grid, control = control_resamples(save_pred = TRUE))

# lasso_rs
```


We now have a set of metrics for each value of the regularization penalty.  

We can extract the relevant information using collect_metrics() and collect_predictions()

See Table \ref{tbl:lasso_metrics} for Lasso Metrics


```r
m_lm <- collect_metrics(lasso_rs)
kable(m_lm, format = "simple", caption = "Lasso Metrics\\label{tbl:lasso_metrics}")
```



Table: Lasso Metrics\label{tbl:lasso_metrics}

   penalty  .metric    .estimator         mean    n     std_err  .config               
----------  ---------  -----------  ----------  ---  ----------  ----------------------
 0.0000100  accuracy   binary        0.8893912   10   0.0005032  Preprocessor1_Model01 
 0.0000100  roc_auc    binary        0.9538016   10   0.0003006  Preprocessor1_Model01 
 0.0000183  accuracy   binary        0.8893912   10   0.0005032  Preprocessor1_Model02 
 0.0000183  roc_auc    binary        0.9538016   10   0.0003006  Preprocessor1_Model02 
 0.0000336  accuracy   binary        0.8893912   10   0.0004969  Preprocessor1_Model03 
 0.0000336  roc_auc    binary        0.9538247   10   0.0003008  Preprocessor1_Model03 
 0.0000616  accuracy   binary        0.8895062   10   0.0004782  Preprocessor1_Model04 
 0.0000616  roc_auc    binary        0.9539107   10   0.0003016  Preprocessor1_Model04 
 0.0001129  accuracy   binary        0.8896719   10   0.0004941  Preprocessor1_Model05 
 0.0001129  roc_auc    binary        0.9540329   10   0.0003014  Preprocessor1_Model05 
 0.0002069  accuracy   binary        0.8897064   10   0.0004424  Preprocessor1_Model06 
 0.0002069  roc_auc    binary        0.9541505   10   0.0003024  Preprocessor1_Model06 
 0.0003793  accuracy   binary        0.8894809   10   0.0004733  Preprocessor1_Model07 
 0.0003793  roc_auc    binary        0.9541045   10   0.0003048  Preprocessor1_Model07 
 0.0006952  accuracy   binary        0.8883996   10   0.0004462  Preprocessor1_Model08 
 0.0006952  roc_auc    binary        0.9534221   10   0.0003069  Preprocessor1_Model08 
 0.0012743  accuracy   binary        0.8853696   10   0.0005228  Preprocessor1_Model09 
 0.0012743  roc_auc    binary        0.9512859   10   0.0003173  Preprocessor1_Model09 
 0.0023357  accuracy   binary        0.8782767   10   0.0005656  Preprocessor1_Model10 
 0.0023357  roc_auc    binary        0.9465029   10   0.0003413  Preprocessor1_Model10 
 0.0042813  accuracy   binary        0.8654758   10   0.0005853  Preprocessor1_Model11 
 0.0042813  roc_auc    binary        0.9375896   10   0.0003789  Preprocessor1_Model11 
 0.0078476  accuracy   binary        0.8447308   10   0.0006268  Preprocessor1_Model12 
 0.0078476  roc_auc    binary        0.9225346   10   0.0003517  Preprocessor1_Model12 
 0.0143845  accuracy   binary        0.8114196   10   0.0008523  Preprocessor1_Model13 
 0.0143845  roc_auc    binary        0.8982350   10   0.0004307  Preprocessor1_Model13 
 0.0263665  accuracy   binary        0.7658711   10   0.0008243  Preprocessor1_Model14 
 0.0263665  roc_auc    binary        0.8620465   10   0.0005772  Preprocessor1_Model14 
 0.0483293  accuracy   binary        0.7075908   10   0.0008798  Preprocessor1_Model15 
 0.0483293  roc_auc    binary        0.8020162   10   0.0008407  Preprocessor1_Model15 
 0.0885867  accuracy   binary        0.6665424   10   0.0010263  Preprocessor1_Model16 
 0.0885867  roc_auc    binary        0.7273321   10   0.0006962  Preprocessor1_Model16 
 0.1623777  accuracy   binary        0.5796028   10   0.0007071  Preprocessor1_Model17 
 0.1623777  roc_auc    binary        0.5000000   10   0.0000000  Preprocessor1_Model17 
 0.2976351  accuracy   binary        0.5796028   10   0.0007071  Preprocessor1_Model18 
 0.2976351  roc_auc    binary        0.5000000   10   0.0000000  Preprocessor1_Model18 
 0.5455595  accuracy   binary        0.5796028   10   0.0007071  Preprocessor1_Model19 
 0.5455595  roc_auc    binary        0.5000000   10   0.0000000  Preprocessor1_Model19 
 1.0000000  accuracy   binary        0.5796028   10   0.0007071  Preprocessor1_Model20 
 1.0000000  roc_auc    binary        0.5000000   10   0.0000000  Preprocessor1_Model20 


What are the best models?  

See Table \ref{tbl:best_lasso_roc} for Best Lasso ROC.


```r
m_blr <- show_best(lasso_rs, "roc_auc")
kable(m_blr, format = "simple", caption = "Best Lasso ROC\\label{tbl:best_lasso_roc}")
```



Table: Best Lasso ROC\label{tbl:best_lasso_roc}

   penalty  .metric   .estimator         mean    n     std_err  .config               
----------  --------  -----------  ----------  ---  ----------  ----------------------
 0.0002069  roc_auc   binary        0.9541505   10   0.0003024  Preprocessor1_Model06 
 0.0003793  roc_auc   binary        0.9541045   10   0.0003048  Preprocessor1_Model07 
 0.0001129  roc_auc   binary        0.9540329   10   0.0003014  Preprocessor1_Model05 
 0.0000616  roc_auc   binary        0.9539107   10   0.0003016  Preprocessor1_Model04 
 0.0000336  roc_auc   binary        0.9538247   10   0.0003008  Preprocessor1_Model03 


See Table \ref{tbl:best_lasso_acc} for Best Lasso Accuracy.


```r
m_bla <- show_best(lasso_rs, "accuracy")
kable(m_bla, format = "simple", caption = "Best Lasso Accuracy\\label{tbl:best_lasso_acc}")
```



Table: Best Lasso Accuracy\label{tbl:best_lasso_acc}

   penalty  .metric    .estimator         mean    n     std_err  .config               
----------  ---------  -----------  ----------  ---  ----------  ----------------------
 0.0002069  accuracy   binary        0.8897064   10   0.0004424  Preprocessor1_Model06 
 0.0001129  accuracy   binary        0.8896719   10   0.0004941  Preprocessor1_Model05 
 0.0000616  accuracy   binary        0.8895062   10   0.0004782  Preprocessor1_Model04 
 0.0003793  accuracy   binary        0.8894809   10   0.0004733  Preprocessor1_Model07 
 0.0000100  accuracy   binary        0.8893912   10   0.0005032  Preprocessor1_Model01 


Let’s visualize these metrics; accuracy and ROC AUC, in Figure \ref{fig:model_7} to see what the best model is.

![Lasso model performance across regularization penalties\label{fig:model_7}](figures/plot_lasso-1.pdf) 


See Table \ref{tbl:lasso_predictions} for Lasso Predictions


```r
m_lp <- collect_predictions(lasso_rs)
kable(head(m_lp), format = "simple", caption = "Lasso Predictions\\label{tbl:lasso_predictions}")
```



Table: Lasso Predictions\label{tbl:lasso_predictions}

id          .pred_0     .pred_1   .row   penalty  .pred_class   label   .config               
-------  ----------  ----------  -----  --------  ------------  ------  ----------------------
Fold01    0.4909890   0.5090110     10     1e-05  1             0       Preprocessor1_Model01 
Fold01    0.0116122   0.9883878     28     1e-05  1             1       Preprocessor1_Model01 
Fold01    0.0145242   0.9854758     30     1e-05  1             1       Preprocessor1_Model01 
Fold01    0.0144831   0.9855169     68     1e-05  1             1       Preprocessor1_Model01 
Fold01    0.2691942   0.7308058     79     1e-05  1             0       Preprocessor1_Model01 
Fold01    0.9884782   0.0115218     86     1e-05  0             0       Preprocessor1_Model01 


Figure \ref{fig:model_8} shows the ROC curve, a visualization of how well a classification model can distinguish between classes

![Lasso model ROC Label 0\label{fig:model_8}](figures/m_lp_roc_0-1.pdf) 

Figure \ref{fig:model_9} shows the ROC curve, a visualization of how well a classification model can distinguish between classes

![Lasso model ROC Label 1\label{fig:model_9}](figures/m_lp_roc_1-1.pdf) 




\newpage
## Results  

We saw that regularized linear models, such as lasso, often work well for text data sets.  

The default performance parameters for binary classification are accuracy and ROC AUC (area under the receiver operator characteristic curve). Here, the best accuracy is:  

Best ROC_AUC is 0.9541505  
Best Accuracy is 0.8897064  

As we go along, we will be comparing different approaches. Let's start by creating a results table with this BLM to get Table \ref{tbl:blm_results_table}:


Table: Baseline Linear Model Results\label{tbl:blm_results_table}

Index   Method     Accuracy  Loss 
------  -------  ----------  -----
1       BLM       0.8897064  NA   


Accuracy and ROC AUC are performance metrics used for classification models. For both, values closer to 1 are better.

Accuracy is the proportion of the data that are predicted correctly. Be aware that accuracy can be misleading in some situations, such as for imbalanced data sets.

ROC AUC measures how well a classifier performs at different thresholds. The ROC curve plots the true positive rate against the false positive rate, and AUC closer to 1 indicates a better-performing model while AUC closer to 0.5 indicates a model that does no better than random guessing.

Figure \ref{fig:model_8} and Figure \ref{fig:model_9} show the ROC curves, a visualization of how well our classification model can distinguish between classes.

The area under each of these curves is the roc_auc metric we have computed. If the curve was close to the diagonal line, then the model’s predictions would be no better than random guessing.

One metric alone cannot give you a complete picture of how well your classification model is performing. The confusion matrix is a good starting point to get an overview of your model performance, as it includes rich information.

Another way to evaluate our model is to evaluate the confusion matrix. A confusion matrix tabulates a model’s false positives and false negatives for each class. The function conf_mat_resampled() computes a separate confusion matrix for each resample and takes the average of the cell counts. This allows us to visualize an overall confusion matrix rather than needing to examine each resample individually.  



# Preprocessing for rest of the models

Preprocessing for deep learning models is different than preprocessing for most other text models. These neural networks model sequences, so we have to choose the length of sequences we would like to include. Sequences that are longer than this length are truncated (information is thrown away) and those that are shorter than this length are padded with zeroes (an empty, non-informative value) to get to the chosen sequence length. This sequence length is a hyperparameter of the model and we need to select this value such that we don’t overshoot and introduce a lot of padded zeroes which would make the model hard to train, or undershoot and cut off too much informative text.

We will use the recipes and textrecipes packages for data preprocessing and feature engineering.

The formula used to specify this recipe ~ text does not have an outcome, because we are using recipes and textrecipes functions on their own, outside of the rest of the tidymodels framework; we don’t need to know about the outcome here. This preprocessing recipe tokenizes our text and filters to keep only the top 20,000 words and then it transforms the tokenized text into a numeric format appropriate for modeling, using step_sequence_onehot().


```r
rm(amazon_folds, amazon_rec, amazon_split, amazon_test, amazon_train, amazon_wf,
    lambda_grid, lasso_rs, lasso_spec, sparse_bp)

gc()
            used  (Mb) gc trigger   (Mb)   max used    (Mb)
Ncells   4147050 221.5   12861945  687.0   20005140  1068.4
Vcells 112719032 860.0  867481767 6618.4 4028113514 30732.1

amazon_subset_train <- readr::read_csv("amazon_review_polarity_csv/amazon_subset_train.csv")

amazon_train <- amazon_subset_train

max_words <- 20000
max_length <- 30
mystopwords <- c("s", "t", "m", "ve", "re", "d", "ll")

amazon_rec <- recipe(~text, data = amazon_subset_train) %>%
    step_text_normalization(text) %>%
    step_tokenize(text) %>%
    step_stopwords(text, stopword_source = "stopwords-iso", custom_stopword_source = mystopwords) %>%
    step_tokenfilter(text, max_tokens = max_words) %>%
    step_sequence_onehot(text, sequence_length = max_length)

amazon_rec
Data Recipe

Inputs:

      role #variables
 predictor          1

Operations:

text_normalizationming for text
Tokenization for text
Stop word removal for text
Text filtering for text
Sequence 1 hot encoding for text
```


The prep() function will compute or estimate statistics from the training set; the output of prep() is a prepped recipe.

When we bake() a prepped recipe, we apply the preprocessing to the data set. We can get out the training set that we started with by specifying new_data = NULL or apply it to another set via new_data = my_other_data_set. The output of bake() is a data set like a tibble or a matrix, depending on the composition argument.

Let’s now prepare and apply our feature engineering recipe amazon_rec so we can use it in our deep learning model.


```r
amazon_prep <- prep(amazon_rec)

amazon_subset_train <- bake(amazon_prep, new_data = NULL, composition = "matrix")
dim(amazon_subset_train)
[1] 579545     30
```

The prep() function will compute or estimate statistics from the training set; the output of prep() is a prepped recipe.
The prepped recipe can be tidied using tidy() to extract the vocabulary, represented in the vocabulary and token columns.


```r
amazon_prep %>%
    tidy(5) %>%
    head(10)
```



\begin{tabular}{l|r|l|l}
\hline
terms & vocabulary & token & id\\
\hline
text & 1 & a & sequence\_onehot\_bCTRZ\\
\hline
text & 2 & à & sequence\_onehot\_bCTRZ\\
\hline
text & 3 & aa & sequence\_onehot\_bCTRZ\\
\hline
text & 4 & aaa & sequence\_onehot\_bCTRZ\\
\hline
text & 5 & aaaa & sequence\_onehot\_bCTRZ\\
\hline
text & 6 & aaliyah & sequence\_onehot\_bCTRZ\\
\hline
text & 7 & aaron & sequence\_onehot\_bCTRZ\\
\hline
text & 8 & ab & sequence\_onehot\_bCTRZ\\
\hline
text & 9 & abandon & sequence\_onehot\_bCTRZ\\
\hline
text & 10 & abandoned & sequence\_onehot\_bCTRZ\\
\hline
\end{tabular}



\newpage
# Model DNN

A densely connected neural network is one of the simplest configurations for a deep learning model and is typically not a model that will achieve the highest performance on text data, but it is a good place to start to understand the process of building and evaluating deep learning models for text.  

In a densely connected neural network, layers are fully connected (dense) by the neurons in a network layer. Each neuron in a layer receives an input from all the neurons present in the previous layer - thus, they’re densely connected.  

The input comes in to the network all at once and is densely (in this case, fully) connected to the first hidden layer. A layer is “hidden” in the sense that it doesn’t connect to the outside world; the input and output layers take care of this. The neurons in any given layer are only connected to the next layer. The numbers of layers and nodes within each layer are variable and are hyperparameters of the model selected by us.

## A Simple flattened dense neural network

Our first deep learning model embeds the Amazon Reviews in sequences of vectors, flattens them, and then trains a dense network layer to predict whether the Review was positive(1) or not(0).

1. We initiate the Keras model as a linear stack of layers with keras_model_sequential().  

2. Our first layer - layer_embedding() turns each observation into an (embedding_dim * sequence_length) = 12 * 30  

3. In total, we will create a (number_of_observations * embedding_dim * sequence_length) data cube.  

4. The next layer_flatten() layer takes the matrix for each observation and flattens them down into one dimension. This will create a (30 * 12) = 360 long vector for each observation.  

5. layer_layer_normalization() - Normalize the activations of the previous layer for each given example in a batch independently.  

6. Lastly, we have 2 densely connected layers. The last layer has a sigmoid activation function to give us an output between 0 and 1, since we want to model a probability for a binary classification problem.


```r
# library(keras) use_python(python =
# '/c/Users/bijoor/.conda/envs/tensorflow-python/python.exe', required = TRUE)
# use_condaenv(condaenv = 'tensorflow-python', required = TRUE)

dense_model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words + 1, output_dim = 12, input_length = max_length) %>%
    layer_flatten() %>%
    layer_layer_normalization() %>%
    # layer_dropout(0.1) %>%
layer_dense(units = 64) %>%
    # layer_activation_leaky_relu() %>%
layer_activation_relu() %>%
    layer_dense(units = 1, activation = "sigmoid")

dense_model
Model
Model: "sequential"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
embedding (Embedding)               (None, 30, 12)                  240012      
________________________________________________________________________________
flatten (Flatten)                   (None, 360)                     0           
________________________________________________________________________________
layer_normalization (LayerNormaliza (None, 360)                     720         
________________________________________________________________________________
dense_1 (Dense)                     (None, 64)                      23104       
________________________________________________________________________________
re_lu (ReLU)                        (None, 64)                      0           
________________________________________________________________________________
dense (Dense)                       (None, 1)                       65          
================================================================================
Total params: 263,901
Trainable params: 263,901
Non-trainable params: 0
________________________________________________________________________________
```


Before we can fit this model to the data it requires an optimizer and a loss function to be able to compile.

When the neural network finishes passing a batch of data through the network, it needs a way to use the difference between the predicted values and true values to update the network’s weights. The algorithm that determines those weights is known as the optimization algorithm. Many optimizers are available within Keras.

We will choose one of the following based on our previous experimentation.  

optimizer_adam() - Adam - A Method for Stochastic Optimization  

optimizer_sgd() - Stochastic gradient descent optimizer  

We can also use various options during training using the compile() function, such as optimizer, loss and metrics.



```r
# opt <- optimizer_adam(lr = 0.0001, decay = 1e-6) opt <- optimizer_sgd(lr =
# 0.001, decay = 1e-6)
opt <- optimizer_sgd()
dense_model %>%
    compile(optimizer = opt, loss = "binary_crossentropy", metrics = c("accuracy"))
```


Finally, we can fit this model.  

Here we specify the Keras defaults for creating a validation split and tracking metrics with an internal validation split of 20%.


```r
dense_history <- dense_model %>%
    fit(x = amazon_subset_train, y = amazon_train$label, batch_size = 1024, epochs = 50,
        initial_epoch = 0, validation_split = 0.2, verbose = 2)

dense_history

Final epoch (plot to see history):
        loss: 0.2457
    accuracy: 0.8994
    val_loss: 0.2688
val_accuracy: 0.8898 
```


"DNN Model 1 Fit History using validation_split" Figure \ref{fig:model_10}

![DNN Model 1 Fit History using validation_split\label{fig:model_10}](figures/dense_model_1_split_hist-1.pdf) 

\newpage
## Evaluation

Instead of using Keras defaults, we can use tidymodels functions to be more specific about these model characteristics. Instead of using the validation_split argument to fit(), we can create our own validation set using tidymodels and use validation_data argument for fit(). We create our validation split from the training set.


```r
set.seed(234)
amazon_val_eval <- validation_split(amazon_train, strata = label)
# amazon_val_eval << I am getting a pandoc stack error printing this
```


The split object contains the information necessary to extract the data we will use for training/analysis and the data we will use for validation/assessment. We can extract these data sets in their raw, unprocessed form from the split using the helper functions analysis() and assessment(). Then, we can apply our prepped preprocessing recipe amazon_prep to both to transform this data to the appropriate format for our neural network architecture.


```r
amazon_analysis <- bake(amazon_prep, new_data = analysis(amazon_val_eval$splits[[1]]),
    composition = "matrix")
dim(amazon_analysis)
[1] 434658     30
```


```r
amazon_assess <- bake(amazon_prep, new_data = assessment(amazon_val_eval$splits[[1]]),
    composition = "matrix")
dim(amazon_assess)
[1] 144887     30
```


Here we get outcome variables for both sets.


```r
label_analysis <- analysis(amazon_val_eval$splits[[1]]) %>%
    pull(label)
label_assess <- assessment(amazon_val_eval$splits[[1]]) %>%
    pull(label)
```


Let's setup a new DNN model 2.

Here we use layer_dropout() - Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.


```r
dense_model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words + 1, output_dim = 12, input_length = max_length) %>%
    layer_flatten() %>%
    layer_layer_normalization() %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 64) %>%
    layer_activation_relu() %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 128) %>%
    layer_activation_relu() %>%
    layer_dense(units = 128) %>%
    layer_activation_relu() %>%
    layer_dense(units = 1, activation = "sigmoid")

opt <- optimizer_adam(lr = 1e-04, decay = 1e-06)
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6) opt <- optimizer_sgd()
dense_model %>%
    compile(optimizer = opt, loss = "binary_crossentropy", metrics = c("accuracy"))
```


We now fit this model to validation_data - amazon_assess and label_assess instead of the Keras default validation_split.


```r
val_history <- dense_model %>%
    fit(x = amazon_analysis, y = label_analysis, batch_size = 2048, epochs = 20,
        validation_data = list(amazon_assess, label_assess), verbose = 2)

val_history

Final epoch (plot to see history):
        loss: 0.2638
    accuracy: 0.8932
    val_loss: 0.2837
val_accuracy: 0.8961 
```


"DNN Model 2 Fit History using validation_data" Figure \ref{fig:model_11}  

![DNN Model 2 Fit History using validation_data\label{fig:model_11}](figures/dense_model_2_fit_val_d_hist-1.pdf) 



Using our own validation set also allows us to flexibly measure
performance using tidymodels functions.

The following function keras_predict() creates prediction results using the Keras model and preprocessed/baked data from using tidymodels, using a 50% probability threshold and works for our binary problem.


```r
keras_predict <- function(model, baked_data, response) {
    predictions <- predict(model, baked_data)[, 1]
    tibble(.pred_1 = predictions, .pred_class = if_else(.pred_1 < 0.5, 0, 1), label = response) %>%
        mutate(across(c(label, .pred_class), ~factor(.x, levels = c(1, 0))))
}
```


See Table \ref{tbl:val_res} for "DNN Model 2 Predictions using validation_data"  


```r
val_res <- keras_predict(dense_model, amazon_assess, label_assess)
# head(val_res)
kable(head(val_res), format = "simple", caption = "DNN Model 2 Predictions using validation data\\label{tbl:val_res}")
```



Table: DNN Model 2 Predictions using validation data\label{tbl:val_res}

   .pred_1  .pred_class   label 
----------  ------------  ------
 0.0070337  0             0     
 0.8914716  1             1     
 0.9580745  1             1     
 0.3149569  0             0     
 0.7441973  1             1     
 0.9910595  1             1     



See Table \ref{tbl:val_res_metrics} for "DNN Model 2 Metrics using Validation data"


```r
m1 <- metrics(val_res, label, .pred_class)
kable(m1, format = "simple", caption = "DNN Model 2 Metrics using Validation data\\label{tbl:val_res_metrics}")
```



Table: DNN Model 2 Metrics using Validation data\label{tbl:val_res_metrics}

.metric    .estimator    .estimate
---------  -----------  ----------
accuracy   binary        0.8960983
kap        binary        0.7870405


"DNN Model 2 Confusion Matrix using Validation data" Figure \ref{fig:model_12}  

![DNN Model 2 Confusion Matrix using Validation data\label{fig:model_12}](figures/val_res_conf_mat-1.pdf) 


"DNN Model 2 ROC curve using Validation data" Figure \ref{fig:model_13}   

![DNN Model 2 ROC curve using Validation data\label{fig:model_13}](figures/val_res_roc-1.pdf) 



\newpage
## Results  

DNN model results Table \ref{tbl:dnn_results_table}:


Table: DNN Model Results\label{tbl:dnn_results_table}

Index   Method     Accuracy        Loss
------  -------  ----------  ----------
1       BLM       0.8897064          NA
2       DNN       0.8960983   0.2790776


\newpage
# Model CNN

A CNN is a neural network in which at least one layer is a convolutional layer. A typical convolutional neural network consists of some combination of the following layers:

1.Convolutional layers - A layer of a deep neural network in which a convolutional filter passes along an input matrix.  
A convolutional operation involves a convolutional filter which is a matrix having the same rank as the input matrix, but a smaller shape and a slice of an input matrix. For example, given a 28x28 input matrix, the filter could be any 2D matrix smaller than 28x28.

For example, in photographic manipulation, all the cells in a convolutional filter are typically set to a constant pattern of ones and zeroes. In machine learning, convolutional filters are typically seeded with random numbers and then the network trains the ideal values.  

2.Pooling layers - Reducing a matrix (or matrices) created by an earlier convolutional layer to a smaller matrix. Pooling usually involves taking either the maximum or average value across the pooled area. For example, suppose we have a 3x3 matrix. A pooling operation, just like a convolutional operation, divides that matrix into slices and then slides that convolutional operation by strides. For example, suppose the pooling operation divides the convolutional matrix into 2x2 slices with a 1x1 stride, then four pooling operations take place. Imagine that each pooling operation picks the maximum value of the four in that slice.  

Pooling helps enforce translational invariance in the input matrix.  

Pooling for vision applications is known more formally as spatial pooling. Time-series applications usually refer to pooling as temporal pooling. Less formally, pooling is often called subsampling or downsampling.  

3.Dense layers - just a fully connected layer.  

Convolutional neural networks have had great success in certain kinds of problems, especially in image recognition.

## A first CNN model


```r
simple_cnn_model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words + 1, output_dim = 16, input_length = max_length) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filter = 64, kernel_size = 3, activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

simple_cnn_model
Model
Model: "sequential_2"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
embedding_2 (Embedding)             (None, 30, 16)                  320016      
________________________________________________________________________________
batch_normalization (BatchNormaliza (None, 30, 16)                  64          
________________________________________________________________________________
conv1d_1 (Conv1D)                   (None, 26, 32)                  2592        
________________________________________________________________________________
max_pooling1d (MaxPooling1D)        (None, 13, 32)                  0           
________________________________________________________________________________
conv1d (Conv1D)                     (None, 11, 64)                  6208        
________________________________________________________________________________
global_max_pooling1d (GlobalMaxPool (None, 64)                      0           
________________________________________________________________________________
dense_7 (Dense)                     (None, 64)                      4160        
________________________________________________________________________________
dense_6 (Dense)                     (None, 1)                       65          
================================================================================
Total params: 333,105
Trainable params: 333,073
Non-trainable params: 32
________________________________________________________________________________
```




```r
simple_cnn_model %>%
    compile(optimizer = opt, loss = "binary_crossentropy", metrics = c("accuracy"))
```




```r
simple_cnn_val_history <- simple_cnn_model %>%
    fit(x = amazon_analysis, y = label_analysis, batch_size = 1024, epochs = 7, initial_epoch = 0,
        validation_data = list(amazon_assess, label_assess), verbose = 2)

simple_cnn_val_history

Final epoch (plot to see history):
        loss: 0.2097
    accuracy: 0.9192
    val_loss: 0.2526
val_accuracy: 0.8983 
```


"CNN Model Fit History using validation_data" Figure \ref{fig:model_14}  

![CNN Model Fit History using validation_data\label{fig:model_14}](figures/simple_cnn_model_fit_val_d_hist-1.pdf) 


See Table \ref{tbl:simple_cnn_val_res} for "CNN Model Predictions using validation data"


```r
simple_cnn_val_res <- keras_predict(simple_cnn_model, amazon_assess, label_assess)
# head(simple_cnn_val_res)
kable(head(simple_cnn_val_res), format = "simple", caption = "CNN Model Predictions using validation data\\label{tbl:simple_cnn_val_res}")
```



Table: CNN Model Predictions using validation data\label{tbl:simple_cnn_val_res}

   .pred_1  .pred_class   label 
----------  ------------  ------
 0.0036336  0             0     
 0.9667417  1             1     
 0.9940560  1             1     
 0.1679935  0             0     
 0.8921552  1             1     
 0.9995990  1             1     


See Table \ref{tbl:simple_cnn_val_res_metrics} for "CNN Model Metrics using validation data"


```r
m2 <- metrics(simple_cnn_val_res, label, .pred_class)
kable(m2, format = "simple", caption = "CNN Model Metrics using validation data\\label{tbl:simple_cnn_val_res_metrics}")
```



Table: CNN Model Metrics using validation data\label{tbl:simple_cnn_val_res_metrics}

.metric    .estimator    .estimate
---------  -----------  ----------
accuracy   binary        0.8982586
kap        binary        0.7911622


"CNN Model Confusion Matrix using validation_data" Figure \ref{fig:model_15}

![CNN Model Confusion Matrix using validation_data\label{fig:model_15}](figures/simple_cnn_val_res_conf_mat-1.pdf) 


"CNN Model ROC curve using validation_data" Figure \ref{fig:model_16}

![CNN Model ROC curve using validation_data\label{fig:model_16}](figures/simple_cnn_val_res_roc-1.pdf) 





\newpage
## Results  

CNN model results Table \ref{tbl:cnn_results_table}:


Table: CNN Model Results\label{tbl:cnn_results_table}

Index   Method     Accuracy        Loss
------  -------  ----------  ----------
1       BLM       0.8897064          NA
2       DNN       0.8960983   0.2790776
3       CNN       0.8982586   0.2526298


\newpage
# Model sepCNN

A convolutional neural network architecture based on [Inception](https://github.com/tensorflow/tpu/tree/master/models/experimental/inception), but where Inception modules are replaced with depthwise separable convolutions. Also known as Xception.

A depthwise separable convolution (also abbreviated as separable convolution) factors a standard 3-D convolution into two separate convolution operations that are more computationally efficient: first, a depthwise convolution, with a depth of 1 (n x n x 1), and then second, a pointwise convolution, with length and width of 1 (1 x 1 x n).

To learn more, see [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf).

## A first sepCNN model


```r
sep_cnn_model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words + 1, output_dim = 16, input_length = max_length) %>%
    # layer_batch_normalization() %>%
layer_dropout(0.2) %>%
    layer_separable_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
    layer_separable_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_separable_conv_1d(filter = 64, kernel_size = 5, activation = "relu") %>%
    layer_separable_conv_1d(filter = 64, kernel_size = 5, activation = "relu") %>%
    layer_global_average_pooling_1d() %>%
    layer_dropout(0.2) %>%
    # layer_dense(units = 64, activation = 'relu') %>%
layer_dense(units = 1, activation = "sigmoid")

sep_cnn_model
Model
Model: "sequential_3"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
embedding_3 (Embedding)             (None, 30, 16)                  320016      
________________________________________________________________________________
dropout_3 (Dropout)                 (None, 30, 16)                  0           
________________________________________________________________________________
separable_conv1d_3 (SeparableConv1D (None, 26, 32)                  624         
________________________________________________________________________________
separable_conv1d_2 (SeparableConv1D (None, 22, 32)                  1216        
________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)      (None, 11, 32)                  0           
________________________________________________________________________________
separable_conv1d_1 (SeparableConv1D (None, 7, 64)                   2272        
________________________________________________________________________________
separable_conv1d (SeparableConv1D)  (None, 3, 64)                   4480        
________________________________________________________________________________
global_average_pooling1d (GlobalAve (None, 64)                      0           
________________________________________________________________________________
dropout_2 (Dropout)                 (None, 64)                      0           
________________________________________________________________________________
dense_8 (Dense)                     (None, 1)                       65          
================================================================================
Total params: 328,673
Trainable params: 328,673
Non-trainable params: 0
________________________________________________________________________________
```




```r
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6) opt <- optimizer_adam() opt <-
# optimizer_sgd()
opt <- optimizer_adam(lr = 1e-04, decay = 1e-06)
sep_cnn_model %>%
    compile(optimizer = opt, loss = "binary_crossentropy", metrics = c("accuracy"))
```


Here we use a [callback](https://tensorflow.rstudio.com/guide/keras/guide_keras/#sts=Callbacks).

keras::callback_early_stopping: Interrupt training when validation performance has stopped improving.  
patience: number of epochs with no improvement after which training will be stopped.

Try ??keras::callback_early_stopping


```r
sep_cnn_val_history <- sep_cnn_model %>%
  fit(
    x = amazon_analysis,
    y = label_analysis,
    batch_size = 128,
    epochs = 20,
    initial_epoch = 0,
    validation_data = list(amazon_assess, label_assess),
    callbacks = list(callback_early_stopping(
        monitor='val_loss', patience=2)),
    verbose = 2
  )

sep_cnn_val_history
```

```
## 
## Final epoch (plot to see history):
##         loss: 0.1894
##     accuracy: 0.9279
##     val_loss: 0.2417
## val_accuracy: 0.9052
```


"sepCNN Model Fit History using validation_data" Figure \ref{fig:model_17}

![sepCNN Model Fit History using validation_data\label{fig:model_17}](figures/sep_cnn_model_fit_val_d_hist-1.pdf) 


See Table \ref{tbl:sep_cnn_val_res} for "sepCNN Model Predictions using validation data"


```r
sep_cnn_val_res <- keras_predict(sep_cnn_model, amazon_assess, label_assess)
# head(sep_cnn_val_res)
kable(head(sep_cnn_val_res), format="simple", caption="sepCNN Model Predictions using validation data\\label{tbl:sep_cnn_val_res}")
```



Table: sepCNN Model Predictions using validation data\label{tbl:sep_cnn_val_res}

   .pred_1  .pred_class   label 
----------  ------------  ------
 0.0000648  0             0     
 0.9813509  1             1     
 0.9909515  1             1     
 0.2639050  0             0     
 0.8420113  1             1     
 0.9995446  1             1     


See Table \ref{tbl:sep_cnn_val_res_metrics} for "sepCNN Model Metrics using validation data"


```r
m3 <- metrics(sep_cnn_val_res, label, .pred_class)
kable(m3, format="simple", caption="sepCNN Model Metrics using validation data\\label{tbl:sep_cnn_val_res_metrics}")
```



Table: sepCNN Model Metrics using validation data\label{tbl:sep_cnn_val_res_metrics}

.metric    .estimator    .estimate
---------  -----------  ----------
accuracy   binary        0.9051675
kap        binary        0.8045877


"sepCNN Model Confusion Matrix using validation_data" Figure \ref{fig:model_18}

![sepCNN Model Confusion Matrix using validation_data\label{fig:model_18}](figures/sep_cnn_val_res_conf_mat-1.pdf) 


"sepCNN Model ROC curve using validation_data" Figure \ref{fig:model_19}

![sepCNN Model ROC curve using validation_data\label{fig:model_19}](figures/sep_cnn_val_res_roc-1.pdf) 





\newpage
## Results  

sepCNN model results Table \ref{tbl:sep_cnn_results_table}:


Table: sepCNN Model Results\label{tbl:sep_cnn_results_table}

Index   Method     Accuracy        Loss
------  -------  ----------  ----------
1       BLM       0.8897064          NA
2       DNN       0.8960983   0.2790776
3       CNN       0.8982586   0.2526298
4       sepCNN    0.9051675   0.2408866


\newpage
# Model BERT

## About BERT  

In this model we will fine-tune BERT to perform sentiment analysis on our dataset.  

[BERT](https://arxiv.org/abs/1810.04805) and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models. The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.  

BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.


## References

[Tensorflow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.  

The [TensorFlow Hub](https://tfhub.dev/) lets you search and discover hundreds of trained, ready-to-deploy machine learning models in one place.

[Tensorflow for R](https://tensorflow.rstudio.com/) provides an R interface for Tensorflow.

## Loading CSV data

Load CSV data from a file into a TensorFlow Dataset using tfdatasets.

## Setup


```r
library(keras)
library(tfdatasets)
library(reticulate)
library(tidyverse)
library(lubridate)
library(tfhub)

# A dependency of the preprocessing for BERT inputs pip install -q -U
# tensorflow-text
import("tensorflow_text")
Module(tensorflow_text)

# You will use the AdamW optimizer from tensorflow/models.  pip install -q
# tf-models-official to create AdamW optimizer
o_nlp <- import("official.nlp")

Sys.setenv(TFHUB_CACHE_DIR = "C:/Users/bijoor/.cache/tfhub_modules")
Sys.getenv("TFHUB_CACHE_DIR")
[1] "C:/Users/bijoor/.cache/tfhub_modules"
```

You could load this using read.csv, and pass the arrays to TensorFlow.
If you need to scale up to a large set of files, or need a loader that
integrates with TensorFlow and tfdatasets then use the make_csv_dataset
function:

Now read the CSV data from the file and create a dataset.

## Make datasets 


```r
train_file_path <- file.path("amazon_review_polarity_csv/amazon_train.csv")

batch_size <- 32

train_dataset <- make_csv_dataset(train_file_path, field_delim = ",", batch_size = batch_size,
    column_names = list("label", "title", "text"), label_name = "label", select_columns = list("label",
        "text"), num_epochs = 1)


train_dataset %>%
    reticulate::as_iterator() %>%
    reticulate::iter_next()  #%>% 
[[1]]
OrderedDict([('text', <tf.Tensor: shape=(32,), dtype=string, numpy=
array([b'I am years old and have read many books like this one before I found that it said little while being very erudite If I was younger I would probably be a bit wowed and perhaps confused by it Maybe if I was in my s I would have found it a better read At my level of development I learned little I didn t already know and found little content',
       b'I would not recommend this flashlight to anyone who actually wants to be able to see in the dark It is really dim and you can hardly see an inch in front of you with this flashlight My recommendation don t waste your money on this',
       b'Has tons of game storage but I bought a starter pack initially which hardly held more than the game I wish this was just a bit bigger If you have a travel case on the game it s then hard to fit all the adapters and accessories in this case too',
       b'An abortion Read other star reviews for more depth but please please do not read anything translated by Andrew Hurley It s sad that people don t realize there are other options out there probably because this complete fictions pop up first on Amazon Start with Labyrinths Do not read anything translated by Andrew Hurley read Di Giovanni Yeats',
       b'I have loved Wrinkle in Time since I first read it in grade school It is a wonderful book that never gets old I thought that the audio version would be perfect for my iPod I have to say while I love the author s books her reading this story is horrible As the another reviewer said she has a lisp or speech impediment or something It is almost impossible to finish this audio version Her reading is also very slow in spots especially when she is reading the parts of Mrs Which I would love to hear this read by a professional reader or an actor Perhaps now that she has passed away someone will revisit making the audio version of Wrinkle in Time',
       b'I hate sounding like a commercial but this product is amazing It melted the mastic into a pool of dirty water that I could clean up with a sponge No scraping at all It not only cleaned up the mastic but it also cleaned up the mess I made attempting to use other mastic removing techniques D And best of all NO FUMES My cat and I were comfortably in the apartment through the whole process',
       b'This book is unusual for a Holocaust book because none of the protagonists are wholly good or entirely bad The villain is capable of love and the female lead character is unable to love Good men are damaged goods and even the innocent next generation is disabled It definitely is worth reading',
       b'This product is strange First off every time i start a game I have to unplug the card to get a reaction from the actual playstation controller I dunno if this is the products fault or the console itself but it s rather irritating And second I noticed that some of my games don t even save That one I m pretty sure I can blame the card for Personally I wouldn t buy this card again',
       b'I used this figure for parts for a custom Darth Sion figure I made It actually makes a great Darth Sion if you repaint it I recommend using the Anakin burn damage head',
       b'I have many problems with this book First it is not a reprint of a old book but a photocopy of the book It is hard to read the text The original book had some color photos but in this photocopy those photos are missing Imagine in a book about matching color and grain patterns with no photos',
       b'Had used it for about months and within the last months I was thinking to throw it to the cemend floor from the roof of my house after every time I used it And today I did It was so hard to load the paper correctly lot of wired parts were blocking the paper I needed to spend around min to feedone piece of paper to the tray till this piece of paper looked like a piece of toilet paper after I tried all my effort to finish the fax the poeple on the other end might receive couple pieces of blank paper I d rather to throw this junk to a garbage can than donate it to anyone who really needs a fax machine because a garbage can is where this machine really belongs to trust me',
       b'The newest remake of this movie doesn t do justice to the story This is an all time great and my year old loves this movie just as much as his dad He had a ton of questions and we spoke about the movie for weeks after we saw it We continue to watch it together again and again',
       b'I have a nine foot tree The bottom three pieces needed to go in their own bag and the top two fit in one For the price I suggest buying an extra bag so you don t have to stuff the tree in',
       b'I have recently purchased this wine making book It contains many receipies for many diferent types and styles of wines I have found it to be very helpful in my efforts to making good sound wines',
       b'I think this is a very comprehensive report on reflux problems I have a sonwith Barrett s Esophagus and he seems to ignore the importance of diet He eatsfast food often and does not pretend to be on a diet I bought this as a gift for him and think it will help him immensely It goes all the way explaining the different drugs and their benefits The menusare a big help for those who want to prepare meals at home It is well worth the price in my opinion',
       b'If I could rate this game lower than star I would You need an NVIDIA Video card to play this game My month old PC doesn t support Pixel Shader so this game won t run on it My yr old son purchased the game with his own money I hope Lego and Eidos realize they have just lost a young customer (and his parents)Be very careful to check your video card before purchasing this game',
       b'I really wanted to love these treats They re healthy and smell wonderful but my dogs will NOT touch them I ve tried everything and neither one a lab mix and a wheaton Havenese mix will not touch them Not sure why because they LOVE the duck wrapped sweat potato treats I ve gotten them beef sticks to chew on and they re just ok with those maybe it s my dogs',
       b'Having read most of her books Agatha Christie still outwits me Very seldom one can guess the final outcome of her books and this is one of the very typical specimens Actually all her books offer surprises and this particular one offers nothing less I would say it s another masterpiece of hers though the revelation of the real murderer actually saddens my heart (Yes very much )I agree with one of the reviews that say the best books of Agatha Christie are And Then There Were None and The Murder On The Orient Express (Try The Secret Adversary also) But the reviewer made a mistake by saying The Murder of Roger Ackroyd is the first book of Poirot actually it s not that should be The Mystery Affair at Styles',
       b'My lb jack russel destroyed two of these in hours of purchase don t waste ur money regular tennis balls are tougher',
       b'this salad spinner does not work and I m surprised at so many positive reviews The old string pull type I had (and recently threw away alas) let you run water into an opening while you spun it then you could turn off the water and spin it dry All the way dry unlike this one Because the string spinner let you get it going fast enough to repel all the water This one won t get going fast enough to get the greens all the way dry So you have to rinse the greens separately then put them in the spinner and then finish off by drying them with a paper towel Give me back my old string spinner',
       b'i am one of paulo coelho s greatest fans when i saw this book in the bookstore couldnt think of any other thing exept that i wanted to go home and start reading after the first twenty pages i discovered that it has nothing similar to the alchamest on the contrary to me this book appeared to be very boring a waste of time it wasnt worth my exitment about it in the first place i don t advise anyone to read it it is not believable even the parts where supposedly paulo was conversing with his angel were very shallow paulo and his wife chris are too boring i felt very happy that i have finished this stupid book so that i can start another one to make make me forget my dissappointment about coelho',
       b'The first time I turned the oven on it gave off a horrible chemical smell and set off the smoke alarm Back it went to Amazon Fortunately Amazon was great about the return Ended up getting a Krups toaster oven more expensive but no major issues yet',
       b'I am an old (electrical engineer graduation ) and new engineering student (just started a bioengineering program) The new v software that came out today ( ) makes this already awesome calculator even better They have made the graphical user interface (GUI) more intuitive and have finally made the connectivity software bit compatible with Windows ( bit) I own both the TI nspires (non CAS and CAS) in order to be able to use the calculator on tests (some tests will not allow the CAS) and have to say that handheld tools have come a long way since using my HP GX for many years',
       b'Excellent product for the value Description should have listed that actual cleats were screw on Shipping was prompt Great cleats for the Weekend Athlete',
       b'I enjoyed Bryson s Walk in the Woods and recommend it frequently I wish I could say the same about this book Bryson would be the ultimate joykill on a road trip arrogant exceedingly negative and critical and unable to poke fun at himself Buy Walk in the Woods instead',
       b'I am not one to stuff into a import cd so I was pretty happy to find a Maaya Sakanoto cd that was reasonbly priced but also has almost all of my favorite tunes from her The only thing I have against this is the album art While I like the cover the album art gets pretty ridicudlous and poppy I understand well that s what J pop is like and the whole cultural thing However it still annoyed me Great cd though',
       b'This book is so far out of date as to be unusable Has very little relationship with C Builder Complete wast of money',
       b'This is not your Blackmore s Purple Tommy Bolin brings a real spark of life to the beast and it shows here s a band having FUN This is by far the most adventurous most creative and in my humble opinion best album Purple ever did Now don t get me wrong I am a fan of Purple and Blackmore but the band never sounded like they enjoyed themselves so much You can hear it listen closely this band lineup is tight All of this is very noticable if you listen to Ian s drumming he s flat out swinging Glenn s vocals and bass is right on the mark Tommy is nothing short of incredible and no he doesn t even try to do Richie he s his own man Thank God Lord and Coverdale are just like you would expect them great If your a biased Purple fan then you might not like this album but if you love great MUSIC then this one s for you',
       b'If you have read evene a few pages of any book by thix Nixonite then you hve tapped into the best that he has to offer (not much )Pass on this one',
       b'I bought the Scosche IPNRFC Wireless Remote mainly to use on my Jetski It makes it so much easier selecting artists songs creating playlists and changing the volume among other neat things I can keep my iPod securely in my glove box still have total control over everything that I want to listen to',
       b'The primary issue with his unit is that the word thermal is misleading This unit will not heat the air contrary to what you may think The bubbles will cool down the bath water',
       b'Berry reaches for the literary high ground but he stumbles on endlessly repeated imagery and numerous stunning errors of fact He s obviously interested in his topic and perhaps a less confident writer would have paid closer attention and gotten the details right'],
      dtype=object)>)])

[[2]]
tf.Tensor([0 0 1 0 0 1 1 0 1 0 0 1 1 1 1 0 0 1 0 0 0 0 1 1 0 1 0 1 0 1 1 0], shape=(32,), dtype=int32)
# reticulate::py_to_r()

# ----------------------

val_file_path <- file.path("amazon_review_polarity_csv/amazon_val.csv")

val_dataset <- make_csv_dataset(val_file_path, field_delim = ",", batch_size = batch_size,
    column_names = list("label", "title", "text"), label_name = "label", select_columns = list("label",
        "text"), num_epochs = 1)


val_dataset %>%
    reticulate::as_iterator() %>%
    reticulate::iter_next()
[[1]]
OrderedDict([('text', <tf.Tensor: shape=(32,), dtype=string, numpy=
array([b'Wish I d caught on earlier Everyone told me that the Harry Potter series was good I my expectations were definitely exceeded',
       b'Jennifer Love Hewitt doesn t look a thing like Audrey Hepburn I truely believe that the producers and casting directors made a terrible call casting Hewitt The actress who I believe reminds people of Audrey as well as looks like her is Natalie Portman (who is an exceptional actress) Natalie would have made a much mcuh better Audrey Hepburn',
       b'Absolutely charming book great for all grade levels I usually run for the gory werewolfie or anne boleynesque books but it was refreshing to pick up something sweet AND interesting for a change Buy this for the MG reader in your life but read it for yourself first',
       b'This movie was made to be a fantasy martial arts movie and it delivers as promised I have a very extensive collection of Martial Arts movies and I can honestly say that this movie is one of my top favorite movies I gave it a instead of a five because it does go a little overboard with the fantasy element but not enough to distract you from the fighting and believe me there s a lot of it',
       b'Aircraft physics cgi are horrible Planes don t fly that way And the German guy is of course dressed in black and he s really bad The hero becomes obsessed with getting even and of course does Most video games are superior to this sillyness Total waste',
       b'I had never read Treasure Island but upon receiving my kindle and the fact this was a free read I downloaded a copy I had always thought I should sit and read this one and I am happy I did The story never really got stagnate As in the story hanging around the sea port town too long or sitting in the island cabin The story keeps moving along with twists and turns It was a great adventure on a great ship with the cook to a little known island If you havn t read it and enjoy adventure stories and mystery dissete and double crossing and most of all sea adventures I suggest you try out this gem It s not too long and a great read',
       b'I tend to agree with some minority reviewers that this book is boring and dull It has some interesting thoughts here and there but that is all All in all the book is quite over rated',
       b'I read about halfway through this book and then I gave up I read James Turn of the Screw and Daisy Miller in high school and I remember liking the former and thinking the latter was just okay (I know I know it s a major classic by one of America s most celebrated writers but just because something has merit doesn t mean I like it better ) One of my all time favorite books was James Washington Square It s hard for me to believe that the same man wrote Square and Maisie This book is only for MAJOR Henry James enthusiasts',
       b'Typical cavalry versus Indian oater Lots of action which is what I ask from a Western',
       b'This movie is pretty good It didn t require an indepth knowledge of the TV show to follow the movie',
       b'This is first time I have been compelled to write a bad review but frankly this movie was bad Don t get me wrong there is nothing I love more than a romantic comedy However this was not romantic and not funny I didn t expect an Academy Award winning movie just a cute entertaining romantic comedy It didn t deliver The two stars have been great in everything I have seen them in but this movie just did not work for them I felt no chemistry between them Chris O Donnell just irritated me the entire movie and the lines these actors were expected to deliver were horrible I was embarrased watching this movie I ve read the previous reviews and I must say I m interested in seeing the original so I can see what they could have done with this movie Skip this one and see Return to Me when it comes out on DVD Video',
       b'The Third Secret is another entry in the Vatican intrigue genre It has the usual elements Vatican conservatives the usual bad guys vs liberal Catholics the usual good guys The Third Secret refers to secret revelations given to Marian visionaries at such places as Fatima and Medjugorge the existence of which is historical fact If there is any suspense in the novel it is not with the plot which is remarkably predictable and derivative The only thing that kept my interest was waiting to find out the actual content of the third secret(s) at least according to the novel s author In the end these proved to be rather predictable too given the author s religious inclinations',
       b'I can t believe I wasted my money on this When I got the product I had before hand read some reviews about it falling apart or something So before I wasted the sand that came with it and caused a mess I gently pressed on the wood base part It was flimsy but I didn t think much of it Then while just doing a quick look over I noticed that there was a small disconect between the bottom base and the black lining This would cause the sand to leak out of the crack Then the base fell write off I noticed that it had very tiny small metal pieces that had wood from the base around it I now realize that they probably put the metal into it painted the top black then pushed on the base to connect the two parts The little stubs that hold it up have superglue stains from where they glued it to the bottom of the base How cheap I really don t know what to do now I ll probably get my dad to put some nails in it Really the Dollar Tree would probably make a better structure',
       b'The Spongebob Squarepants Movie had an unoriginal title but I LOVED the movie anyway I only liked two songs from it tho Goofy Goober Rock and The Best Day Ever But overall I recommend it for kids yrs old to adult This movie isn t a waste of time or money so buy it now I also recommend this for anyone who loves Spongebob (and Patrick) who loves silly but sometimes twisted comedy or someone who just wants to have a good time',
       b'Under the table surface there are two factory assembled bars used to connect the four legs Clearly the bars on my table were misaligned as they almost stick out of the table surface at one end (I will upload a photo for that) and left no room for the legs This rendered the whole table useless It surprised me how such a defected part could get through the manufacturer s quality control You don t need any fancy tool to discover it Looking at it would suffice Finally I chose to fix it myself It was not very difficult for me just taking off the screws that hold the bars realigning the bars to the correct position and then put the screws back in again The extra minutes work was still worth it comparing to repackaging the whole set the hassle to ask for a return plus shopping for another table The quality of the table (other than the misaligned bars) is OK That s why I still give it two stars But I would really hesitate to buy again from this manufacturer',
       b'I read the comment by the reviewer from Montgomery Alabama back in September that the ASV would be available from Star Bible in November I can find no information about Star Bible and the ASV still appears as out of stock or out of print in all my searches on Amazon Can anyone help me find a source for a printed copy of the ASV If so please respond here or email to nathancci hotmail com Thank you very much',
       b'I ordered these beds because my grandkids were sleeping on the floor and couldnt afford real beds and in less than months both beds have popped They didnt jump on them they just lost air seemed like the seams were the problem bed was replaced but the other remains not replaced I would never buy these type of beds ever again or this brand They were not cheap and I cant afford to replace them again So now they are back to sleeping on the floor again I heard they were good or I would not have ordered them I am terribly disappointed',
       b'I looked forward to seeing this movie but it was a huge letdown Besides the historical inaccuracies as detailed by earlier reviewers here on Amazon I felt the actors chosen to play these music icons weren t based on ability but to try and get a younger audience to pay attention to the film Mos Def is NOT an actor in any way shape or form his portrayal of Chuck Berry was wooden and inaccurate On the other hand Eamonn Walker s take on Howlin Wolf was STUNNING I m sure the makers intentions were good but the obvious budget restraints and poor choices in actors left me feeling like a great opportunity to shed light on an otherwise under documented period of music history was squandered',
       b'Bought this book by accident and now I love it It is easy to take to clinicals or even to look up different disease process it has nsg diagnosis with each one also and it is cheap',
       b'I went into this movie that it was going to be stupid But I didn t even find it funny Sure there are some scenes that make you laugh but its cause they are so pathetic The acting was terrible all around and the story made no sense what so ever This is just a bad movie with a cool title',
       b'And these masters are Michael Bay and his computers And we have no one but ourselves to blame This movie is like a hours car crash When you think it can t get any worse it does It would deserve stars if it didn t have any human actors But Michael Bay is actually controlled by his computers and they made him put these humans in this movie to make us suffer even more This is all I have to say Transformers IV is coming soon to the theaters near you Preorder your tickets now',
       b'I was surprised and delighted to see this remake After purchasing I was well disappointed I guess because the framework of the story was the same but nothing else was The modernisation of the action was poor the story only Ok and I didn t find it engaging except in one scene A satisfying ending with all the story nicely wrapped up but I think the cast deserved a better story line and sticking to the original concept and doing a real remake with modern cinema techniques could have made this a four star candidate Oh well I guess we ll never see a good modern updated version because this certainly is not it',
       b'The whole album is amazing and hyper ballad has got to be one of the most beautiful songs ever',
       b'I have a Samsung Blu Ray and the BR disc of the th Anniversary version of Dirty Dancing The picture quality of the movie is just awful particularly on the indoor scenes Faces and backgrounds were almost fuchsia almost always I did not adjust my TV color since all of the extras including extended scenes outtakes and deleted scenes were perfect quality What a pity My first BR disc purchase and such a disappointment The manufacturer should be ashamed Dirty Dancing ( th Anniversary Edition) Blu ray',
       b'I like family love movie very much From the view point of a father this movie is excellent In the movie Frank saved his son John who was just murdered at the last scene I was moved by the great love of a father Probably I wouldn t show my love to my sons as much as Frank In Japan mothers usually take care of children and fathers aren t good at showing their love to family members So a father tends to be isolated from his family Of course I am always concerned about my family But I don t tell them my concerns directly So my affections aren t handed to them I think Frank is a father of fathers I try to imitate Frank The movie tells me an important role of a father',
       b'Mine quite pumping water after cups I should have known better since my oldest son had sent his back times before getting one that worked Now his is once again screwing up I thought that for the amount of time this model had been on the market it would have been fixed I have an older B we have been using for a long time but it is screwing up now That s why I ordered this one The B won t fit under our cabinets or I would have tried it No more Keurig s for me I will try another brand now',
       b'I am the author of this book It is no longer being printed I gave it only four stars because its successor is deserving of five stars Go to the following Amazon com page to see West Side Publishing s Armchair Reader The Last Survivors of Historical Events Movies Disasters and More Armchair Reader The Last Survivors of Historical Events Movies Disasters and MoreRobert Ernest Hubbard',
       b'This is the book that got me interested in reading mystery novels It s suspenseful thought provoking and totally unpredictable I have yet to find another mystery that really adds up to what Doyle has created in these few pages I want to read all his novels now and I can t wait',
       b'I did not really care for this film though it did have a few funny moments a lot of the attempts at humor fell flat for me I really like Martin Short but didn t think he was very funny in this It was a little corny I also thought that it was going to be more of a musical but there was very little singing so I was disappointed in that I also didn t like the moral behind the plot of the wish itself getting the little girls dad the role in the play by using any means to sabbotage the character who had the role I don t think that s a good message for kids even if the methods used are done by fairy godmothers I know a lot of shows do that like Bruce Almighty but in even that show in the end he gave the newscaster s job back to the character that he stole it from I wouldn t recommend this movie to anyone',
       b'This book had so much potential to be far more than what it was I agree with other customers who said it started off well but all of sudden there were so many U turns and loops and twists that it seems as though the writers were still brainstorming what they wanted to do with all the information they had It was poorly written and it takes away what probably could have been a journey worth reading I think maybe Lula should have considered someone else to do her biography Either these writers were inexperienced or had too many projects on their plates and just stuffed this one in',
       b'I wear a size I took the info left by others and ordered a size They fit great Warm but not hot and the this rubber sole means I can let the dogs out without having to take them off',
       b'Mr Waltari s storytelling style comes forth in colorful detail and complex characters The mix of history intrigue and involves the reader in a way that one feels as if one is in these ancient places with these people The characters are fleshed out to the point that one could almost toch these people and engage them in a conversation The Egyptian is a tour de force of storyteling and social commentary It is a great read and a compelling story for lovers of history and epics'],
      dtype=object)>)])

[[2]]
tf.Tensor([1 0 1 1 0 1 0 0 1 1 0 0 0 1 0 1 0 0 1 0 0 0 1 0 1 0 1 1 0 0 1 1], shape=(32,), dtype=int32)

# -----------------------------------

test_file_path <- file.path("amazon_review_polarity_csv/amazon_test.csv")


test_dataset <- make_csv_dataset(test_file_path, field_delim = ",", batch_size = batch_size,
    column_names = list("label", "title", "text"), label_name = "label", select_columns = list("label",
        "text"), num_epochs = 1)

test_dataset %>%
    reticulate::as_iterator() %>%
    reticulate::iter_next()
[[1]]
OrderedDict([('text', <tf.Tensor: shape=(32,), dtype=string, numpy=
array([b'The one star is for Liam s hair on the inside cover At least that still has it s dignity As for the music I believe they have finally lost their touch The Importance of Being Idle Part of the Queue and Let There Be Love are mediocre Unfortunately all of the other tracks are completely unlistenable It s sad that a band who was once so great are putting out this kind of trash but luckily for their fans they let us down slowly with Standing On The Shoulders Of Giants and Heathen Chemistry so at least it didn t come as a big surprise',
       b'The spreader is much larger and wider than I wanted I have found it useless for buttering bread sticks which is why it was ordered This is my fault and not that of the spreader',
       b'This is so typical of H James writing very make that incredibly dense Wordy beyond belief At a pace that surely puts me fast asleep after a mere pages A ridiculous pair of uncaring parents essentially abandon their daughter to whomever comes along As an amazing coincidence the divorced parents each take a new spouse only to watch the wife of one have an affair with the husband of the other If you are really into dialogue that no one has ever spoken in the history of English and willing to devote hours to really obscure language construction this is the very winner for you',
       b'I previous had a similar jar opener and have places and wanted to keep one in one place and the other in another This jar opener is an inferior product and the worse thing I ever purchased from Amazon(and I purchased numerous things)It doesn t grips all jars and it should have advised me of what it was made of It looked various similar to the other one I own and I can t wait till I can find another jar opener like the other one so I can throw this one in the garbage',
       b'Wasn t sure what was wrong with my ice maker so I bought this kit Replaced just the main ice maker component (took less than minutes) and the unit started right up It has been making ice reliably Great deal',
       b'This book is a watershed in human intellectual history In it Freud undermines the picture of mankind as primarily a being of reason and presents the idea that we are all creatures of our wishes our inner unconscious lives Dreams are not nothing and they are not in Freud s eyes rare religious gifts but rather to the key to our own mental life Freud in this book presents a vast world of examples and interpretations I am not a psychologist and do not consider myself competent to really judge how much of what Freud presents here is valid or even capable of scientific testing I do know that this work is one which like a great literary masterpiece has inspired countless interpretations and reinterpretations Understanding human Intellectual History is now impossible without knowing this work',
       b'The game includes nowhere to store the black sheets for the screen The storage drawer jams every time you close it so opening it tends to result in pegs flying all over the room You can turn the light on but not off I seem to recall that we had on off light technology even way back when I was a kid The pegs don t stay in well the blue pegs tend not to glow at all unless they re directly above the light bulb You d think with many generations of children to evolve and perfect this toy it would have gotten better rather than worse your expectations would be frustrated',
       b'This is a very good cd you should buy it My favourite songs are Why does my heart feel so bad Porcelain guitar flute and stings and although almost all songs on the album are good',
       b'I thought at first that the vacuum was extremely hard to push until I realized that the lowest setting was for smooth floors rather than carpet Once I adjusted the dial the vacuum moved much more easily It has great suction and picks up so much dirt and dog hair from my carpet that I m embarrassed about my housekeeping I really like being able to see in the clear cylinder what it s picking up from the floor and it is very easy to empty the canister The cord rewinder is excellent The vacuum s a bit loud but no more than most that I ve had It s definitely worth the money',
       b'There are many accolades and volumes of in depth analysis for this film so my review will be short The good Strong weak sad wise frail crazy calm etc the full gamut of characters that really work together to twist complex situations The story keeps you guessing first time It is timeless and the subitles did not bother me because the pace of the movie is easy enough to keep up with reading but also watching The bad It can move too slow at times and I had to break up the film and view over two nights Conclusion This is a movie I will watch again probably a few times if lucky',
       b'This book did not move me at all The plot is unrealistic and the characters are shallow they do not seem like real people at all',
       b'Aaliyah has definitely grown up The songs are about love hurt pain sex etc The songs are more sensual than her usual style My favorite track is Rock the boat then I care for you then I Refuse I am a huge Aaliyah fan Keep up the good work baby girl',
       b'Whoa boy I ve seen some pretty lame flicks last year but The Skulls definitely comes in as one of the worst It s a mind bogglingly stupid tale about a pretty boy (Joshua Jackson) who stupidly decides to join a secret society and it seems all the members of this society called The Skulls are pretty dumb in their own rights too Of course this didn t prepare me for the lame as heck finale which features a duel that s right a duel involving two flintlock pistols',
       b'In Flames and Soilwork are sold out so my only chance to find another album that blow my mind like Dark Tranqility s Character is this This album is solid musicianship throghout There s no filler here or the commercial flavor like the other two(In Flames Soilwork) this is great melodic death metal with no complaints There s industrial touches like in Character but sounds really good Check this out you won t regret it',
       b'I guess I should preface this by saying that I m really not an Asian movie person I just don t get them and I m sure it s a cultural thing Tampopo Crouching Tiger In the Realm of the Senses and now this one I couldn t sit through any of them Unless you are a martial arts or samurai fanatic or a big lover of Japanese culture and history I would say definitely skip this especially if you are female I think this is one of those Emperor s New Movies that we re all supposed to like and no one will admit to NOT liking it for fear of being thought ignorant',
       b'OPENED THE BOX AND IT WAS NOT CARBON FIBER IT WAS CLEAR AMAZON NEEDS TO CHECK THEIR INVENTORY ON THIS update NOV Amazon s replacement was still the same it s not d carbon fiber will return for a refund',
       b'Danger Kitty has mass produced some of the worst music in the history of Pop To all of the BJ fans out there look up the word cliche Understand its danger and the shallow nature that this word connotes It s ahem like a wet slippery loaded gun If you re going to write lyrics this horrible juvenile and unimaginative why bother finishing Junior High at all Drop out of school get yourself a job at Hot Dog on a Stick and say goodbye to every one of your brain cells Patriotism demands that you rid your CD player of anything that makes you look like you couldn t finish the th grade Do it for your country',
       b'I am only years old but at least read a book a week When i first picked up this book i felt i would not beable to finish it because of my ablity to get bored really quick I started reading it and couldn t put it down I got in trouble in classes because i had been reading it under my desk until i reached a point (which was very rare) that i could put it down I love Jane Eyre s way of thinking and how she trys to make her life as full as possible I would recomend this to everyone I keep on going back in the book to a part i liked best or a place in the story that captivates me It seems i will never grow tired of this book I hope all that have read it feel the same',
       b'I thought the book was very good It does get a little boring to me because of so much explicit detail written in on everything The book is around pages but well worth the read Can t wait for the final book',
       b'This book was given to me I read it and placed it on my book shelf As a writer I have used the quotes in articles and letters A perfect addition to your library if you write to or about children',
       b'The first of this movie are spent developing the characters of Liu Xing and Prof Reiser and then all the strongest character traits of those two people are just tossed aside for the ridiculous ending That made the entire film a complete waste of time You re left wondering what moron wrote that idiotic ending and why Anyone with a shred of intellect isn t going to be moved by this except maybe to anger at having squandered minutes on it It could have been a decent movie It did have a really cheap quality about it as you never see ANY other students at the school',
       b'The Nantucket Blend is one of my favorite k cup coffees that I ve made It s a medium blend coffee that tastes great with flavored creamer or just regular creamer or milk I will definitely buy more of this one',
       b'My wife and I are planning a trip to Tuscany in May and ordered this video to get ideas about off the beaten track places to go The video is beautifully shot well narrated and contains useful information On the negative side though its a bit thin on detail and far too short in running time It left us wanting more but certainly excited about our upcoming trip In that sense I suppose it was worth the money',
       b'Okay if you re a special effects freak you might like this However there are much better films with equally good effects If you re a fan of minor things like say oh plot character development etc then stay away The Robert Wise film is the scariest film I ve ever seen and the scares come from what Wise DOESN T show you The version was brilliantly plotted and paced with one of the greatest ending lines in film history The version has none of this Do yourself a favor and see the movie or better yet read the Shirley Jackson novel The only reason this gets one star is because C Zeta Jones is just SO easy on the eyes',
       b'I checked this book out from my local La Leche League I m surprised they still have this book available Being nearly twenty years old maybe some of my problems are with it being outdated However my main gripe is that it is very condescending and judgmental Instead of just informing you on different options (none of which would work for me and only one or so that would apply well to a full time job) this book presents a certain ideal and if the job arrangements do not fit that ideal they re not good arrangements and you aren t a good mother If you want helpful non opinionated advice this is not the book for you Currently I m reading The Working Mother s Guide to Life which seems much more friendly and helpful',
       b'Bought this for wife as a stocking stuffer I m sure she ll enjoy backing her team without taking up the whole window',
       b'Recently bought the Sony SS B speakers As many have mentioned are a little larger than most bookshelf speakers However these speakers are an exceptional value for the money Plus they sound fantastic clear highs smooth mid range and good bass response Most music lovers will be satisfied with the sound and these are also nice looking speakers You cannot go wrong',
       b'Was Very Very disappointed Movie though brand new just out of the box it was stopping all the time during the latter part of the movie',
       b'This is the worst movie you will ever watch no plot no story terrible casting and nothing to do with the original series Ugh It sucks',
       b'I just got this unit and must say that I am disappointed It maybe because I was expecting it fully decode hdmi audio instead of having another optical in for the same source So when I connect my PS I can not listen to full uncompressed audio in LPCM which is available only in hdmi If you have only optical audio to take care of you will be find with this receiver But if you are looking at it just because it has HDMI beware and read the fine prints',
       b'This book is stupid the romance the vampires everything and why did she use that poem Amelia never actually stated if these vampires are demon or not They hold emotions of love and that doesn t make em demon',
       b'My Xantrex XPower Powerpack Heavy Duty is not good for an emergency It is an emergency The unit does not hold a charge It takes days to fully charge and loses its charge by each day Xantrex customer service has not been helpful other than to say that I should charge it through my car cigarette lighter I ask what should I do charge it all the time Thats the only way their will be any power when you need it I bought this item to help if I have a problem It is the problem'],
      dtype=object)>)])

[[2]]
tf.Tensor([0 0 0 0 1 1 0 1 1 1 0 1 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 0 0 0 0 0], shape=(32,), dtype=int32)

rm(amazon_orig_train, amazon_orig_test, amazon_train, amazon_val)
Warning in rm(amazon_orig_train, amazon_orig_test, amazon_train, amazon_val):
object 'amazon_orig_train' not found
Warning in rm(amazon_orig_train, amazon_orig_test, amazon_train, amazon_val):
object 'amazon_orig_test' not found
Warning in rm(amazon_orig_train, amazon_orig_test, amazon_train, amazon_val):
object 'amazon_val' not found
rm(ids_train, train_file_path, test_file_path, val_file_path)
Warning in rm(ids_train, train_file_path, test_file_path, val_file_path): object
'ids_train' not found
```

\newpage
## The preprocessing model

Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT. TensorFlow Hub provides a matching preprocessing model for the BERT models, which implements this transformation using TF ops from the TF.text library.  

The preprocessing model must be the one referenced by the documentation of the BERT model, which you can read at the URL [bert_en_uncased_preprocess](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)


```r
bert_preprocess_model <- layer_hub(handle = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    trainable = FALSE, name = "preprocessing")
```

## Using the BERT model

The BERT models return a map with 3 important keys: pooled_output, sequence_output, encoder_outputs:  

"pooled_output" represents each input sequence as a whole. The shape is [batch_size, H]. You can think of this as an embedding for the entire Amazon review.  

"sequence_output" represents each input token in the context. The shape is [batch_size, seq_length, H]. You can think of this as a contextual embedding for every token in the Amazon review.  

"encoder_outputs" are the intermediate activations of the L Transformer blocks. outputs["encoder_outputs"][i] is a Tensor of shape [batch_size, seq_length, 1024] with the outputs of the i-th Transformer block, for 0 <= i < L. The last value of the list is equal to sequence_output.  

For the fine-tuning you are going to use the pooled_output array.  

For more information about the base model's input and output you can follow the model's URL at [small_bert/bert_en_uncased_L-4_H-512_A-8](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2)



```r
bert_model <- layer_hub(handle = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
    trainable = TRUE, name = "BERT_encoder")
```

## Define your model

We will create a very simple fine-tuned model, with the preprocessing model, the selected BERT model, one Dense and a Dropout layer.  


```r
input <- layer_input(shape = shape(), dtype = "string", name = "text")

output <- input %>%
    bert_preprocess_model() %>%
    bert_model %$%
    pooled_output %>%
    layer_dropout(0.1) %>%
    # layer_dense(units = 16, activation = 'relu') %>%
layer_dense(units = 1, activation = "sigmoid", name = "classifier")

# summary(model)
```


```r
model <- keras_model(input, output)
```


Loss function  

Since this is a binary classification problem and the model outputs a probability (a single-unit layer), you'll use "binary_crossentropy" loss function.  

Optimizer  

For fine-tuning, let's use the same optimizer that BERT was originally trained with: the "Adaptive Moments" (Adam). This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments), which is also known as [AdamW](https://arxiv.org/abs/1711.05101).  

We will use the AdamW optimizer from [tensorflow/models](https://github.com/tensorflow/models).

For the learning rate (init_lr), you will use the same schedule as BERT pre-training: linear decay of a notional initial learning rate, prefixed with a linear warm-up phase over the first 10% of training steps (num_warmup_steps). In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5).



```r
epochs = 5
steps_per_epoch <- 2e+06
num_train_steps <- steps_per_epoch * epochs
num_warmup_steps <- as.integer(0.1 * num_train_steps)

init_lr <- 3e-05
opt <- o_nlp$optimization$create_optimizer(init_lr = init_lr, num_train_steps = num_train_steps,
    num_warmup_steps = num_warmup_steps, optimizer_type = "adamw")

model %>%
    compile(loss = "binary_crossentropy", optimizer = opt, metrics = "accuracy")

summary(model)
Model: "model"
________________________________________________________________________________
Layer (type)              Output Shape      Param #  Connected to               
================================================================================
text (InputLayer)         [(None,)]         0                                   
________________________________________________________________________________
preprocessing (KerasLayer {'input_mask': (N 0        text[0][0]                 
________________________________________________________________________________
BERT_encoder (KerasLayer) {'default': (None 28763649 preprocessing[0][0]        
                                                     preprocessing[0][1]        
                                                     preprocessing[0][2]        
________________________________________________________________________________
dropout_4 (Dropout)       (None, 512)       0        BERT_encoder[0][5]         
________________________________________________________________________________
classifier (Dense)        (None, 1)         513      dropout_4[0][0]            
================================================================================
Total params: 28,764,162
Trainable params: 28,764,161
Non-trainable params: 1
________________________________________________________________________________
```

REMEMBER to change tr_count back to 10000 for better training.

Try a sample/subset to train/test code, and to reduce training time due to resource constraints use a smaller tr_count below.


```r
# 10000 will take approx 40 mins per epoch on my gpu/mem etc 1000 will take
# approx 4 mins per epoch on my gpu/mem etc

tr_count <- 10000
take_tr <- 0.8 * tr_count
train_slice <- train_dataset %>%
    dataset_shuffle_and_repeat(buffer_size = take_tr * batch_size) %>%
    dataset_take(take_tr)

take_val <- 0.2 * tr_count
val_slice <- val_dataset %>%
    dataset_shuffle_and_repeat(buffer_size = take_val * batch_size) %>%
    dataset_take(take_val)
```
  
  


```r
epochs <- 5
seed = 42

history <- model %>%
    fit(train_slice, epochs = epochs, validation_data = val_slice, initial_epoch = 0,
        verbose = 2)
```


"BERT Model Fit History using validation_data slice" Figure \ref{fig:model_20}  

![BERT Model Fit History using validation_data slice\label{fig:model_20}](figures/keras_model_fit_history-1.pdf) 
  
  
Evaluate the model  

Let's see how the model performs. Two values will be returned. Loss (a number which represents the error, lower values are better), and accuracy.

Takes too long, so skipping it for now. Using test_slice instead.

```r
model %>%
    evaluate(test_dataset)
```
  
Using test_slice instead.  


```r
test_slice <- test_dataset %>%
    dataset_take(100)

model %>%
    evaluate(test_slice)
     loss  accuracy 
0.2556886 0.9012500 
```





\newpage
## Results  

BERT model results Table \ref{tbl:bert_results_table}:


Table: BERT Model Results\label{tbl:bert_results_table}

Index   Method     Accuracy        Loss
------  -------  ----------  ----------
1       BLM       0.8897064          NA
2       DNN       0.8960983   0.2790776
3       CNN       0.8982586   0.2526298
4       sepCNN    0.9051675   0.2408866
5       BERT      0.9051719   0.2386644
 



```r
# Stop the clock
# proc.time() - ptm
Sys.time()
```

```
## [1] "2021-07-28 05:13:36 EDT"
```


<!-- ```{r knit_exit} -->
<!-- knitr::knit_exit() -->
<!-- ``` -->

<!-- \newpage -->

<!-- # Results -->

---  

\newpage
# Conclusion

In this project, we attempted to significantly simplify the process of selecting a text classification model. For a given dataset, our goal was to find the algorithm that achieved close to maximum accuracy while minimizing computation time required for training. 

CNNs are a type of neural network that can learn local spatial patterns. They essentially perform feature extraction, which can then be used efficiently in later layers of a network. Their simplicity and fast running time, compared to other models, makes them excellent candidates for supervised models for text.  

Based on our results and inspite of using only a fraction of our data due to (my) resource limitations, we agree with Google and conclude that sepCNN's and/or BERT helped us achieve our goal of simplicity, minimum compute time and maximum accuracy.  

As of now, my future attempts in ML will be in NLP related activities.

---  

\newpage
# Appendix: All code for this report


```r
knitr::knit_hooks$set(time_it = local({
  now <- NULL
  function(before, options) {
    if (before) {
      # record the current time before each chunk
      now <<- Sys.time()
    } else {
      # calculate the time difference after a chunk
      res <- difftime(Sys.time(), now)
      # return a character string to show the time
      # paste("Time for this code chunk to run:", res)
      paste("Time for the chunk", options$label, "to run:", res)
    }
  }
}))

# knit_hooks$get("inline")
# knitr::opts_chunk$set(fig.pos = "!H", out.extra = "")
knitr::opts_chunk$set(echo = TRUE,
                      fig.path = "figures/")

# Beware, using the "time_it" hook messes up fig.cap, \label, \ref
# knitr::opts_chunk$set(time_it = TRUE)
#knitr::opts_chunk$set(eval = FALSE)
options(tinytex.verbose = TRUE)

# set pandoc stack size
stack_size <- getOption("pandoc.stack.size", default = "100000000")
args <- c(c("+RTS", paste0("-K", stack_size), "-RTS"), args)
# library(dplyr)
# library(tidyr)
# library(purrr)
# library(readr)
library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(ngram)
library(keras)
library(stopwords)

# Used in Baseline model
library(hardhat)

# BERT setup in its own section
# library(keras)
# library(tfdatasets)
# library(reticulate)
# library(tidyverse)
# library(lubridate)
# library(tfhub)
# import("tensorflow_text")
# o_nlp <- import("official.nlp")
# 
# Sys.setenv(TFHUB_CACHE_DIR="C:/Users/bijoor/.cache/tfhub_modules")
# Sys.getenv("TFHUB_CACHE_DIR")

set.seed(234)

# Start the clock!
# ptm <- proc.time()
Sys.time()
  library(ggplot2)
  library(kableExtra)
untar("amazon_review_polarity_csv.tar.gz",list=TRUE)  ## check contents
untar("amazon_review_polarity_csv.tar.gz")
train_file_path <- file.path("amazon_review_polarity_csv/train.csv")

test_file_path <- file.path("amazon_review_polarity_csv/test.csv")

# read data, ensure "utf-8" encoding, add column names, exclude rows with missing values(NA)
amazon_orig_train <- readr::read_csv(
  train_file_path,
  # skip = 0,
  col_names = c("label", "title", "text"),
  locale = locale(encoding = "UTF-8")) %>% na.omit()

# change labels from (1,2) to (0,1) - easier for binary classification
amazon_orig_train$label[amazon_orig_train$label==1] <- 0
amazon_orig_train$label[amazon_orig_train$label==2] <- 1

# removed numbers as they were too many and did not contribute any info

# amazon_orig_train$text <- str_replace_all(amazon_orig_train$text,"[^([[:alnum:]_])]"," ") %>% trimws() %>% str_squish()
# 
# amazon_orig_train$title <- str_replace_all(amazon_orig_train$title,"[^([[:alnum:]_])]"," ") %>% trimws() %>% str_squish()

# remove leading/trailing whitespace (trimws)
# trim whitespace from a string (str_squish)
# replace non alphabet chars with space

amazon_orig_train$text <- str_replace_all(amazon_orig_train$text,"[^([[:alpha:]_])]"," ") %>% trimws() %>% str_squish()

amazon_orig_train$title <- str_replace_all(amazon_orig_train$title,"[^([[:alpha:]_])]"," ") %>% trimws() %>% str_squish()

# create a validation set for training purposes
ids_train <- sample.int(nrow(amazon_orig_train), size = 0.8*nrow(amazon_orig_train))
amazon_train <- amazon_orig_train[ids_train,]
amazon_val <- amazon_orig_train[-ids_train,]

head(amazon_train)

# save cleaned up data for later use
write_csv(amazon_train,"amazon_review_polarity_csv/amazon_train.csv", col_names = TRUE)
write_csv(amazon_val,"amazon_review_polarity_csv/amazon_val.csv", col_names = TRUE)

# -----------------------------------------------
# read data, ensure "utf-8" encoding, add column names, exclude rows with missing values(NA)
amazon_orig_test <- readr::read_csv(
  test_file_path,
  # skip = 0,
  col_names = c("label", "title", "text"),
  locale = locale(encoding = "UTF-8")) %>% na.omit()

# change labels from (1,2) to (0,1) - easier for binary classification
amazon_orig_test$label[amazon_orig_test$label==1] <- 0
amazon_orig_test$label[amazon_orig_test$label==2] <- 1

# remove leading/trailing whitespace (trimws)
# trim whitespace from a string (str_squish)
# replace non alphabet chars with space

amazon_orig_test$text <- str_replace_all(amazon_orig_test$text,"[^([[:alpha:]_])]"," ") %>% trimws() %>% str_squish()

amazon_orig_test$title <- str_replace_all(amazon_orig_test$title,"[^([[:alpha:]_])]"," ") %>% trimws() %>% str_squish()

# amazon_orig_test$text <- str_replace_all(amazon_orig_test$text,"[^([[:alnum:]_])]"," ") %>% trimws() %>% str_squish()
# 
# amazon_orig_test$title <- str_replace_all(amazon_orig_test$title,"[^([[:alnum:]_])]"," ") %>% trimws() %>% str_squish()

head(amazon_orig_test)

# save cleaned up data for later use
write_csv(amazon_orig_test,"amazon_review_polarity_csv/amazon_test.csv", col_names = TRUE)

rm(amazon_orig_train, amazon_orig_test)
rm(ids_train, test_file_path, train_file_path)

# free unused R memory
gc()
#### To be deleted later
amazon_train <- readr::read_csv("amazon_review_polarity_csv/amazon_train.csv")
glimpse(amazon_train)
# head(amazon_train)
kable(amazon_train[1:10,], "latex", escape=FALSE, booktabs=TRUE, linesep="", caption="Amazon Train  data\\label{tbl:amazon_train}") %>%
    kable_styling(latex_options=c("HOLD_position"), font_size=6)
  # kable_styling(full_width = F)
unique(amazon_train$label)
(num_samples <- nrow(amazon_train))
(num_classes <- length(unique(amazon_train$label)))
# Pretty Balanced classes
(num_samples_per_class <- amazon_train %>% count(label))
# break up the strings in each row by " "
temp <- strsplit(amazon_train$text, split=" ")

# sapply(temp[c(1:3)], length)
# count the number of words as the length of the vectors
amazon_train_text_wordCount <- sapply(temp, length)

(mean_num_words_per_sample <- mean(amazon_train_text_wordCount))

(median_num_words_per_sample <- median(amazon_train_text_wordCount))
# Frequency distribution of words(ngrams)
train_words <- amazon_train %>% unnest_tokens(word, text) %>% count(word,sort = TRUE)

total_words <- train_words %>%
    summarize(total = sum(n))

# Zipf’s law states that the frequency that a word appears is inversely proportional to its rank.
train_words <- train_words %>%
    mutate(total_words) %>%
     mutate(rank = row_number(),
         `term frequency` = n/total)
# head(train_words)
kable(train_words[1:10,], "latex", escape=FALSE, booktabs=TRUE, linesep="", caption="Frequency distribution of words\\label{tbl:train_words}") #%>%
    # kable_styling(latex_options=c("HOLD_position"), font_size=6)
train_words %>%
  top_n(25, n) %>%
  ggplot(aes(reorder(word,n),n)) +
  geom_col(binwidth = 1, alpha = 0.8) +
   coord_flip() +
  labs(y="n - Frequency distribution of words(ngrams)",
       x="Top 25 words")
length(stopwords(source = "smart"))
length(stopwords(source = "snowball"))
length(stopwords(source = "stopwords-iso"))
mystopwords <- c("s", "t", "m", "ve", "re", "d", "ll")

# Frequency distribution of words(ngrams)
train_words_sw <- amazon_train %>% unnest_tokens(word, text) %>%
    anti_join(get_stopwords(source = "stopwords-iso"))%>%
    filter(!(word %in% mystopwords)) %>%
    count(word,sort = TRUE)

total_words_sw <- train_words_sw %>%
    summarize(total = sum(n))

# Zipf’s law states that the frequency that a word appears is inversely proportional to its rank.

train_words_sw <- train_words_sw %>%
    mutate(total_words_sw) %>%
     mutate(rank = row_number(),
         `term frequency` = n/total)
# head(train_words_sw)
kable(train_words_sw[1:10,], "latex", escape=FALSE, booktabs=TRUE, linesep="", caption="Frequency distribution of words excluding stopwords\\label{tbl:train_words_sw}") #%>%
    # kable_styling(latex_options=c("HOLD_position"), font_size=6)
train_words_sw %>%
  top_n(25, n) %>%
  ggplot(aes(reorder(word,n),n)) +
  geom_col(binwidth = 1, alpha = 0.8) +
   coord_flip() +
  labs(y="n - Frequency distribution of words(ngrams) excluding stopwords",
       x="Top 25 words")
# 3. If the ratio is greater than 1500, tokenize the text as
# sequences and use a sepCNN model
#    see above

(S_W_ratio <- num_samples / median_num_words_per_sample)
amazon_train %>%
  mutate(n_words = tokenizers::count_words(text)) %>%
  ggplot(aes(n_words)) +
  geom_bar() +
  labs(x = "Number of words per review text",
       y = "Number of review texts")
amazon_train %>%
  mutate(n_words = tokenizers::count_words(title)) %>%
  ggplot(aes(n_words)) +
  geom_bar() +
  labs(x = "Number of words per review title",
       y = "Number of review titles")
amazon_train %>%
  group_by(label) %>%
  mutate(n_words = tokenizers::count_words(text)) %>%
  ggplot(aes(n_words)) +
  # ggplot(aes(nchar(text))) +
  geom_histogram(binwidth = 1, alpha = 0.8) +
  facet_wrap(~ label, nrow = 1) +
  labs(x = "Number of words per review text by label",
       y = "Number of reviews")
amazon_train %>%
  group_by(label) %>%
  mutate(n_words = tokenizers::count_words(title)) %>%
  ggplot(aes(n_words)) +
  # ggplot(aes(nchar(title))) +
  geom_histogram(binwidth = 1, alpha = 0.8) +
  facet_wrap(~ label, nrow = 1) +
  labs(x = "Number of words per review title by label",
       y = "Number of reviews")
amazon_subset_train <- amazon_train %>% select(-title) %>%
  mutate(n_words = tokenizers::count_words(text)) %>%
  filter((n_words < 35) & (n_words > 5)) %>% select(-n_words) 

dim(amazon_subset_train)
# head(amazon_subset_train)
kable(amazon_subset_train[1:10,], "latex", escape=FALSE, booktabs=TRUE, linesep="", caption="Sample/Subset of our training dataset\\label{tbl:amazon_subset_train}") #%>%
    # kable_styling(latex_options=c("HOLD_position"), font_size=6)
# Free computer resources
rm(amazon_train, amazon_val, amazon_train_text_wordCount,num_samples_per_class, temp, total_words, train_words)
rm(mean_num_words_per_sample, median_num_words_per_sample, num_classes, num_samples, S_W_ratio)
gc()

# save(amazon_subset_train)
write_csv(amazon_subset_train,"amazon_review_polarity_csv/amazon_subset_train.csv", col_names = TRUE)

amazon_train <- amazon_subset_train

amazon_train <- amazon_train %>%
  mutate(label = as.factor(label))

# amazon_val <- amazon_train %>%
#   mutate(label = as.factor(label))
set.seed(1234)

amazon_split <- amazon_train %>% initial_split()

amazon_train <- training(amazon_split)
amazon_test <- testing(amazon_split)

set.seed(123)
amazon_folds <- vfold_cv(amazon_train)
# amazon_folds
# library(textrecipes)

amazon_rec <- recipe(label ~ text, data = amazon_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 5e3) %>%
  step_tfidf(text)

amazon_rec
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

lasso_spec
library(hardhat)
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")
lambda_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 20)
lambda_grid
amazon_wf <- workflow() %>%
  add_recipe(amazon_rec, blueprint = sparse_bp) %>%
  add_model(lasso_spec)

amazon_wf
set.seed(2020)
lasso_rs <- tune_grid(
  amazon_wf,
  amazon_folds,
  grid = lambda_grid,
  control = control_resamples(save_pred = TRUE)
)

# lasso_rs
m_lm <- collect_metrics(lasso_rs)
kable(m_lm, format = "simple", caption="Lasso Metrics\\label{tbl:lasso_metrics}")
m_blr <- show_best(lasso_rs, "roc_auc")
kable(m_blr, format = "simple", caption="Best Lasso ROC\\label{tbl:best_lasso_roc}")
m_bla <- show_best(lasso_rs, "accuracy")
kable(m_bla, format = "simple", caption="Best Lasso Accuracy\\label{tbl:best_lasso_acc}")
autoplot(lasso_rs) +
  labs(
    title = "Lasso model performance across regularization penalties",
    subtitle = "Performance metrics can be used to identify the best penalty"
  )
m_lp <- collect_predictions(lasso_rs)
kable(head(m_lp), format = "simple", caption="Lasso Predictions\\label{tbl:lasso_predictions}")
m_lp %>%
  # mutate(.pred_class=as.numeric(levels(.pred_class)[.pred_class])) %>%
  group_by(id) %>%
  roc_curve(truth = label, .pred_0) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Lasso model Label 0",
    subtitle = "Each resample fold is shown in a different color"
  )
m_lp %>%
  group_by(id) %>%
  roc_curve(truth = label, .pred_1) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Lasso model Label 1",
    subtitle = "Each resample fold is shown in a different color"
  )
# Best ROC_AUC 
blm_best_roc <- max(m_blr$mean)  

# Best Accuracy
blm_best_acc <- max(m_bla$mean) 
results_table <- tibble(Index = "1", Method = "BLM", Accuracy = blm_best_acc, Loss = NA)

kable(results_table, "simple",caption="Baseline Linear Model Results\\label{tbl:blm_results_table}")
# %>%
#     kable_styling(latex_options=c("HOLD_position"), font_size=7)
rm(amazon_folds, amazon_rec, amazon_split, amazon_test, amazon_train, amazon_wf, lambda_grid, lasso_rs, lasso_spec, sparse_bp)

gc()

amazon_subset_train <- readr::read_csv("amazon_review_polarity_csv/amazon_subset_train.csv")

amazon_train <- amazon_subset_train

max_words <- 2e4
max_length <- 30
mystopwords <- c("s", "t", "m", "ve", "re", "d", "ll")

amazon_rec <- recipe(~ text, data = amazon_subset_train) %>%
  step_text_normalization(text) %>%
  step_tokenize(text) %>%
  step_stopwords(text, 
                 stopword_source = "stopwords-iso",
                 custom_stopword_source = mystopwords) %>%
  step_tokenfilter(text, max_tokens = max_words) %>%
  step_sequence_onehot(text, sequence_length = max_length)

amazon_rec
amazon_prep <-  prep(amazon_rec)

amazon_subset_train <- bake(amazon_prep, new_data = NULL, composition = "matrix")
dim(amazon_subset_train)
amazon_prep %>% tidy(5) %>% head(10)
# library(keras)
# use_python(python = "/c/Users/bijoor/.conda/envs/tensorflow-python/python.exe", required = TRUE)
# use_condaenv(condaenv = "tensorflow-python", required = TRUE)

dense_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = 12,
                  input_length = max_length) %>%
  layer_flatten() %>%
  layer_layer_normalization() %>%
  # layer_dropout(0.1) %>%
  layer_dense(units = 64) %>%
  # layer_activation_leaky_relu() %>%
  layer_activation_relu() %>%
  layer_dense(units = 1, activation = "sigmoid")

dense_model
# opt <- optimizer_adam(lr = 0.0001, decay = 1e-6)
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6)
opt <- optimizer_sgd()
dense_model %>% compile(
  optimizer = opt,
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
dense_history <- dense_model %>%
  fit(
    x = amazon_subset_train,
    y = amazon_train$label,
    batch_size = 1024,
    epochs = 50,
    initial_epoch = 0,
    validation_split = 0.20,
    verbose = 2
  )

dense_history

plot(dense_history)
set.seed(234)
amazon_val_eval <- validation_split(amazon_train, strata = label)
# amazon_val_eval << I am getting a pandoc stack error printing this
amazon_analysis <- bake(amazon_prep, new_data = analysis(amazon_val_eval$splits[[1]]),
                        composition = "matrix")
dim(amazon_analysis)
amazon_assess <- bake(amazon_prep, new_data = assessment(amazon_val_eval$splits[[1]]),
                      composition = "matrix")
dim(amazon_assess)
label_analysis <- analysis(amazon_val_eval$splits[[1]]) %>% pull(label)
label_assess <- assessment(amazon_val_eval$splits[[1]]) %>% pull(label)
dense_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = 12,
                  input_length = max_length) %>%
  layer_flatten() %>%
  layer_layer_normalization() %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 64) %>%
  layer_activation_relu() %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 128) %>%
  layer_activation_relu() %>%
  layer_dense(units = 128) %>%
  layer_activation_relu() %>%
  layer_dense(units = 1, activation = "sigmoid")

opt <- optimizer_adam(lr = 0.0001, decay = 1e-6)
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6)
# opt <- optimizer_sgd()
dense_model %>% compile(
  optimizer = opt,
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
val_history <- dense_model %>%
  fit(
    x = amazon_analysis,
    y = label_analysis,
    batch_size = 2048,
    epochs = 20,
    validation_data = list(amazon_assess, label_assess),
    verbose = 2
  )

val_history
plot(val_history)
keras_predict <- function(model, baked_data, response) {
  predictions <- predict(model, baked_data)[, 1]
  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 < 0.5, 0, 1),
    label = response) %>% 
    mutate(across(c(label, .pred_class), 
                  ~ factor(.x, levels = c(1, 0))))
}
val_res <- keras_predict(dense_model, amazon_assess, label_assess)
# head(val_res)
kable(head(val_res), format="simple", caption="DNN Model 2 Predictions using validation data\\label{tbl:val_res}")
m1 <- metrics(val_res, label, .pred_class)
kable(m1, format = "simple", caption="DNN Model 2 Metrics using Validation data\\label{tbl:val_res_metrics}")
val_res %>%
  conf_mat(label, .pred_class) %>%
  autoplot(type = "heatmap")
val_res %>%
  roc_curve(truth = label, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for Amazon Reviews"
  )
# Best DNN accuracy
dnn_best_acc <- max(val_history$metrics$val_accuracy) 

# Lowest DNN Loss
dnn_lowest_loss <- min(val_history$metrics$val_loss)
results_table <- bind_rows(results_table,
                           tibble(Index = "2",
                                  Method = "DNN",
                                  Accuracy = dnn_best_acc,
                                  Loss = dnn_lowest_loss))

kable(results_table, "simple",caption="DNN Model Results\\label{tbl:dnn_results_table}")
# %>%
#     kable_styling(latex_options=c("HOLD_position"), font_size=7)
simple_cnn_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 16,
                  input_length = max_length) %>%
  layer_batch_normalization() %>%
  layer_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filter = 64, kernel_size = 3, activation = "relu") %>%
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

simple_cnn_model
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6)
# opt <- optimizer_adam()
# opt <- optimizer_sgd()
opt <- optimizer_adam(lr = 0.0001, decay = 1e-6)
simple_cnn_model %>% compile(
  optimizer = opt,
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
simple_cnn_val_history <- simple_cnn_model %>%
  fit(
    x = amazon_analysis,
    y = label_analysis,
    batch_size = 1024,
    epochs = 7,
    initial_epoch = 0,
    validation_data = list(amazon_assess, label_assess),
    verbose = 2
  )

simple_cnn_val_history
plot(simple_cnn_val_history)
simple_cnn_val_res <- keras_predict(simple_cnn_model, amazon_assess, label_assess)
# head(simple_cnn_val_res)
kable(head(simple_cnn_val_res), format="simple", caption="CNN Model Predictions using validation data\\label{tbl:simple_cnn_val_res}")
m2 <- metrics(simple_cnn_val_res, label, .pred_class)
kable(m2, format="simple", caption="CNN Model Metrics using validation data\\label{tbl:simple_cnn_val_res_metrics}")
simple_cnn_val_res %>%
  conf_mat(label, .pred_class) %>%
  autoplot(type = "heatmap")
simple_cnn_val_res %>%
  roc_curve(truth = label, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for Amazon Reviews"
  )
# Best CNN accuracy
cnn_best_acc <- max(simple_cnn_val_history$metrics$val_accuracy) 

# Lowest CNN Loss
cnn_lowest_loss <- min(simple_cnn_val_history$metrics$val_loss)
results_table <- bind_rows(results_table,
                           tibble(Index = "3",
                                  Method = "CNN",
                                  Accuracy = cnn_best_acc,
                                  Loss = cnn_lowest_loss))

kable(results_table, "simple",caption="CNN Model Results\\label{tbl:cnn_results_table}")
# %>%
#     kable_styling(latex_options=c("HOLD_position"), font_size=7)
sep_cnn_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 16,
                  input_length = max_length) %>%
  # layer_batch_normalization() %>%
  layer_dropout(0.2) %>%
  layer_separable_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
  layer_separable_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_separable_conv_1d(filter = 64, kernel_size = 5, activation = "relu") %>%
  layer_separable_conv_1d(filter = 64, kernel_size = 5, activation = "relu") %>%
  layer_global_average_pooling_1d() %>% 
  layer_dropout(0.2) %>%
  # layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

sep_cnn_model
# opt <- optimizer_sgd(lr = 0.001, decay = 1e-6)
# opt <- optimizer_adam()
# opt <- optimizer_sgd()
opt <- optimizer_adam(lr = 0.0001, decay = 1e-6)
sep_cnn_model %>% compile(
  optimizer = opt,
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
sep_cnn_val_history <- sep_cnn_model %>%
  fit(
    x = amazon_analysis,
    y = label_analysis,
    batch_size = 128,
    epochs = 20,
    initial_epoch = 0,
    validation_data = list(amazon_assess, label_assess),
    callbacks = list(callback_early_stopping(
        monitor='val_loss', patience=2)),
    verbose = 2
  )

sep_cnn_val_history
plot(sep_cnn_val_history)
sep_cnn_val_res <- keras_predict(sep_cnn_model, amazon_assess, label_assess)
# head(sep_cnn_val_res)
kable(head(sep_cnn_val_res), format="simple", caption="sepCNN Model Predictions using validation data\\label{tbl:sep_cnn_val_res}")
m3 <- metrics(sep_cnn_val_res, label, .pred_class)
kable(m3, format="simple", caption="sepCNN Model Metrics using validation data\\label{tbl:sep_cnn_val_res_metrics}")
sep_cnn_val_res %>%
  conf_mat(label, .pred_class) %>%
  autoplot(type = "heatmap")
sep_cnn_val_res %>%
  roc_curve(truth = label, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for Amazon Reviews"
  )
# Best sepCNN accuracy
sep_cnn_best_acc <- max(sep_cnn_val_history$metrics$val_accuracy) 

# Lowest sepCNN Loss
sep_cnn_lowest_loss <- min(sep_cnn_val_history$metrics$val_loss)
results_table <- bind_rows(results_table,
                           tibble(Index = "4",
                                  Method = "sepCNN",
                                  Accuracy = sep_cnn_best_acc,
                                  Loss = sep_cnn_lowest_loss))

kable(results_table, "simple",caption="sepCNN Model Results\\label{tbl:sep_cnn_results_table}")
# %>%
#     kable_styling(latex_options=c("HOLD_position"), font_size=7)
library(keras)
library(tfdatasets)
library(reticulate)
library(tidyverse)
library(lubridate)
library(tfhub)

# A dependency of the preprocessing for BERT inputs
# pip install -q -U tensorflow-text
import("tensorflow_text")

# You will use the AdamW optimizer from tensorflow/models.
# pip install -q tf-models-official
# to create AdamW optimizer
o_nlp <- import("official.nlp")

Sys.setenv(TFHUB_CACHE_DIR="C:/Users/bijoor/.cache/tfhub_modules")
Sys.getenv("TFHUB_CACHE_DIR")

train_file_path <- file.path("amazon_review_polarity_csv/amazon_train.csv")

batch_size <- 32

train_dataset <- make_csv_dataset(
  train_file_path, 
  field_delim = ",",
  batch_size = batch_size,
  column_names = list("label", "title", "text"),
  label_name = "label",
  select_columns = list("label", "text"),
  num_epochs = 1
)


train_dataset %>%
  reticulate::as_iterator() %>% 
  reticulate::iter_next() #%>% 
  # reticulate::py_to_r()

# ----------------------

val_file_path <- file.path("amazon_review_polarity_csv/amazon_val.csv")

val_dataset <- make_csv_dataset(
  val_file_path, 
  field_delim = ",",
  batch_size = batch_size,
  column_names = list("label", "title", "text"),
  label_name = "label",
  select_columns = list("label", "text"),
  num_epochs = 1
)


val_dataset %>%
  reticulate::as_iterator() %>% 
  reticulate::iter_next()

# -----------------------------------

test_file_path <- file.path("amazon_review_polarity_csv/amazon_test.csv")


test_dataset <- make_csv_dataset(
  test_file_path, 
  field_delim = ",",
  batch_size = batch_size,
  column_names = list("label", "title", "text"),
  label_name = "label",
  select_columns = list("label", "text"),
  num_epochs = 1
)

test_dataset %>%
  reticulate::as_iterator() %>% 
  reticulate::iter_next()

rm(amazon_orig_train, amazon_orig_test, amazon_train, amazon_val)
rm(ids_train, train_file_path, test_file_path, val_file_path)
bert_preprocess_model <- layer_hub(
  handle = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", trainable = FALSE, name='preprocessing'
)
bert_model <- layer_hub(
  handle = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
  trainable = TRUE, name='BERT_encoder'
)
input <- layer_input(shape=shape(), dtype="string", name='text')

output <- input %>%
  bert_preprocess_model() %>%
  bert_model %$%
  pooled_output %>% 
  layer_dropout(0.1) %>%
  # layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid", name='classifier')

# summary(model)
model <- keras_model(input, output)
epochs = 5
steps_per_epoch <- 2e6
num_train_steps <- steps_per_epoch * epochs
num_warmup_steps <- as.integer(0.1*num_train_steps)

init_lr <- 3e-5
opt <- o_nlp$optimization$create_optimizer(init_lr=init_lr,
                                     num_train_steps=num_train_steps,
                                  num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
  )

summary(model)
# 10000 will take approx 40 mins per epoch on my gpu/mem etc
# 1000 will take approx 4 mins per epoch on my gpu/mem etc

tr_count <- 10000
take_tr <- 0.8 * tr_count
train_slice <- train_dataset  %>% 
  dataset_shuffle_and_repeat(buffer_size = take_tr * batch_size) %>% 
  dataset_take(take_tr)

take_val <- 0.2 * tr_count
val_slice <- val_dataset  %>% 
  dataset_shuffle_and_repeat(buffer_size = take_val * batch_size) %>% 
  dataset_take(take_val)
epochs <- 5
seed = 42

history <- model %>% 
  fit(
    train_slice,
    epochs = epochs,
    validation_data = val_slice,
    initial_epoch = 0,
    verbose = 2
  )
plot(history)
model %>% evaluate(test_dataset)
test_slice <- test_dataset  %>% 
  dataset_take(100)

model %>% evaluate(test_slice)
# Best BERT accuracy
bert_best_acc <-  max(history$metrics$val_accuracy) 

# Lowest BERT Loss
bert_lowest_loss <- min(history$metrics$val_loss)
results_table <- bind_rows(results_table,
                           tibble(Index = "5",
                                  Method = "BERT",
                                  Accuracy = bert_best_acc,
                                  Loss = bert_lowest_loss))

kable(results_table, "simple",caption="BERT Model Results\\label{tbl:bert_results_table}")
# %>%
#     kable_styling(latex_options=c("HOLD_position"), font_size=7)
# Stop the clock
# proc.time() - ptm
Sys.time()
	knitr::knit_exit()
```

---  

\newpage

Terms like generate\index{generate} and some\index{others} will also show up.

\printindex



```r
	knitr::knit_exit()
```

