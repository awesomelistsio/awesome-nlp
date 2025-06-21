# Awesome Natural Language Processing (NLP) [![Awesome Lists](https://srv-cdn.himpfen.io/badges/awesome-lists/awesomelists-flat.svg)](https://github.com/awesomelistsio/awesome)

[![Ko-Fi](https://srv-cdn.himpfen.io/badges/kofi/kofi-flat.svg)](https://ko-fi.com/awesomelists) &nbsp; [![PayPal](https://srv-cdn.himpfen.io/badges/paypal/paypal-flat.svg)](https://www.paypal.com/donate/?hosted_button_id=3LLKRXJU44EJJ) &nbsp; [![Stripe](https://srv-cdn.himpfen.io/badges/stripe/stripe-flat.svg)](https://tinyurl.com/e8ymxdw3) &nbsp; [![X](https://srv-cdn.himpfen.io/badges/twitter/twitter-flat.svg)](https://x.com/ListsAwesome) &nbsp; [![Facebook](https://srv-cdn.himpfen.io/badges/facebook-pages/facebook-pages-flat.svg)](https://www.facebook.com/awesomelists)

> A curated list of awesome frameworks, libraries, tools, datasets, tutorials, and research papers for Natural Language Processing (NLP). This list covers a variety of NLP tasks, from text processing and tokenization to state-of-the-art language models and applications like sentiment analysis and machine translation.

## Contents

- [Frameworks and Libraries](#frameworks-and-libraries)
- [Text Processing and Tokenization](#text-processing-and-tokenization)
- [Pretrained Language Models](#pretrained-language-models)
- [NLP Tasks](#nlp-tasks)
- [Tools and Applications](#tools-and-applications)
- [Datasets](#datasets)
- [Research Papers](#research-papers)
- [Learning Resources](#learning-resources)
- [Books](#books)
- [Community](#community)
- [Contribute](#contribute)
- [License](#license)

## Frameworks and Libraries

- [Hugging Face Transformers](https://huggingface.co/transformers/) - A comprehensive library of state-of-the-art NLP models like BERT, GPT, and RoBERTa.
- [spaCy](https://spacy.io/) - An open-source library for advanced natural language processing in Python.
- [NLTK (Natural Language Toolkit)](https://www.nltk.org/) - A comprehensive library for text processing and analysis.
- [Stanford NLP](https://stanfordnlp.github.io/CoreNLP/) - A suite of NLP tools developed by the Stanford NLP Group.
- [AllenNLP](https://allennlp.org/) - An open-source NLP research library built on top of PyTorch.
- [TextBlob](https://textblob.readthedocs.io/) - A simple library for processing textual data in Python.

## Text Processing and Tokenization

- [Moses Tokenizer](https://github.com/moses-smt/mosesdecoder) - A widely used tokenizer for machine translation tasks.
- [BPE (Byte Pair Encoding)](https://arxiv.org/abs/1508.07909) - A subword tokenization technique used by models like GPT and BERT.
- [SentencePiece](https://github.com/google/sentencepiece) - A language-independent tokenization and text processing library.
- [RegexpTokenizer (NLTK)](https://www.nltk.org/api/nltk.tokenize.html) - A tokenizer that uses regular expressions to split text into tokens.
- [spaCy Tokenizer](https://spacy.io/usage/linguistic-features#tokenization) - A fast and efficient tokenizer integrated within the spaCy library.

## Pretrained Language Models

- [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805) - A Transformer-based model for a variety of NLP tasks.
- [GPT-3 (Generative Pre-trained Transformer 3)](https://arxiv.org/abs/2005.14165) - A powerful generative language model by OpenAI.
- [RoBERTa](https://arxiv.org/abs/1907.11692) - An optimized variant of BERT, focusing on robustly optimized pretraining.
- [T5 (Text-to-Text Transfer Transformer)](https://arxiv.org/abs/1910.10683) - A model that treats every NLP task as a text-to-text problem.
- [XLNet](https://arxiv.org/abs/1906.08237) - A generalized autoregressive pretraining model that outperforms BERT on several tasks.
- [DistilBERT](https://arxiv.org/abs/1910.01108) - A smaller, faster, and lighter version of BERT.

## NLP Tasks

- **Sentiment Analysis**: The process of determining the sentiment (positive, negative, or neutral) of a text.
  - [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)
  - [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- **Named Entity Recognition (NER)**: Identifying and classifying entities in text (e.g., names, dates).
  - [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
  - [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.html)
- **Machine Translation**: Translating text from one language to another.
  - [OpenNMT](https://opennmt.net/) - A neural machine translation framework.
  - [Fairseq](https://fairseq.readthedocs.io/en/latest/) - A Facebook AI research framework for sequence-to-sequence models.
- **Text Summarization**: Generating a concise summary of a given text.
  - [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation](https://arxiv.org/abs/1910.13461)
  - [PEGASUS](https://arxiv.org/abs/1912.08777) - A pre-trained model specifically designed for text summarization.

## Tools and Applications

- [Gensim](https://radimrehurek.com/gensim/) - A Python library for topic modeling and document similarity.
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) - A suite of NLP tools for linguistic analysis.
- [FastText](https://fasttext.cc/) - A library for efficient text classification and representation learning.
- [Polyglot](https://polyglot.readthedocs.io/) - A multilingual NLP toolkit supporting various languages.
- [LexRank](https://github.com/crabcamp/lexrank) - A text summarization library using graph-based ranking algorithms.

## Datasets

- [GLUE Benchmark](https://gluebenchmark.com/) - A collection of resources for evaluating natural language understanding systems.
- [SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) - A dataset for reading comprehension and question answering tasks.
- [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) - A dataset for named entity recognition.
- [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) - A dataset for sentiment analysis.
- [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) - A collection of high-quality text from Wikipedia for language modeling tasks.

## Research Papers

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - The paper that introduced the Transformer architecture, revolutionizing NLP.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805) - The introduction of the BERT model.
- [GloVe: Global Vectors for Word Representation (2014)](https://nlp.stanford.edu/pubs/glove.pdf) - A model for generating word embeddings.
- [Word2Vec: Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/abs/1301.3781) - The introduction of Word2Vec, a method for learning word embeddings.
- [ELMo: Deep Contextualized Word Representations (2018)](https://arxiv.org/abs/1802.05365) - A model for contextual word embeddings.

## Learning Resources

- [Coursera: Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing) - A comprehensive course on NLP by Deeplearning.ai.
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) - A popular university course on NLP.
- [Fast.ai NLP Course](https://www.fast.ai/) - A practical course on NLP using the fastai library.
- [Hugging Face Tutorials](https://huggingface.co/course/chapter1) - Official tutorials for using the Hugging Face NLP library.

## Books

- *Speech and Language Processing* by Daniel Jurafsky and James H. Martin - A comprehensive textbook on NLP.
- *Natural Language Processing with Python* by Steven Bird, Ewan Klein, and Edward Loper - An introduction to NLP using Python.
- *Deep Learning for Natural Language Processing* by Palash Goyal, Sumit Pandey, and Karan Jain - A book covering deep learning techniques in NLP.

## Community

- [Reddit: r/NLP](https://www.reddit.com/r/LanguageTechnology/) - A subreddit for discussions on natural language processing.
- [Hugging Face Community](https://discuss.huggingface.co/) - A forum for discussing the Hugging Face NLP library.
- [NLP Summit](https://www.nlpsummit.org/) - An annual conference focused on NLP research and applications.

## Contribute

Contributions are welcome!

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)](http://creativecommons.org/licenses/by-sa/4.0/)
