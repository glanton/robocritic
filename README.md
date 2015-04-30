# Robocritic
Robocritic is a Python-based machine learning binary text classifier. It uses a bag-of-words approach to parse text, and either a naive Bayes or random forest for training and classification.

The "critic" in Robocritc is simply a reflection of the datasets used in its development. It can potentially be applied toward any binary text classification problem. Emphasis on _binary_. Training datasets with more or fewer than two classes will not pass validation.

### Installation Instructions
Robocritic is written in Python 3 (specifically, 3.4.2) and will require an installed Python 3 interpreter to run. Installation instructions can be found here:

https://www.python.org/downloads/

Robocritic uses the Natural Langauge Toolkit (NLTK) to tokenize text into individual words. Installation instructions for NLTK can be found here:

http://www.nltk.org/install.html

Once NLTK is installed, Punkt Tokenizer Models will need to be installed from within NLTK. This can be done by opening a Python shell, and entering the following commands:

```sh
import nltk
nltk.download()
```

The NLTK Downloader will open, and Punkt Tokenizer Models can be found in the Models tab by the identifier "punkt".

### Use Instructions
Robocritic should be started by running __robocritic.py__ and uses a simple command line interface. There are three commands:

- __train__ : takes the arguments __algorithm__ ("nb" for naive Bayes or "rf" for random forest) and __file_name__ (which cannot contain whitespaces and must be in the __input_data__ folder)
- __classify__ : takes the argument __file_name__ (also no whitespaces and in __input_data__)
- __quit__ : bid farewell to Robocritic

The command __classify__ cannot be run until a classifier has been trained. Only one trained classifier can exist at a time, and they do not persist between sessions. Once data has been classified it is output to the __output_data__ folder with the file name "classified_data.csv".

All input data, whether for training or testing, must be in the format of a UTF-8 encoded CSV file.

The __parser.py__ and __rf.py__ files contain parameters than can be tuned to balance accuracy and performance. As a rule the naive Bayes algorithm runs much faster, and there is no way to halt or adjust training once it has begun.

### Due Credit
Robocritic is useless without training data, and could not have been tested and tuned without the following datasets:

- [Movie Review Data], provided by Bo Pang and Lillian Lee at Cornell University
- [Large Movie Review Dataset], provided by Andrew Mass at Stanford University

### Questions
Any questions, comments, or feedback is welcomed at [alexander_friberg@harvard.edu](mailto:alexander_friberg@harvard.edu).

[Movie Review Data]:http://www.cs.cornell.edu/people/pabo/movie-review-data/
[Large Movie Review Dataset]:http://ai.stanford.edu/~amaas/data/sentiment/