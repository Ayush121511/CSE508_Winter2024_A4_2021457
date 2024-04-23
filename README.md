# IR Assignment 4 Report

## Model Trained - results
## Assumptions 
To reduce complexity, we have taken 50000 samples from the dataset.
## Step-wise Explanation of the code
1. Data Import Configuration:
This section establishes the setup for importing data from Kaggle. It includes defining constants
and functions essential for downloading and extracting data from specified URLs.
2. Data Retrieval and Unpacking:
The script retrieves the designated dataset from Kaggle via the URLs provided, identifying
whether the file is in ZIP or TAR format and appropriately extracting its contents.
3. Data Loading and Initial Processing:
Following retrieval and extraction, the script loads the dataset into a Pandas DataFrame. It
initially filters the dataset to the first 20,000 rows for subsequent processing.
4. Text Data Preprocessing:
The script executes a series of text preprocessing steps, encompassing lowercase conversion,
tokenization, punctuation removal, stopword elimination, and lemmatization using NLTK tools to
standardize words.
5. Stemming and Lemmatization:
A function called `lemmatize_text` is defined to perform lemmatization on text tokens. This
function is then applied to the 'processed_text' column of the DataFrame.
6. Removal of Duplicates:
Duplicate entries are identified and removed based on the preprocessed text column to ensure
uniqueness in the dataset.
7. Library Imports:
Essential libraries are imported, including pandas for data manipulation, numpy for numerical
operations, nltk for natural language processing tasks, and re for regular expressions.
8. Acquisition of NLTK Resources:
Necessary NLTK resources such as word tokenizer, stopwords, and WordNet corpus are
downloaded to facilitate text processing.
9. Data Cleansing:
The script conducts data cleansing procedures, involving the removal of rows with NaN values,
lowercase conversion, tokenization, punctuation removal, stopword elimination, and token
concatenation.
10. Data Splitting and Structure Creation:
The dataset is partitioned into training and testing sets using a 75-25 split ratio. Pandas
DataFrames are constructed for both sets, incorporating processed text and summaries.
11. Index Resetting (Optional):
An optional step to reset the index of DataFrames for consistency.
12. Text Combination and Formatting:
The script merges the first 100 words of processed text with the processed summary, separated
by 'TEXT:' and 'SUMMARY:' respectively. Additionally, an 'END' marker is appended to signify
the sequence's end.
13. Google Drive Integration (Optional):
For Google Colab execution, the script facilitates Google Drive mounting to enable file saving or
loading.
14. Sample Output to Text File:
Samples from the 'processed_text' column of the 'train_data' DataFrame are written to a text file,
each prefixed with "Sample {index}:".
15. Custom Dataset Class Definition:
A custom dataset class named `CustomDataset`, inheriting from `torch.utils.data.Dataset`, is
defined. This class accommodates file path, tokenizer, and block size as input parameters.
16. Dataset Loading Function Modification:
The `load_dataset` function is adjusted to utilize the custom dataset class (`CustomDataset`),
initializing it with provided parameters.
17. Data Collation Function Establishment:
A function named `load_data_collator` is introduced to create a data collator tailored for
language modeling tasks, configuring parameters such as the tokenizer and whether masked
language modeling (MLM) is employed.
18. Training Routine Definition:
A function named `train` is outlined to execute fine-tuning of the GPT-2 model on the custom
dataset. It encompasses dataset loading via `load_dataset`, data collation via
`load_data_collator`, model initialization, training arguments setup, instantiation of a `Trainer`
object, training execution, and trained model preservation.
19. Training Parameter Configuration:
Parameters essential for training, including file paths, model names, output directories, batch
sizes, epoch numbers, and save steps, are defined.
20. Training Invocation:
The `train` function is invoked with the specified parameters to commence the training process.
21. Function Definitions:
Functions to load the model (`load_model`), load the tokenizer (`load_tokenizer`), and generate
text (`generate_text`) are outlined.
22. Text Generation Loop:
A loop is established to generate summaries for each sequence in the 'test_data' DataFrame
using the fine-tuned GPT-2 model. Summaries are generated with a maximum length
determined by the sequence length plus 100 words.
23. Evaluation:
An elucidation is provided on the process of generating summaries for test data using the
fine-tuned GPT-2 model, iterating through sequences and appending generated summaries to a
list until all sequences are processed.
## Results
Rogue scores
The test set which consisted of 12500 samples had the following mean ROUGE score for all the
samples -

'rouge-1': {RECALL: 0.7946693121693122, PRECISION: 0.818880952380952, F1-SCORE:
0.8007613221107203}

'rouge-2': {RECALL: 0.6603968253968252, PRECISION: 0.6641111111111112, F1-SCORE:
0.656248801718073}

'rouge-l': {RECALL: 0.7946693121693122, PRECISION: 0.818880952380952, F1-SCORE:
0.8007613221107203}}
