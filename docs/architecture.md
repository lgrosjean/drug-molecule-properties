# Model architecture

As a reminder, there were two problems to solve:

1. The first model has to predict molecule property based on byte representation of a Smile (a 2048-size vector)
2. The second model has to predict the same property based on the raw Smile representation (string)

## Model 1

The model 1 architecture has been designed to deal with the Problem 1.

![model1](https://user-images.githubusercontent.com/34337781/107257136-a7280f80-6a3a-11eb-8dcf-e2fdc7aceef1.png)

The detail of the architecture:

| Layer         | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| Inputs        | a 2048-vector                                                 |
| Normalization | Normalize the inputs to centralize/standardize inputs         |
| Dense 1       | A fully-connected layer to reduce inputs information          |
| ReLU 1        | A reLU layer to activate the output of the Dense 1 layer      |
| Dense 2       | A second fully-connected layer to reduce information          |
| Dense 3       | A final dense with only one output to predict O or 1 property |

## Model 2

![model2](https://user-images.githubusercontent.com/34337781/107257774-5664e680-6a3b-11eb-9b68-9fddc7a4be6e.png)

The detail of the architecture:

| Layer              | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| Inputs             | a 1-size vector for string input                                               |
| Text Vectorization | A Layer which vectorizes the raw string inputs with different hyperparameters  |
| Embedding          | Embedding layer to convert the input vectorization into multi-vectorial vector |
| Dropout            | Dropout layer to avoid overfitting at the end of embedding layer               |
| Conv1D             | A 1D convolutional layer to compress information                               |
| MaxPooling1D       | MaxPooling at the end of convolutional layer                                   |
| LSTM               | LSTM to compress vectorize information                                         |
| Dense              | Final dense to predict 0 or 1 property                                         |