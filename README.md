# DB_AI


## how to run the project


first create a conda environment

conda create --name sentiment-env python=3.13.2



```
pip install -r requirements.txt
```

```
conda install -c conda-forge onnx onnxruntime
```


then run flask server

```
python app.py
```


then go to the url

```
http://127.0.0.1:5000
```



## the data analysis and model training code is in the sentiment-analysis.ipynb file



## Observations & model accuracy

### Vector Embedding Results

Neural Network : 71%

ML Models : 65-67 %

### TF-IDF results

Neural Network : 56% 

ML models : 56-65 %


## final verdict

- Vector Embedding-based sentiment analysis works better than the TF-IDF method, yielding better accuracy.
- Because the dataset is limited, there is a possibility that if the dataset size is increased, then the model would be better able to grasp the sentiments more accurately.
- Overall, LLMs are much more suited for multi-class sentiment classification, as they can understand nuances present in the mail, which embedding or traditional NLP methods may ignore right away.


