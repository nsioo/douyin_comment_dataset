## 4 unsupervised method to get the sentiment score of Chinese TikTok comment


## Run
### Word2Vec

the dataset will be a txt file, one line represent one comment
`python3 w2v.py you_dataset.txt`

the output are 3 file, cleaned data (one line comment splted by space), model file and vec file

### PCA
you can use word vecter to do PCA experiment, then get the setiment score  
`run PCA_exp.ipynb`, change the path to `you .vec file`


### Cluter
you can use .model vector filt to do k-mean cluster experiment, then get the corresponding setiment score.  
`python3 kmean.py you_vec_file.vec`

### Ulra dense word embedding
you can use .vec file to do the ultra dense word embedding expriment, then get the 3 kind of setiment score with different seed number.
`python3 ulradense_sentiment/implemented_with_tf/densifier_v2.py you_vec_file.vec`
