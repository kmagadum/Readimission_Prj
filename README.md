# CONTENT
Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

# Citation to the original paper
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0195024#abstract0

# Link to the original paper’s repo (if applicable)
https://github.com/danicaxiao/CONTENT/blob/master/CONTENT.py

# Dependencies
Not as such. Mostly we need to check python version and compatible libraries for code.

# Data download instruction
Data is available with paper for training and testing model.Download data from the mentioned link.

# Preprocessing code + command (if applicable)
Not applicable

# Training code + command (if applicable)
We need to uncomment specific code so that data can be created for the Training model.
Uncomment dump_vocab() function in main() in file transform.py file.

# Evaluation code + command (if applicable)
We need to uncomment specific code so that data can be created for the evaluation of a model.
Uncomment dump_vocab() function in main() in file transform.py file.

# Pretrained model (if applicable)
Not applicable.

# Table of results (no need to include additional experiments, but main reproducibility result should be included)
<!-------  RESULT ---------!>
-----------------------------------------
|  Model  |		PR-AUC      |	ACC     |
-----------------------------------------
| CONTENT |	   0.4307 		|	0.7360	|
-----------------------------------------
