# Molecular-graph-BERT 

semi-supervised learning for molecular property prediction

requried package: tensorflow==2.3.0,rdkit==2020.03.2,numpy==1.18.5,pandas==1.1.0, openbabel==2.3.1

-- pretrain:
    contains the codes for masked atom prediction pre-training task.
    
-- classification and regression:
    contain the code for fune-tuning on specified tasks
    
-- dataset:
    contain the code to building dataset for pre-traing and fine-tuning 
    
-- utils:
    contain the code to convert molecules to graphs
    
 --data:
    data used for pretraining and fine-tuning 
    
    
User should first unzip the data file and place it in the right place. Then pre-training the MG-BERT for 10~20 epoch. After that, the classification or the regression file is used to predict specific molecular property.
