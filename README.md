# TranxConalaBaseline
Reimplementing the Tranx system (Yin and Neubig 2018) for the Conala dataset and python3. 

To train: python experiment.py train path/to/conala/train/data.json
You must have a saved_models directory.

To test: python experiment.py test path/to/conala/test/data.json path/to/trained/model
You must have a saved_decode directory
