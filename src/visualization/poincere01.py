from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

# Read the sample relations file and train the model
relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
model = PoincareModel(train_data=relations, size=2)
model.train(epochs=50)
