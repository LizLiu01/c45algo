from c45.decision_tree_util import *
import json

c, f = read_feature_metadata('metadata/iris.metadata.json')

training_data = read_csv_data('data/iris.train.data.txt')

training_data = prepare_training_data(training_data, f)

feat = list(map(lambda x: x['feature'], f))


tree = generate_tree(training_data, feat,feat, c)

with open('model/tree.json', 'w+') as j_file:
    json.dump(tree, j_file, indent=1)


with open('model/tree.json', 'r') as j_file:
    j_data = j_file.read()


obj_tree = json.loads(j_data)

print_tree(obj_tree)

output2 = list(map(lambda x: " ".join(x), output))
logic = "\n".join(output2)

feat = list(map(lambda x: x['feature'], f))

params = ','.join(feat)

model_txt = f"""
def predict({params}):
{logic}
"""

with open('model/model.txt', 'w+') as f:
    f.write(model_txt)
