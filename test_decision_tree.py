import csv
from c45.decision_tree_util import read_csv_data

data = read_csv_data('data/iris.test.data.txt')

with open('model/model.txt', 'r') as f:
    model_txt = f.read()

exec(model_txt)

correct_acc = 0
total_acc = 0

with open('result/model_evaluation.csv', mode='w') as result_f:
    res_writer = csv.writer(result_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for line in data:
        result = predict(float(line[0]), float(line[1]), float(line[2]), float(line[3]))
        line.append(result)
        line.append(result == line[4])
        res_writer.writerow(line)
        total_acc = total_acc + 1

        if result == line[4]:
            correct_acc = correct_acc + 1

print('Correct: ' + str(correct_acc))
print('Total: ' + str(total_acc))