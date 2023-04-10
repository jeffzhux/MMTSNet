import csv
import tensorflow

with open('./file/submission.csv', 'r') as file:
    csvreader = csv.reader(file)
    for idx, row in enumerate(csvreader):
        print(row)
        print([type(r) for r in row])
        print('------------------------')
        if idx > 3:
            break

# converter = tf.lite.TFLiteConverter.from