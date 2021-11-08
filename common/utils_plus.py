import json
import pickle
import csv
import jsonlines
from nltk.tokenize import sent_tokenize as nltk_tokenize

class jsonline_plus():
    @staticmethod
    def load(file):
        with jsonlines.open(file) as f:
            data = [x for x in f]
            return data
    @staticmethod
    def dump(questions,file):
        with jsonlines.open(file, "w") as f:
            f.write_all(questions)
            return f


class json_plus():
    @staticmethod
    def load(infile):
        with open(infile, 'r') as df:
            f = json.load(df)
        return f

    @staticmethod
    def dump(payload, outfile):
        with open(outfile, 'w') as df:
            f = json.dump(payload, df)
        return f


class pickle_plus():
    @staticmethod
    def load(infile):
        with open(infile, 'rb') as df:
            f = pickle.load(df)
        return f

    @staticmethod
    def dump(payload, outfile):
        with open(outfile, 'wb') as df:
            f = pickle.dump(payload, df)
        return f

class csv_plus():
    @staticmethod
    def dump(content, write_path):
        csv_columns = list(content[0].keys())
        with open(write_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in content:
                writer.writerow(data)
    
    @staticmethod
    def load(infile):
        return_row = []
        with open(infile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                return_row.append(row)
        return return_row

class tsv_plus():
    @staticmethod
    def load(infile):
        return_row = []
        tsv_file = open(infile)
        read_tsv = csv.reader(tsv_file,delimiter = "\t")
        for row in read_tsv:
            return_row.append(row)
        return return_row