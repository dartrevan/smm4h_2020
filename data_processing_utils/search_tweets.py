from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pymorphy2
import json
import re


def lemmatize(document, morph_analyzer):
    document_tokenized = word_tokenize(document)
    return ' '.join([morph_analyzer.parse(token)[0].normal_form for token in document_tokenized])


def compose_search_string(vocab):
    s_elements = []
    for vocab_entry in vocab:
        vocab_entry = re.escape(vocab_entry)
        s_elements.append(vocab_entry)
    return '|'.join(s_elements)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_ds', type=str, help='Path to the json formatted documents')
    parser.add_argument('--vocab', type=str, help='Path to the filtering vocab')
    parser.add_argument('--save_ds', type=str, help='Path to save filtered documents')
    parser.add_argument('--lemmatize', action='store_true', help='Use lemmatization')
    parser.add_argument('--lowercase', action='store_true', help='Lowercased search')

    args = parser.parse_args()

    with open(args.input_ds, encoding='utf-8') as input_stream:
        dataset = [json.loads(line) for line in input_stream]

    with open(args.vocab, encoding='utf-8') as input_stream:
        vocab = [line.strip() for line in input_stream]

    morph_analyzer = pymorphy2.MorphAnalyzer()
    if args.lemmatize:
        vocab = [lemmatize(vocab_entry, morph_analyzer) for vocab_entry in vocab]
    if args.lowercase:
        vocab = [vocab_entry.lower() for vocab_entry in vocab]
    search_string = compose_search_string(vocab)

    with open(args.save_ds, 'w', encoding='utf-8') as output_stream:
        for document in tqdm(dataset):
            document_text = document['text']
            if args.lemmatize:
                document_text = lemmatize(document_text, morph_analyzer)
            if args.lowercase:
                document_text = document_text.lower()
            if re.search(search_string, document_text):
                output_stream.write(json.dumps(document) + '\n')



