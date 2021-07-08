#!/usr/bin/env python

import argparse
import itertools
import logging
import os
import sys

import deeppavlov
import spacy
from spacy.lang.ru import Russian
import tqdm


MAX_LENGTH = 500000000
GROUPER_SIZE = 8


def grouper(iterable, n, fillvalue = None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue = fillvalue)


def main(args):
    ud_model = deeppavlov.build_model("ru_syntagrus_joint_parsing")
    with open(args.input, "r") as source:
        # TODO: explain what you're doing here...
        text = source.read().replace("\n", " ")
    # TODO: fix import
    nlp = Russian()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.max_length = MAX_LENGTH
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    with open(args.output, "w") as sink:
        for group in tqdm.tqdm(grouper(sentences, GROUPER_SIZE)):
            try:
                for sent in ud_model(group):
                    print(sent, file=sink)
                    print(file=sink)
            except Exception as err:
                logging.warning("Ignoring exception: %s", err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    main(parser.parse_args())
