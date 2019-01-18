#!/usr/bin/env python

'''
Simple tagger service using spaCy.

Author:     Ben Olayinka
Version:    2017-10-12
'''

from argparse import ArgumentParser
from cgi import FieldStorage
import os
from os.path import dirname, join as path_join

import spacy

try:
    from json import dumps
except ImportError:
    # likely old Python; try to fall back on ujson in brat distrib
    from sys import path as sys_path
    sys_path.append(path_join(dirname(__file__), '../../server/lib/ujson'))
    from ujson import dumps

from sys import stderr
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

### Constants
ARGPARSER = ArgumentParser(description='XXX')#XXX:
ARGPARSER.add_argument('-p', '--port', type=int, default=47111,
        help='port to run the HTTP service on (default: 47111)')
ARGPARSER.add_argument('-m', '--model', default='en',
        help='spaCy model to load (default: "en", trained: "ner")')
TAGGER = None

class spacyTaggerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print >> stderr, 'Received request'
        field_storage = FieldStorage(
                headers=self.headers,
                environ={
                    'REQUEST_METHOD':'POST',
                    'CONTENT_TYPE':self.headers['Content-Type'],
                    },
                fp=self.rfile)

        global TAGGER
        json_dic = tag_to_json(TAGGER,field_storage.value.decode('utf8'))

        # Write the response
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()

        self.wfile.write(dumps(json_dic))
        print >> stderr, ('Generated %d annotations' % len(json_dic))

    def log_message(self, format, *args):
        return # Too much noise from the default implementation

def tag_to_json(nlp, text):
    doc = nlp(text)
    annotations = {}

    def _add_ann(start, end, _type):
        line = text[start:end]
        if line.strip('\n'):
            annotations[len(annotations)] = {
                    'type': _type,
                    'offsets': ((start, end), ),
                    'texts': ((text[start:end]), ),
                    }
        else:
            print >> stderr, ('Ignoring annotation:', (repr(line), _type, start, end))

    for ent in doc.ents:
        _add_ann(ent.start_char, ent.end_char, ent.label_)

    return annotations

def main(args):
    argp = ARGPARSER.parse_args(args[1:])
    if argp.model == 'en':
        MODEL = 'en'
        print >> stderr, 'Loading '+ MODEL+'\n',
    elif argp.model == 'ner':
        dir_path = os.path.dirname(os.path.realpath(__file__))
        MODEL = os.path.join(dir_path,'ner')
        print >> stderr, 'Loading '+ MODEL+'\n',
    else:
        MODEL = 'en'
        print >> stderr, 'Unknown model. Loading "en"\n',

    print >> stderr, "WARNING: Don't use this in a production environment!"

    print >> stderr, 'Starting spaCy',
    global TAGGER
    TAGGER = spacy.load(MODEL)
    print >> stderr, 'Done!'

    server_class = HTTPServer
    httpd = server_class(('localhost', argp.port), spacyTaggerHandler)
    print >> stderr, 'spacy tagger service started'
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print >> stderr, 'spacy tagger service stopped'

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
