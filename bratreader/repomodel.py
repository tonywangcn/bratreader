from annotateddocument import AnnotatedDocument
from annotationimporter import importann
from glob import glob
import fnmatch
import cPickle as pickle
import pathlib

import os


class RepoModel(object):
    """
    A class for modeling a local repository annotated with BRAT.

    Corpora annotated with Brat use 2 files for each document in the corpus:
    an .ann file containing the annotations in Brat Standoff Format
    (http://brat.nlplab.org/standoff.html), and a .txt file containing the
    actual text. This tool takes a folder containing pairs of these files as
    input, and creates a RepoModel object. This RepoModel object can be
    exported in an XML format, or operated on in memory.

    Currently the program ignores Notes, or # annotations.
    """

    def __init__(self, pathtorepo, recursive=False, cached = False):
        """
        Create a RepoModel object.

        :param pathtorepo: (string) the path to a local repository, which
        contains pairs of .ann and .txt files. No checking is done to guarantee
        that the repository is consistent.
        :param recursive: looks in subdirectories for .ann files
        :path cached: when True, the model is loaded from a cache file at the
        repo directory, if it exists. To clear a cache, run as False.
        :return: None
        """
        # Each document is saved as a textunit.

        self.documents = {}
        self.pathtorepo = pathtorepo
        self.pathtocache = os.path.join(self.pathtorepo,'model.cache')

        cache_loaded = False

        if os.path.isdir(pathtorepo):
            if cached:
                try:
                    with open(self.pathtocache, 'rb') as file:
                        self.__dict__.clear()
                        self.__dict__.update(pickle.load(file))
                    cache_loaded = True
                    print('Repomodel loaded from cache at %s', self.pathtocache)
                except IOError:
                    print('Warning: no cache found, building new cache')

            if not cache_loaded:
                    paths = []
                    if recursive:
                        for root, dirnames, filenames in os.walk(pathtorepo):
                            for filename in fnmatch.filter(filenames, '*.ann'):
                                paths.append(os.path.join(root, filename))
                    else:
                        paths.extend(glob("{0}/*.ann".format(pathtorepo)))
                        
                    for path in paths:
                        # The key of each document is the document name without
                        # the suffix (i.e. "001.ann" becomes "001")
                        key = os.path.splitext(path)[0]
                        key = os.path.split(key)[-1]
                        context = importann(path)
                        self.documents[key] = AnnotatedDocument(key, context)

                    self.cache_model()

        else:
            raise IOError(u"{0} is not a valid directory".format(pathtorepo))

    def save_xml(self, pathtofolder):
        """
        Export a RepoModel as a XML to the specified folder.

        If the folder doesn't exist, it is created.
        :param pathtofolder: (string) the path to the folder where the XML
        should be exported.
        """
        if not os.path.isdir(pathtofolder):
            os.mkdir(pathtofolder)

        for document in self.documents.values():
            path = os.path.join(pathtofolder,
                                "{0}.xml".format(str(document.key)))
            document.export_xml(path)

    def cache_model(self):
        '''simple cache for the lazy, save pickle in repo dir'''
        with open(self.pathtocache, 'wb') as file:
            pickle.dump(self.__dict__, file)