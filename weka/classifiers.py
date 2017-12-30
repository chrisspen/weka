#!/usr/bin/python
"""
2010.2.19 CKS
Light wrapper around Weka.

2011.3.6 CKS
Added method load_raw() to load a raw Weka model file directly.
Added support to retrieving probability distribution of a prediction.
"""
from __future__ import print_function

from collections import namedtuple
import gzip
import math
import os
import re
import shutil
import subprocess
from subprocess import Popen, PIPE
import sys
import tempfile
import time
import traceback
from decimal import Decimal

#https://pythonhosted.org/six/
from six.moves import cPickle as pickle
from six import string_types as basestring # pylint: disable=redefined-builtin
from six import iteritems
from six import u as unicode # pylint: disable=redefined-builtin

import arff
from arff import SPARSE, DENSE, Num, Nom, Int, Str, Date

DEFAULT_WEKA_JAR_PATH = '/usr/share/java/weka.jar:/usr/share/java/libsvm.jar'

BP = os.path.dirname(os.path.abspath(__file__))
CP = os.environ.get('WEKA_JAR_PATH', DEFAULT_WEKA_JAR_PATH)
for _cp in CP.split(os.pathsep):
    assert os.path.isfile(_cp), ("Weka JAR file %s not found. Ensure the " + \
        "file is installed or update your environment's WEKA_JAR_PATH to " + \
        "only include valid locations.") % (_cp,)

# http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
WEKA_CLASSIFIERS = [
'weka.classifiers.bayes.AODE',
'weka.classifiers.bayes.BayesNet',
'weka.classifiers.bayes.ComplementNaiveBayes',
'weka.classifiers.bayes.NaiveBayes',
'weka.classifiers.bayes.NaiveBayesMultinomial',
'weka.classifiers.bayes.NaiveBayesSimple',
'weka.classifiers.bayes.NaiveBayesUpdateable',
'weka.classifiers.functions.LeastMedSq',
'weka.classifiers.functions.LibSVM',
'weka.classifiers.functions.LinearRegression',
'weka.classifiers.functions.Logistic',
'weka.classifiers.functions.MultilayerPerceptron',
'weka.classifiers.functions.PaceRegression',
'weka.classifiers.functions.RBFNetwork',
'weka.classifiers.functions.SimpleLinearRegression',
'weka.classifiers.functions.SimpleLogistic',
'weka.classifiers.functions.SGD',
'weka.classifiers.functions.SMO',
'weka.classifiers.functions.SMOreg',
'weka.classifiers.functions.VotedPerceptron',
'weka.classifiers.functions.Winnow',
'weka.classifiers.lazy.IB1', 
'weka.classifiers.lazy.IBk',
'weka.classifiers.lazy.KStar',
'weka.classifiers.lazy.LBR',
'weka.classifiers.lazy.LWL',
'weka.classifiers.meta.RacedIncrementalLogitBoost',
'weka.classifiers.misc.HyperPipes',
'weka.classifiers.misc.VFI',
'weka.classifiers.rules.ConjunctiveRule',
'weka.classifiers.rules.DecisionTable',
'weka.classifiers.rules.JRip',
'weka.classifiers.rules.NNge',
'weka.classifiers.rules.OneR',
'weka.classifiers.rules.Prism',
'weka.classifiers.rules.PART',
'weka.classifiers.rules.Ridor',
'weka.classifiers.rules.ZeroR',
'weka.classifiers.trees.ADTree',
'weka.classifiers.trees.DecisionStump',
'weka.classifiers.trees.Id3',
'weka.classifiers.trees.J48',
'weka.classifiers.trees.LMT',
'weka.classifiers.trees.NBTree',
'weka.classifiers.trees.RandomForest',
'weka.classifiers.trees.REPTree',
]

def cmp(a, b): # pylint: disable=redefined-builtin
    return (a > b) - (a < b)

class _Helper(object):
    
    def __init__(self, name, ckargs, *args):
        self.name = name
        self.args = [name] + list(args)
        self.ckargs = ckargs
        
    def __call__(self, *args, **kwargs):
        args = list(self.args) + list(args)
        ckargs = self.ckargs
        ckargs.update(kwargs)
        return Classifier(ckargs=ckargs, *args)
    
    def load(self, fn, *args, **kwargs):
        args = list(self.args) + list(args)
        #kwargs.update(self.kwargs)
        return Classifier.load(fn, *args, **kwargs)
    
    def __repr__(self):
        return self.name.split('.')[-1]
    
# Generate shortcuts for instantiating each classifier.
for _name in WEKA_CLASSIFIERS:
    _parts = _name.split(' ')
    _name = _parts[0]
    _proper_name = _name.split('.')[-1]
    _ckargs = {}
    _arg_name = None
    for _arg in _parts[1:]:
        if _arg.startswith('-'):
            _arg_name = _arg[1:]
        else:
            _ckargs[_arg_name] = _arg
    _func = _Helper(name=_name, ckargs=_ckargs)
    exec('%s = _func' % _proper_name) # pylint: disable=exec-used

# These can be trained incrementally.
# http://weka.sourceforge.net/doc/weka/classifiers/UpdateableClassifier.html
UPDATEABLE_WEKA_CLASSIFIERS = [
'weka.classifiers.bayes.AODE',
'weka.classifiers.lazy.IB1', 
'weka.classifiers.lazy.IBk',
'weka.classifiers.lazy.KStar',
'weka.classifiers.lazy.LWL',
'weka.classifiers.bayes.NaiveBayesUpdateable',
'weka.classifiers.rules.NNge',
'weka.classifiers.meta.RacedIncrementalLogitBoost',
'weka.classifiers.functions.SGD',
'weka.classifiers.functions.Winnow',
]
UPDATEABLE_WEKA_CLASSIFIER_NAMES = set(_.split('.')[-1] for _ in UPDATEABLE_WEKA_CLASSIFIERS)

WEKA_ACCURACY_REGEX = re.compile(r'===\s+Stratified cross-validation\s+===' + \
    r'\n+\s*\n+\s*Correctly Classified Instances\s+[0-9]+\s+([0-9\.]+)\s+%',
    re.DOTALL)
WEKA_TEST_ACCURACY_REGEX = re.compile(r'===\s+Error on test data\s+===\n+\s' + \
    r'*\n+\s*Correctly Classified Instances\s+[0-9]+\s+([0-9\.]+)\s+%',
    re.DOTALL)

class PredictionResult(object):
    
    def __init__(self, actual, predicted, probability):
        self.actual = actual
        self.predicted = predicted
        self.probability = probability
    
    def __unicode__(self):
        return unicode('<%s: actual=%s, predicted=%s, probability=%s>' % (type(self).__name__, self.actual, self.predicted, self.probability))
    
    def __str__(self):
        return '<%s: actual=%s, predicted=%s, probability=%s>' % (type(self).__name__, self.actual, self.predicted, self.probability)
    
#     def __repr__(self):
#         return unicode(self)
    
    @property
    def certainty(self):
        return self.probability.get(self.predicted)
    
    @classmethod
    def avg(cls, *instances):
        total = Decimal(len(instances))
        predicted = sum([instance.predicted for instance in instances if instance.predicted is not None])/total
        probs = [instance.probability for instance in instances if instance.probability is not None]
        if probs:
            probability = sum(probs)
        else:
            probability = None
        return cls(actual=None, predicted=predicted, probability=probability)
    
    def __hash__(self):
        return hash((self.actual, self.predicted, self.probability))
    
    # Note, this is ignored in Python3.
    def __cmp__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return cmp(
            (self.actual, self.predicted, self.probability),
            (other.actual, other.predicted, other.probability),
        )
    
    # Needed for Python3.
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.actual, self.predicted, self.probability) == (other.actual, other.predicted, other.probability)

def get_weka_accuracy(arff_fn, arff_test_fn, cls):
    assert cls in WEKA_CLASSIFIERS, "Unknown Weka classifier: %s" % (cls,)
    cmd = "java -cp /usr/share/java/weka.jar:/usr/share/java/libsvm.jar %(cls)s -t \"%(arff_fn)s\" -T \"%(arff_test_fn)s\"" % locals()
    print(cmd)
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    try:
        acc = float(WEKA_TEST_ACCURACY_REGEX.findall(output)[0])
        return acc
    except IndexError:
        return 0
    except TypeError:
        return 0
    except Exception as e:
        print('!'*80)
        print("Unexpected Error: %s" % e)
        return 0

class TrainingError(Exception):
    pass

class PredictionError(Exception):
    pass

class Classifier(object):
    
    def __init__(self, name, ckargs=None, model_data=None):
        self._model_data = model_data
        self.name = name # Weka classifier class name.
        self.schema = None
        self.ckargs = ckargs
        
        self.last_training_stdout = None
        self.last_training_stderr = None

    @classmethod
    def load(cls, fn, compress=True, *args, **kwargs):
        if compress and not fn.strip().lower().endswith('.gz'):
            fn = fn + '.gz'
        assert os.path.isfile(fn), 'File %s does not exist.' % (fn,)
        if compress:
            return pickle.load(gzip.open(fn, 'rb'))
        return pickle.load(open(fn, 'rb'))

    @classmethod
    def load_raw(cls, model_fn, schema, *args, **kwargs):
        """
        Loads a trained classifier from the raw Weka model format.
        Must specify the model schema and classifier name, since
        these aren't currently deduced from the model format.
        """
        c = cls(*args, **kwargs)
        c.schema = schema.copy(schema_only=True)
        c._model_data = open(model_fn, 'rb').read()
        return c
        
    def save(self, fn, compress=True):
        if compress and not fn.strip().lower().endswith('.gz'):
            fn = fn + '.gz'
        if compress:
            pickle.dump(self, gzip.open(fn, 'wb'))
        else:
            pickle.dump(self, open(fn, 'wb'))
        
    def _get_ckargs_str(self):
        ckargs = []
        if self.ckargs:
            for k, v in iteritems(self.ckargs):
                if not k.startswith('-'):
                    k = '-'+k
                if v is None:
                    ckargs.append('%s' % (k,))
                else:
                    ckargs.append('%s %s' % (k, v))
        ckargs = ' '.join(ckargs)
        return ckargs

    @property
    def training_correlation_coefficient(self):
        matches = re.findall(r'Correlation coefficient\s+([0-9\.]+)', self.last_training_stdout)
        if matches:
            return float(matches[0])

    @property
    def training_mean_absolute_error(self):
        matches = re.findall(r'Mean absolute error\s+([0-9\.]+)', self.last_training_stdout)
        if matches:
            return float(matches[0])

    def train(self, training_data, testing_data=None, verbose=False):
        """
        Updates the classifier with new data.
        """
        model_fn = None
        training_fn = None
        clean_training = False
        testing_fn = None
        clean_testing = False
        try:
            
            # Validate training data.
            if isinstance(training_data, basestring):
                assert os.path.isfile(training_data)
                training_fn = training_data
            else:
                assert isinstance(training_data, arff.ArffFile)
                fd, training_fn = tempfile.mkstemp(suffix='.arff')
                os.close(fd)
                with open(training_fn, 'w') as fout:
                    fout.write(training_data.write())
                clean_training = True
            assert training_fn
                
            # Validate testing data.
            if testing_data:
                if isinstance(testing_data, basestring):
                    assert os.path.isfile(testing_data)
                    testing_fn = testing_data
                else:
                    assert isinstance(testing_data, arff.ArffFile)
                    fd, testing_fn = tempfile.mkstemp(suffix='.arff')
                    os.close(fd)
                    with open(testing_fn, 'w') as fout:
                        fout.write(testing_data.write())
                    clean_testing = True
            else:
                testing_fn = training_fn
            assert testing_fn
                
            # Validate model file.
            fd, model_fn = tempfile.mkstemp()
            os.close(fd)
            if self._model_data:
                fout = open(model_fn, 'wb')
                fout.write(self._model_data)
                fout.close()
            
            # Call Weka Jar.
            args = dict(
                CP=CP,
                classifier_name=self.name,
                model_fn=model_fn,
                training_fn=training_fn,
                testing_fn=testing_fn,
                ckargs=self._get_ckargs_str(),
            )
            if self._model_data:
                # Load existing model.
                cmd = (
                    "java -cp %(CP)s %(classifier_name)s -l \"%(model_fn)s\" "
                    "-t \"%(training_fn)s\" -T \"%(testing_fn)s\" -d \"%(model_fn)s\"") % args
            else:
                # Create new model file.
                cmd = (
                    "java -cp %(CP)s %(classifier_name)s -t \"%(training_fn)s\" "
                    "-T \"%(testing_fn)s\" -d \"%(model_fn)s\" %(ckargs)s") % args
            if verbose:
                print(cmd)
            p = Popen(
                cmd,
                shell=True,
                stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=sys.platform != "win32")
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            
            self.last_training_stdout = stdout_str
            self.last_training_stderr = stderr_str
            
            if verbose:
                print('stdout:')
                print(stdout_str)
                print('stderr:')
                print(stderr_str)
            if stderr_str:
                raise TrainingError(stderr_str)
            
            # Save schema.
            if not self.schema:
                self.schema = arff.ArffFile.load(training_fn, schema_only=True).copy(schema_only=True)
            
            # Save model.
            with open(model_fn, 'rb') as fin:
                self._model_data = fin.read()
            assert self._model_data
        finally:
            # Cleanup files.
            if model_fn:
                os.remove(model_fn)
            if training_fn and clean_training:
                os.remove(training_fn)
            if testing_fn and clean_testing:
                os.remove(testing_fn)
        
    def predict(self, query_data, verbose=False, distribution=False, cleanup=True):
        """
        Iterates over the predicted values and probability (if supported).
        Each iteration yields a tuple of the form (prediction, probability).
        
        If the file is a test file (i.e. contains no query variables),
        then the tuple will be of the form (prediction, actual).
        
        See http://weka.wikispaces.com/Making+predictions
        for further explanation on interpreting Weka prediction output.
        """
        model_fn = None
        query_fn = None
        clean_query = False
        stdout = None
        try:
            
            # Validate query data.
            if isinstance(query_data, basestring):
                assert os.path.isfile(query_data)
                query_fn = query_data
            else:
                #assert isinstance(query_data, arff.ArffFile) #TODO: doesn't work in Python 3.*?
                assert type(query_data).__name__ == 'ArffFile', 'Must be of type ArffFile, not "%s"' % type(query_data).__name__
                fd, query_fn = tempfile.mkstemp(suffix='.arff')
                if verbose:
                    print('writing', query_fn)
                os.close(fd)
                open(query_fn, 'w').write(query_data.write())
                clean_query = True
            assert query_fn
                
            # Validate model file.
            fd, model_fn = tempfile.mkstemp()
            os.close(fd)
            assert self._model_data, "You must train this classifier before predicting."
            fout = open(model_fn, 'wb')
            fout.write(self._model_data)
            fout.close()
            
#            print(open(model_fn).read()
#            print(open(query_fn).read()
            # Call Weka Jar.
            args = dict(
                CP=CP,
                classifier_name=self.name,
                model_fn=model_fn,
                query_fn=query_fn,
                #ckargs = self._get_ckargs_str(),
                distribution=('-distribution' if distribution else ''),
            )
            cmd = ("java -cp %(CP)s %(classifier_name)s -p 0 %(distribution)s -l \"%(model_fn)s\" -T \"%(query_fn)s\"") % args
            if verbose:
                print(cmd)
            p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            if verbose:
                print('stdout:')
                print(stdout_str)
                print('stderr:')
                print(stderr_str)
            if stderr_str:
                raise PredictionError(stderr_str)
            
            if stdout_str:
                # inst#     actual  predicted error prediction
                #header = 'inst,actual,predicted,error'.split(',')
                query = arff.ArffFile.load(query_fn)
                query_variables = [
                    query.attributes[i]
                    for i, v in enumerate(query.data[0])
                    if v == arff.MISSING]
                if not query_variables:
                    query_variables = [query.attributes[-1]]
#                assert query_variables, \
#                    "There must be at least one query variable in the query."
                if verbose:
                    print('query_variables:', query_variables)
                header = 'predicted'.split(',')
                # sample line:     1        1:?       4:36   +   1
                
                # Expected output without distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error prediction
                #     1        1:? 11:Acer_tr   +   1

                #=== Predictions on test data ===
                #
                # inst#     actual  predicted      error
                #     1          ?      7              ? 

                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error prediction
                #     1        1:?        1:0       0.99 
                #     2        1:?        1:0       0.99 
                #     3        1:?        1:0       0.99 
                #     4        1:?        1:0       0.99 
                #     5        1:?        1:0       0.99 

                # Expected output with distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error distribution
                #     1        1:? 11:Acer_tr   +   0,0,0,0,0,0,0,0,0,0,*1,0,0,0,0,0...

                q = re.findall(
                    r'J48 pruned tree\s+\-+:\s+([0-9]+)\s+',
                    stdout_str.decode('utf-8'), re.MULTILINE|re.DOTALL)
                if q:
                    class_label = q[0]
                    prob = 1.0
                    yield PredictionResult(
                        actual=None,
                        predicted=class_label,
                        probability=prob,)
                elif re.findall(r'error\s+(?:distribution|prediction)', stdout_str.decode('utf-8')):
                    # Check for distribution output.
                    matches = re.findall(
                        r"^\s*[0-9\.]+\s+[a-zA-Z0-9\.\?\:]+\s+(?P<cls_value>[a-zA-Z0-9_\.\?\:]+)"
                        r"\s+\+?\s+(?P<prob>[a-zA-Z0-9\.\?\,\*]+)",
                        stdout_str,
                        re.MULTILINE)
                    assert matches, ("No results found matching distribution pattern in stdout: %s") % stdout_str
                    for match in matches:
                        prediction, prob = match
                        class_index, class_label = prediction.split(':')
                        class_index = int(class_index)
                        if distribution:
                            # Convert list of probabilities into a hash linking the prob
                            # to the associated class value.
                            prob = dict(zip(
                                query.attribute_data[query.attributes[-1]],
                                map(float, prob.replace('*', '').split(','))))
                        else:
                            prob = float(prob)
                        class_label = query.attribute_data[query.attributes[-1]][class_index-1]
                        yield PredictionResult(
                            actual=None,
                            predicted=class_label,
                            probability=prob,)
                else:
                    # Otherwise, assume a simple output.
                    matches = re.findall(
                        r"^\s*([0-9\.]+)\s+([a-zA-Z0-9\.\?\:]+)\s+([a-zA-Z0-9_\.\?\:]+)\s+",
                        stdout_str.decode('utf-8'),
                        re.MULTILINE)
                    assert matches, "No results found matching simple pattern in stdout: %s" % stdout_str
                    #print('matches:',len(matches)
                    for match in matches:
                        inst, actual, predicted = match
                        class_name = query.attributes[-1]
                        actual_value = query.get_attribute_value(class_name, actual)
                        predicted_value = query.get_attribute_value(class_name, predicted)
                        yield PredictionResult(
                            actual=actual_value,
                            predicted=predicted_value,
                            probability=None,)
        finally:
            # Cleanup files.
            if cleanup:
                if model_fn:
                    self._model_data = open(model_fn, 'rb').read()
                    os.remove(model_fn)
                if query_fn and clean_query:
                    os.remove(query_fn)
                
    def test(self, test_data, verbose=0):
        data = arff.ArffFile.load(test_data)
        data_itr = iter(data)
        i = 0
        correct = 0
        total = 0
        for result in self.predict(test_data, verbose=verbose):
            i += 1
            if verbose:
                print(i, result)
            row = next(data_itr)
            total += 1
            correct += result.predicted == result.actual
        return correct/float(total)

class EnsembleClassifier(object):
    
    def __init__(self, classes=None):
        self.best = None, None # score, cls
        self.training_results = {} # {name: score}
        self.trained_classifiers = {} # {name: classifier instance}
        self.prediction_results = {} # {name: results}
        self.classes = list(classes or WEKA_CLASSIFIERS)
        for cls in self.classes:
            assert cls in WEKA_CLASSIFIERS, 'Invalid class: %s' % cls
    
    @classmethod
    def load(cls, fn, compress=True, *args, **kwargs):
        raise NotImplementedError
    
    def get_training_best(self):
        results = list(self.training_results.items())
        results = sorted(results, key=lambda o: o[1])
        print('name: <name> <coef> <inv mae>')
        for name, data in results:
            if isinstance(data, basestring):
                continue
            (coef, inv_mae) = data
            print('name:', name, (coef, inv_mae))
    
    def get_training_errors(self):
        results = list(self.training_results.items())
        results = sorted(results)
        for name, data in results:
            if not isinstance(data, basestring):
                continue
            print('name:', name)
            print(data)
    
    def get_training_coverage(self):
        """
        Returns a ratio of classifiers that were able to be trained successfully.
        """
        total = len(self.training_results)
        i = sum(1 for data in self.training_results.values() if not isinstance(data, basestring))
        return i/float(total)
    
    def train(self, training_data, testing_data=None, verbose=False):

        total = len(self.classes)
        i = 0
        for name in self.classes:
            i += 1
            try:
                c = Classifier(name=name)
                print('Training classifier %i of %i %.02f%% %s...' % (i+1, total, i/float(total)*100, name))
                t0 = time.time()
                c.train(training_data=training_data, testing_data=testing_data, verbose=verbose)
                self.trained_classifiers[name] = c
                td = time.time() - t0
                print('Training seconds:', td)
                coef = c.training_correlation_coefficient
                print('correlation_coefficient:', coef)
                mae = c.training_mean_absolute_error
                print('mean_absolute_error:', mae)
                self.training_results[name] = (coef, 1/(1+float(mae)))
            except Exception:
                traceback.print_exc()
                self.training_results[name] = traceback.format_exc()

    def get_best_predictors(self, tolerance, verbose=False):
        best_coef = -1e9999999999
        best_names = set()
        for name, data in sorted(self.training_results.items(), key=lambda o: o[1][0], reverse=True):
            if isinstance(data, basestring):
                continue
            (coef, inv_mae) = data
            if verbose:
                print('name:', name, coef, inv_mae)
            if coef > best_coef:
                best_coef = coef
                best_names = set([name])
            elif (coef + tolerance) >= best_coef:
                best_names.add(name)
        return best_names

    def predict(self, query_data, tolerance=0, **kwargs):
        verbose = kwargs.get('verbose', False)
        assert self.training_results, 'Classifier must be trained first!'
        
        best_names = self.get_best_predictors(tolerance=tolerance)

        total = len(best_names)
        i = 0
        for name in best_names:
            i += 1
            try:
                c = self.trained_classifiers[name]
                if verbose:
                    print('Querying classifier %i of %i %.02f%% %s...' % (i, total, i/float(total)*100, name))
                t0 = time.time()
                results = list(c.predict(query_data=query_data, **kwargs))
                td = time.time() - t0
                self.prediction_results[name] = results
            except Exception:
                traceback.print_exc()
                self.prediction_results[name] = traceback.format_exc()
        
        results = {} # {index, [results]}
        for k, v in self.prediction_results.items():
            for i, result in enumerate(v):
                if isinstance(v, basestring):
                    continue
                results.setdefault(i, [])
                results[i].append(result)
        
        results = [PredictionResult.avg(*data) for i, data in sorted(results.items())]

        return results
