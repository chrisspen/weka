from __future__ import print_function

import os
import unittest

from weka.classifiers import Classifier, PredictionResult, PredictionError, BP, DENSE, UPDATEABLE_WEKA_CLASSIFIER_NAMES
from weka.classifiers import IBk # pylint: disable=no-name-in-module
from weka import arff
from weka.arff import Num, Nom, Int, Str, Date

class Test(unittest.TestCase):
    
    def test_arff(self):
    
        data = arff.ArffFile.load(os.path.join(BP, 'fixtures/abalone-train.arff'))
        self.assertEqual(len(data.attributes), 9)
    
    def test_numeric(self):
        
        n1 = Num(1.23)
        n2 = Num(4.56)
        self.assertEqual(n1.value, 1.23)
        self.assertEqual(n2.value, 4.56)
        
        n3 = n1 + n2
        self.assertEqual(n3.value, n1.value + n2.value)
        self.assertNotEqual(n3, n1)
        self.assertNotEqual(n3, n2)
        
        n3 += 1
        self.assertEqual(n3.value, n1.value + n2.value + 1)
        
        s = sum([n1, n2, n3], Num(0))
        print(s)
        self.assertTrue(isinstance(s, Num))
        self.assertEqual(s.value, (n1 + n2 + n3).value)
        
        n4 = n1 / 10
        self.assertEqual(n4.value, 0.123)
        
        n4 /= 10
        self.assertEqual(n4.value, 0.0123)
    
    def test_IBk(self):
        
        # Train a classifier.
        print('Training IBk classifier...')
        c = Classifier(name='weka.classifiers.lazy.IBk', ckargs={'-K':1})
        training_fn = os.path.join(BP, 'fixtures/abalone-train.arff')
        c.train(training_fn, verbose=1)
        self.assertTrue(c._model_data)
        
        # Make a valid query.
        print('Using IBk classifier...')
        query_fn = os.path.join(BP, 'fixtures/abalone-query.arff')
        predictions = list(c.predict(query_fn, verbose=1, cleanup=0))
        pred0 = predictions[0]
        print('pred0:', pred0)
        pred1 = PredictionResult(actual=None, predicted=7, probability=None)
        print('pred1:', pred1)
        self.assertEqual(pred0, pred1)
            
        # Make a valid query.
        with self.assertRaises(PredictionError):
            query_fn = os.path.join(BP, 'fixtures/abalone-query-bad.arff')
            predictions = list(c.predict(query_fn, verbose=1, cleanup=0))
            
        # Make a valid query manually.
        query = arff.ArffFile(relation='test', schema=[
            ('Sex', ('M', 'F', 'I')),
            ('Length', 'numeric'),
            ('Diameter', 'numeric'),
            ('Height', 'numeric'),
            ('Whole weight', 'numeric'),
            ('Shucked weight', 'numeric'),
            ('Viscera weight', 'numeric'),
            ('Shell weight', 'numeric'),
            ('Class_Rings', 'integer'),
        ])
        query.append(['M', 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07, '?'])
        data_str0 = """% 
@relation test
@attribute 'Sex' {F,I,M}
@attribute 'Length' numeric
@attribute 'Diameter' numeric
@attribute 'Height' numeric
@attribute 'Whole weight' numeric
@attribute 'Shucked weight' numeric
@attribute 'Viscera weight' numeric
@attribute 'Shell weight' numeric
@attribute 'Class_Rings' integer
@data
M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,?
"""
        data_str1 = query.write(fmt=DENSE)
#        print(data_str0
#        print(data_str1
        self.assertEqual(data_str0, data_str1)
        predictions = list(c.predict(query, verbose=1, cleanup=0))
        self.assertEqual(predictions[0],
            PredictionResult(actual=None, predicted=7, probability=None))
        
        # Test pickling.
        fn = os.path.join(BP, 'fixtures/IBk.pkl')
        c.save(fn)
        c = Classifier.load(fn)
        predictions = list(c.predict(query, verbose=1, cleanup=0))
        self.assertEqual(predictions[0],
            PredictionResult(actual=None, predicted=7, probability=None))
        #print('Pickle verified.')
        
        # Make a valid dict query manually.
        query = arff.ArffFile(relation='test', schema=[
            ('Sex', ('M', 'F', 'I')),
            ('Length', 'numeric'),
            ('Diameter', 'numeric'),
            ('Height', 'numeric'),
            ('Whole weight', 'numeric'),
            ('Shucked weight', 'numeric'),
            ('Viscera weight', 'numeric'),
            ('Shell weight', 'numeric'),
            ('Class_Rings', 'integer'),
        ])
        query.append({
            'Sex': 'M',
            'Length': 0.35,
            'Diameter': 0.265,
            'Height': 0.09,
            'Whole weight': 0.2255,
            'Shucked weight': 0.0995,
            'Viscera weight': 0.0485,
            'Shell weight': 0.07,
            'Class_Rings': arff.MISSING,
        })
        predictions = list(c.predict(query, verbose=1, cleanup=0))
        self.assertEqual(predictions[0],
            PredictionResult(actual=None, predicted=7, probability=None))

    def test_shortcut(self):
        c = IBk(K=1) # pylint: disable=undefined-variable
        
        training_fn = os.path.join(BP, 'fixtures/abalone-train.arff')
        c.train(training_fn, verbose=1)
        self.assertTrue(c._model_data)
        
        # Make a valid query.
        query_fn = os.path.join(BP, 'fixtures/abalone-query.arff')
        predictions = list(c.predict(query_fn, verbose=1, cleanup=0))
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0],
            PredictionResult(actual=None, predicted=7, probability=None))
        
    def test_updateable(self):
        """
        Confirm updateable classifiers are used so that their model is in fact
        updated and not overwritten.
        """
        c = IBk(K=1) # pylint: disable=undefined-variable
        self.assertTrue('IBk' in UPDATEABLE_WEKA_CLASSIFIER_NAMES)
        
        train_fn1 = os.path.join(BP, 'fixtures/updateable-train-1.arff')
        train_fn2 = os.path.join(BP, 'fixtures/updateable-train-2.arff')
        save_fn = os.path.join(BP, 'fixtures/IBk.updated.pkl')
        if os.path.isfile(save_fn):
            os.remove(save_fn)
        
        c.train(train_fn1)
        self.assertTrue(c._model_data)
        
        # It should have a perfect accuracy when tested on the same file
        # it was trained with.
        acc = c.test(train_fn1, verbose=1)
        self.assertEqual(acc, 1.0)
        
        # It should have horrible accuracy on a completely different data
        # file that it hasn't been trained on.
        acc = c.test(train_fn2, verbose=1)
        self.assertEqual(acc, 0.0)
        pre_del_model = c._model_data
        
        # Reload the classifier from a pickle.
        c.save(save_fn)
        del c
        
        c = IBk.load(save_fn) # pylint: disable=undefined-variable
        self.assertTrue(c._model_data)
        self.assertEqual(c._model_data, pre_del_model)
        
        # Confirm the Weka model was persisted by confirming we still have
        # perfect accuracy on the initial training file.
        acc = c.test(train_fn1, verbose=1)
        self.assertEqual(acc, 1.0)
        
        # Train the classifier on a completely different data set.
        c.train(train_fn2)
        
        # Confirm it has perfect accuracy on the new data set.
        acc = c.test(train_fn2, verbose=1)
        self.assertEqual(acc, 1.0)
        
        # Confirm we still have perfect accuracy on the original data set.
        acc = c.test(train_fn1, verbose=1)
        self.assertEqual(acc, 1.0)

    def test_PredictionResult_cmp(self):
        
        a = PredictionResult(1, 2, 3)
        b = PredictionResult(1, 2, 3)
        
        self.assertEqual(a, b)
    
    def test_sparse_stream(self):
        
        s0 = """% 
@relation test-abalone
@attribute 'Diameter' numeric
@attribute 'Length' numeric
@attribute 'Sex' {F,M}
@attribute 'Timestamp' date "yyyy-MM-dd HH:mm:ss"
@attribute 'Whole_weight' numeric
@attribute 'Class_Rings' integer
@data
{0 0.286, 1 0.35, 2 M, 3 "2017-12-01 00:00:00", 5 15}
{0 0.86, 2 F, 3 "2017-12-27 00:00:00", 4 0.98, 5 7}
"""
        
        rows = [
            dict(
                Sex=Nom('M'),
                Length=Num(0.35),
                Diameter=Num(0.286),
                Timestamp=Date('2017-12-01'),
                Class_Rings=Int(15, cls=True)),
            dict(
                Sex=Nom('F'),
                Diameter=Num(0.86),
                Timestamp=Date('2017-12-27'),
                Whole_weight=Num(0.98),
                Class_Rings=Int(7, cls=True)),
        ]
        rows_extra = [
            dict(
                Sex=Nom('N'),
                Length=Num(0.35),
                Diameter=Num(0.286),
                Class_Rings=Int(15, cls=True)),
            dict(
                Sex=Nom('B'),
                Diameter=Num(0.86),
                Whole_weight=Num(0.98),
                Class_Rings=Int(7, cls=True)),
        ]
        
        a1 = arff.ArffFile(relation='test-abalone')
        for row in rows:
            a1.append(row)
        self.assertEqual(a1.class_attr_name, 'Class_Rings')
        a1.alphabetize_attributes()
        print('a1.attributes:', a1.attributes)
        self.assertEqual(a1.attributes, ['Diameter', 'Length', 'Sex', 'Timestamp', 'Whole_weight', 'Class_Rings'])
        s1 = a1.write()
        print('s0:', s0)
        print('s1:', s1)
        self.assertEqual(s1, s0)
        
        a2 = arff.ArffFile.parse(s1)
        for i, line in enumerate(a2.data):
            self.assertEqual(line, a1.data[i])
        s2 = a2.write()
        self.assertEqual(s1, s2)
        
        a3 = arff.ArffFile(relation='test-abalone')
        self.assertEqual(len(a3.data), 0)
        #a3.open_stream(class_attr_name='Class_Rings')
        
        # When streaming, you have to provide your schema ahead of time,
        # since otherwise we'd have to update the indexes on all files
        # previously written to the file.
        for row in rows:
            a3.append(row, schema_only=True)
            self.assertEqual(len(a3.data), 0)
        
        a3.alphabetize_attributes()    
        a3.open_stream(class_attr_name='Class_Rings')
        for row in (rows+rows_extra):
            print('row:', row)
            a3.append(row)
            self.assertEqual(len(a3.data), 0)
        
        fn = a3.close_stream()
        s3 = open(fn, 'r').read()
        print('s3:', s3)
        s4 = """% 
@relation test-abalone
@attribute 'Diameter' numeric
@attribute 'Length' numeric
@attribute 'Sex' {F,M}
@attribute 'Timestamp' date "yyyy-MM-dd HH:mm:ss"
@attribute 'Whole_weight' numeric
@attribute 'Class_Rings' integer
@data
{0 0.286, 1 0.35, 2 M, 3 "2017-12-01 00:00:00", 5 15}
{0 0.86, 2 F, 3 "2017-12-27 00:00:00", 4 0.98, 5 7}
{0 0.286, 1 0.35, 5 15}
{0 0.86, 4 0.98, 5 7}
"""
        print('s4:', s4)
        os.remove(fn)
        # Note the rows that have features violating the schema are
        # automatically omitted when in streaming mode.
        self.assertEqual(s3, s4)

if __name__ == '__main__':
    unittest.main()
