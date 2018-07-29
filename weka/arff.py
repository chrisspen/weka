# Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
2011.3.6 CKS
Fixed regular expression to handle special characters in attribute names.
Added options for parsing and copying only the schema.
2012.11.4 CKS
Added support for streaming and sparse data.
"""
from __future__ import print_function

import os
import sys
import re
import copy
import unittest
import tempfile
from datetime import date, datetime
from decimal import Decimal

from six import StringIO
from six import string_types as basestring # pylint: disable=redefined-builtin

import dateutil.parser

MISSING = '?'

def is_numeric(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

DENSE = 'dense'
SPARSE = 'sparse'
FORMATS = (DENSE, SPARSE)

TYPE_INTEGER = 'integer'
TYPE_NUMERIC = 'numeric' # float or integer
TYPE_REAL = 'real'
TYPE_STRING = 'string'
TYPE_NOMINAL = 'nominal'
TYPE_DATE = 'date'
TYPES = (
    TYPE_INTEGER,
    TYPE_NUMERIC,
    TYPE_STRING,
    TYPE_NOMINAL,
    TYPE_DATE,
)
NUMERIC_TYPES = (
    TYPE_INTEGER,
    TYPE_NUMERIC,
    TYPE_REAL,
)

STRIP_QUOTES_REGEX = re.compile('^[\'\"]|[\'\"]$')

#DEFAULT_DATE_FORMAT = "yyyy-MM-dd'T'HH:mm:ss" # Weka docs say this is the default, but using this causes Weka to throw an java.io.IOException: unparseable date
DEFAULT_DATE_FORMAT = "yyyy-MM-dd HH:mm:ss"
#DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def convert_weka_to_py_date_pattern(p):
    """
    Converts the date format pattern used by Weka to the date format pattern used by Python's datetime.strftime().
    """
    # https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    # https://www.cs.waikato.ac.nz/ml/weka/arff.html
    p = p.replace('yyyy', r'%Y')
    p = p.replace('MM', r'%m')
    p = p.replace('dd', r'%d')
    p = p.replace('HH', r'%H')
    p = p.replace('mm', r'%M')
    p = p.replace('ss', r'%S')
    return p

def cmp(a, b): # pylint: disable=redefined-builtin
    return (a > b) - (a < b)

class Value(object):
    """
    Base helper class for tagging units of data with an explicit schema type.
    """
    
    __slots__ = ('value', 'cls')
    
    def __init__(self, v, cls=False):
        self.value = v
        self.cls = cls
    
    def __hash__(self):
        #return hash((self.value, self.cls))
        return hash(self.value)
    
    def __eq__(self, other):
        if isinstance(other, Value):
            return self.value == other.value
        return NotImplemented
    
    def __cmp__(self, other):
        if isinstance(other, Value):
            return cmp(self.value, other.value)
        return NotImplemented
    
    def __repr__(self):
        return repr(self.value)

class Integer(Value):
    c_type = TYPE_INTEGER
    def __init__(self, v, *args, **kwargs):
        if v != MISSING:
            v = int(v)
        super(Integer, self).__init__(v, *args, **kwargs)
        
    def __add__(self, other):
        if isinstance(other, Integer):
            return Integer(v=self.value + other.value, cls=self.cls)
        elif isinstance(other, (int, float, bool)):
            return Integer(v=self.value + other, cls=self.cls)
        return NotImplemented
        
    def __iadd__(self, other):
        if isinstance(other, Integer):
            self.value += other.value
            return self
        elif isinstance(other, (int, float, bool)):
            self.value += other
            return self
        return NotImplemented
            
Int = Integer

class Numeric(Value):
    c_type = TYPE_NUMERIC
    def __init__(self, v, *args, **kwargs):
        # TODO:causes loss of precision?
        if v != MISSING:
            v = float(v)
        super(Numeric, self).__init__(v, *args, **kwargs)
        
    def __add__(self, other):
        if isinstance(other, Numeric):
            return Numeric(v=self.value + other.value, cls=self.cls)
        elif isinstance(other, (int, float, bool)):
            return Numeric(v=self.value + other, cls=self.cls)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Numeric):
            self.value += other.value
            return self
        elif isinstance(other, (int, float, bool)):
            self.value += other
            return self
        return NotImplemented
        
    def __div__(self, other):
        if isinstance(other, Numeric):
            return Numeric(v=self.value / other.value, cls=self.cls)
        elif isinstance(other, (int, float, bool)):
            return Numeric(v=self.value / other, cls=self.cls)
        return NotImplemented
        
    def __truediv__(self, other):
        if isinstance(other, Numeric):
            return Numeric(v=self.value / other.value, cls=self.cls)
        elif isinstance(other, (int, float, bool)):
            return Numeric(v=self.value / other, cls=self.cls)
        return NotImplemented

    def __idiv__(self, other):
        if isinstance(other, Numeric):
            self.value /= other.value
            return self
        elif isinstance(other, (int, float, bool)):
            self.value /= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Numeric):
            self.value /= other.value
            return self
        elif isinstance(other, (int, float, bool)):
            self.value /= other
            return self
        return NotImplemented

Num = Numeric

class String(Value):
    c_type = TYPE_STRING
    def __init__(self, v, *args, **kwargs):
        v = str(v)
        super(String, self).__init__(v, *args, **kwargs)
Str = String

class Nominal(Value):
    c_type = TYPE_NOMINAL
Nom = Nominal

class Date(Value):
    c_type = TYPE_DATE
Dt = Date

TYPE_TO_CLASS = {
    TYPE_INTEGER: Integer,
    TYPE_NUMERIC: Numeric,
    TYPE_STRING: String,
    TYPE_NOMINAL: Nominal,
    TYPE_REAL: Numeric,
    TYPE_DATE: Date,
}

def wrap_value(v):
    if isinstance(v, Value):
        return v
    if v == MISSING:
        return Str(v)
    if isinstance(v, basestring):
        return Str(v)
    try:
        return Num(v)
    except ValueError:
        pass
    try:
        return Int(v)
    except ValueError:
        pass
    try:
        return Date(v)
    except ValueError:
        pass

class ArffFile(object):
    """An ARFF File object describes a data set consisting of a number
    of data points made up of attributes. The whole data set is called
    a 'relation'. Supported attributes are:

    - 'numeric': floating point numbers
    - 'string': strings
    - 'nominal': taking one of a number of possible values

    Not all features of ARFF files are supported yet. The most notable
    exceptions are:

    - no sparse data
    - no support for date and relational attributes

    Also, parsing of strings might still be a bit brittle.

    You can either load or save from files, or write and parse from a
    string.

    You can also construct an empty ARFF file and then fill in your
    data by hand. To define attributes use the define_attribute method.

    Attributes are:

    - 'relation': name of the relation
    - 'attributes': names of the attributes
    - 'attribute_types': types of the attributes
    - 'attribute_data': additional data, for example for nominal attributes.
    - 'comment': the initial comment in the file. Typically contains some
                 information on the data set.
    - 'data': the actual data, by data points.
    """
    def __init__(self, relation='', schema=None):
        """Construct an empty ARFF structure."""
        self.relation = relation
        self.clear()
        
        # Load schema.
        if schema:
            for name, data in schema:
                name = STRIP_QUOTES_REGEX.sub('', name)
                self.attributes.append(name)
                if isinstance(data, (tuple, list)):
                    self.attribute_types[name] = TYPE_NOMINAL
                    self.attribute_data[name] = set(data)
                else:
                    self.attribute_types[name] = data
                    self.attribute_data[name] = None
        
    def clear(self):
        self.attributes = [] # [attr_name, attr_name, ...]
        self.attribute_types = dict() # {attr_name:type}
        self.attribute_data = dict() # {attr_name:[nominal values]}
        self._filename = None
        self.comment = []
        self.data = []
        self.lineno = 0
        self.fout = None
        self.class_attr_name = None
    
    def get_attribute_value(self, name, index):
        """
        Returns the value associated with the given value index
        of the attribute with the given name.
        
        This is only applicable for nominal and string types.
        """
        if index == MISSING:
            return
        elif self.attribute_types[name] in NUMERIC_TYPES:
            at = self.attribute_types[name]
            if at == TYPE_INTEGER:
                return int(index)
            return Decimal(str(index))
        else:
            assert self.attribute_types[name] == TYPE_NOMINAL
            cls_index, cls_value = index.split(':')
            #return self.attribute_data[name][index-1]
            if cls_value != MISSING:
                assert cls_value in self.attribute_data[name], \
                    'Predicted value "%s" but only values %s are allowed.' \
                        % (cls_value, ', '.join(self.attribute_data[name]))
            return cls_value
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for d in self.data:
            named = dict(zip(
                [re.sub(r'^[\'\"]|[\'\"]$', '', _) for _ in self.attributes],
                d))
            assert len(d) == len(self.attributes)
            assert len(d) == len(named)
            yield named

    @classmethod
    def load(cls, filename, schema_only=False):
        """
        Load an ARFF File from a file.
        """
        o = open(filename)
        s = o.read()
        a = cls.parse(s, schema_only=schema_only)
        if not schema_only:
            a._filename = filename
        o.close()
        return a

    @classmethod
    def parse(cls, s, schema_only=False):
        """
        Parse an ARFF File already loaded into a string.
        """
        a = cls()
        a.state = 'comment'
        a.lineno = 1
        for l in s.splitlines():
            a.parseline(l)
            a.lineno += 1
            if schema_only and a.state == 'data':
                # Don't parse data if we're only loading the schema.
                break
        return a

    def copy(self, schema_only=False):
        """
        Creates a deepcopy of the instance.
        If schema_only is True, the data will be excluded from the copy.
        """
        o = type(self)()
        o.relation = self.relation
        o.attributes = list(self.attributes)
        o.attribute_types = self.attribute_types.copy()
        o.attribute_data = self.attribute_data.copy()
        if not schema_only:
            o.comment = list(self.comment)
            o.data = copy.deepcopy(self.data)
        return o

    def flush(self):
        if self.fout:
            self.fout.flush()

    def open_stream(self, class_attr_name=None, fn=None):
        """
        Save an arff structure to a file, leaving the file object
        open for writing of new data samples.
        This prevents you from directly accessing the data via Python,
        but when generating a huge file, this prevents all your data
        from being stored in memory.
        """
        if fn:
            self.fout_fn = fn
        else:
            fd, self.fout_fn = tempfile.mkstemp()
            os.close(fd)
        self.fout = open(self.fout_fn, 'w')
        if class_attr_name:
            self.class_attr_name = class_attr_name
        self.write(fout=self.fout, schema_only=True)
        self.write(fout=self.fout, data_only=True)
        self.fout.flush()
        
    def close_stream(self):
        """
        Terminates an open stream and returns the filename
        of the file containing the streamed data.
        """
        if self.fout:
            fout = self.fout
            fout_fn = self.fout_fn
            self.fout.flush()
            self.fout.close()
            self.fout = None
            self.fout_fn = None
            return fout_fn

    def save(self, filename=None):
        """
        Save an arff structure to a file.
        """
        filename = filename or self._filename
        o = open(filename, 'w')
        o.write(self.write())
        o.close()

    def write_line(self, d, fmt=SPARSE):
        """
        Converts a single data line to a string.
        """
        
        def smart_quote(s):
            if isinstance(s, basestring) and ' ' in s and s[0] != '"':
                s = '"%s"' % s
            return s
        
        if fmt == DENSE:
            #TODO:fix
            assert not isinstance(d, dict), NotImplemented
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at in NUMERIC_TYPES:
                    line.append(str(e))
                elif at == TYPE_STRING:
                    line.append(self.esc(e))
                elif at == TYPE_NOMINAL:
                    line.append(e)
                else:
                    raise Exception("Type " + at + " not supported for writing!")
            s = ','.join(map(str, line))
            return s
        elif fmt == SPARSE:
            line = []
            
            # Convert flat row into dictionary.
            if isinstance(d, (list, tuple)):
                d = dict(zip(self.attributes, d))
                for k in d:
                    at = self.attribute_types.get(k)
                    if isinstance(d[k], Value):
                        continue
                    elif d[k] == MISSING:
                        d[k] = Str(d[k])
                    elif at in (TYPE_NUMERIC, TYPE_REAL):
                        d[k] = Num(d[k])
                    elif at == TYPE_STRING:
                        d[k] = Str(d[k])
                    elif at == TYPE_INTEGER:
                        d[k] = Int(d[k])
                    elif at == TYPE_NOMINAL:
                        d[k] = Nom(d[k])
                    elif at == TYPE_DATE:
                        d[k] = Date(d[k])
                    else:
                        raise Exception('Unknown type: %s' % at)

            for i, name in enumerate(self.attributes):
                v = d.get(name)
                if v is None:
#                    print 'Skipping attribute with None value:', name
                    continue
                elif v == MISSING or (isinstance(v, Value) and v.value == MISSING):
                    v = MISSING
                elif isinstance(v, String):
                    v = '"%s"' % v.value
                elif isinstance(v, Date):
                    date_format = self.attribute_data.get(name, DEFAULT_DATE_FORMAT)
                    date_format = convert_weka_to_py_date_pattern(date_format)
                    if isinstance(v.value, basestring):
                        _value = dateutil.parser.parse(v.value)
                    else:
                        assert isinstance(v.value, (date, datetime))
                        _value = v.value
                    v.value = v = _value.strftime(date_format)
                elif isinstance(v, Value):
                    v = v.value

                if v != MISSING and self.attribute_types[name] == TYPE_NOMINAL and str(v) not in map(str, self.attribute_data[name]):
                    pass
                else:
                    line.append('%i %s' % (i, smart_quote(v)))

            if len(line) == 1 and MISSING in line[-1]:
                # Skip lines with nothing other than a missing class.
                return
            elif not line:
                # Don't write blank lines.
                return
            return '{' + (', '.join(line)) + '}'
        else:
            raise Exception('Uknown format: %s' % (fmt,))

    def write_attributes(self, fout=None):
        close = False
        if fout is None:
            close = True
            fout = StringIO()
        for a in self.attributes:
            at = self.attribute_types[a]
            if at == TYPE_INTEGER:
                print("@attribute " + self.esc(a) + " integer", file=fout)
            elif at in (TYPE_NUMERIC, TYPE_REAL):
                print("@attribute " + self.esc(a) + " numeric", file=fout)
            elif at == TYPE_STRING:
                print("@attribute " + self.esc(a) + " string", file=fout)
            elif at == TYPE_NOMINAL:
                nom_vals = [_ for _ in self.attribute_data[a] if _ != MISSING]
                nom_vals = sorted(nom_vals)
                print("@attribute " + self.esc(a) + " {" + ','.join(map(str, nom_vals)) + "}", file=fout)
            elif at == TYPE_DATE:
                # https://weka.wikispaces.com/ARFF+(stable+version)#Examples-The%20@attribute%20Declarations-Date%20attributes
                print('@attribute %s date "%s"' % (self.esc(a), self.attribute_data.get(a, DEFAULT_DATE_FORMAT)), file=fout)
            else:
                raise Exception("Type " + at + " not supported for writing!")
        if isinstance(fout, StringIO) and close:
            return fout.getvalue()

    def write(self,
        fout=None,
        fmt=SPARSE,
        schema_only=False,
        data_only=False):
        """
        Write an arff structure to a string.
        """
        assert not (schema_only and data_only), 'Make up your mind.'
        assert fmt in FORMATS, 'Invalid format "%s". Should be one of: %s' % (fmt, ', '.join(FORMATS))
        close = False
        if fout is None:
            close = True
            fout = StringIO()
        if not data_only:
            print('% ' + re.sub("\n", "\n% ", '\n'.join(self.comment)), file=fout)
            print("@relation " + self.relation, file=fout)
            self.write_attributes(fout=fout)
        if not schema_only:
            print("@data", file=fout)
            for d in self.data:
                line_str = self.write_line(d, fmt=fmt)
                if line_str:
                    print(line_str, file=fout)
        if isinstance(fout, StringIO) and close:
            return fout.getvalue()

    def esc(self, s):
        """
        Escape a string if it contains spaces.
        """
        return ("\'" + s + "\'").replace("''", "'")

    def define_attribute(self, name, atype, data=None):
        """
        Define a new attribute. atype has to be one of 'integer', 'real', 'numeric', 'string', 'date' or 'nominal'.
        For nominal attributes, pass the possible values as data.
        For date attributes, pass the format as data.
        """
        self.attributes.append(name)
        assert atype in TYPES, "Unknown type '%s'. Must be one of: %s" % (atype, ', '.join(TYPES),)
        self.attribute_types[name] = atype
        self.attribute_data[name] = data

    def parseline(self, l):
        if self.state == 'comment':
            if l and l[0] == '%':
                self.comment.append(l[2:])
            else:
                self.comment = '\n'.join(self.comment)
                self.state = 'in_header'
                self.parseline(l)
        elif self.state == 'in_header':
            ll = l.lower()
            if ll.startswith('@relation '):
                self.__parse_relation(l)
            if ll.startswith('@attribute '):
                self.__parse_attribute(l)
            if ll.startswith('@data'):
                self.state = 'data'
        elif self.state == 'data':
            if l and l[0] != '%':
                self._parse_data(l)

    def __parse_relation(self, l):
        l = l.split()
        self.relation = l[1]

    def __parse_attribute(self, l):
        p = re.compile(r'[a-zA-Z_][a-zA-Z0-9_\-\[\]]*|\{[^\}]*\}|\'[^\']+\'|\"[^\"]+\"')
        l = [s.strip() for s in p.findall(l)]
        name = l[1]
        name = STRIP_QUOTES_REGEX.sub('', name)
        atype = l[2]#.lower()
        if atype == TYPE_INTEGER:
            self.define_attribute(name, TYPE_INTEGER)
        elif (atype == TYPE_REAL or atype == TYPE_NUMERIC):
            self.define_attribute(name, TYPE_NUMERIC)
        elif atype == TYPE_STRING:
            self.define_attribute(name, TYPE_STRING)
        elif atype == TYPE_DATE:
            data = None
            if len(l) >= 4:
                data = STRIP_QUOTES_REGEX.sub('', l[3])
            self.define_attribute(name, TYPE_DATE, data=data)
        elif atype[0] == '{' and atype[-1] == '}':
            values = [s.strip() for s in atype[1:-1].split(',')]
            self.define_attribute(name, TYPE_NOMINAL, values)
        else:
            raise NotImplementedError("Unsupported type " + atype + " for attribute " + name + ".")

    def _parse_data(self, l):
        if isinstance(l, basestring):
            l = l.strip()
            if l.startswith('{'):
                assert l.endswith('}'), 'Malformed sparse data line: %s' % (l,)
                assert not self.fout, NotImplemented
                dline = {}
                parts = re.split(r'(?<!\\),', l[1:-1])
                for part in parts:
                    index, value = re.findall(r'(^[0-9]+)\s+(.*)$', part.strip())[0]
                    index = int(index)
                    if value[0] == value[-1] and value[0] in ('"', "'"):
                        # Strip quotes.
                        value = value[1:-1]
                    # TODO:0 or 1-indexed? Weka uses 0-indexing?
                    #name = self.attributes[index-1]
                    name = self.attributes[index]
                    ValueClass = TYPE_TO_CLASS[self.attribute_types[name]]
                    if value == MISSING:
                        dline[name] = Str(value)
                    else:
                        dline[name] = ValueClass(value)
                self.data.append(dline)
                return
            else:
                # Convert string to list.
                l = [s.strip() for s in l.split(',')]
        elif isinstance(l, dict):
            assert len(l) == len(self.attributes), \
                "Sparse data not supported."
            # Convert dict to list.
            #l = dict((k,v) for k,v in l.iteritems())
            # Confirm complete feature name overlap.
            assert set(self.esc(a) for a in l) == \
                set(self.esc(a) for a in self.attributes)
            l = [l[name] for name in self.attributes]
        else:
            # Otherwise, confirm list.
            assert isinstance(l, (tuple, list))
        if len(l) != len(self.attributes):
            print("Warning: line %d contains %i values but it should contain %i values" % (self.lineno, len(l), len(self.attributes)))
            return 

        datum = []
        for n, v in zip(self.attributes, l):
            at = self.attribute_types[n]
            if v == MISSING:
                datum.append(v)
            elif at == TYPE_INTEGER:
                datum.append(int(v))
            elif at in (TYPE_NUMERIC, TYPE_REAL):
                datum.append(Decimal(str(v)))
            elif at == TYPE_STRING:
                datum.append(v)
            elif at == TYPE_NOMINAL:
                if v in self.attribute_data[n]:
                    datum.append(v)
                else:
                    raise Exception('Incorrect value %s for nominal attribute %s' % (v, n))
        if self.fout:
            # If we're streaming out data, then don't even bother saving it to
            # memory and just flush it out to disk instead.
            line_str = self.write_line(datum)
            if line_str:
                print(line_str, file=self.fout)
            self.fout.flush()
        else:
            self.data.append(datum)

    def __print_warning(self, msg):
        print(('Warning (line %d): ' % self.lineno) + msg)

    def dump(self):
        """Print an overview of the ARFF file."""
        print("Relation " + self.relation)
        print("  With attributes")
        for n in self.attributes:
            if self.attribute_types[n] != TYPE_NOMINAL:
                print("    %s of type %s" % (n, self.attribute_types[n]))
            else:
                print("    " + n + " of type nominal with values " + ', '.join(self.attribute_data[n]))
        for d in self.data:
            print(d)
    
    def set_class(self, name):
        assert name in self.attributes
        self.attributes.remove(name)
        self.attributes.append(name)
    
    def set_nominal_values(self, name, values):
        assert name in self.attributes
        assert self.attribute_types[name] == TYPE_NOMINAL
        self.attribute_data.setdefault(name, set())
        self.attribute_data[name] = set(self.attribute_data[name])
        self.attribute_data[name].update(values)
    
    def alphabetize_attributes(self):
        """
        Orders attributes names alphabetically, except for the class attribute, which is kept last.
        """
        self.attributes.sort(key=lambda name: (name == self.class_attr_name, name))
    
    def append(self, line, schema_only=False, update_schema=True):
        schema_change = False
        if isinstance(line, dict):
            # Validate line types against schema.
            if update_schema:
                for k, v in list(line.items()):
                    prior_type = self.attribute_types.get(k, v.c_type if isinstance(v, Value) else None)
                    if not isinstance(v, Value):
                        if v == MISSING:
                            v = Str(v)
                        else:
                            print('prior_type:', prior_type, k, v)
                            v = TYPE_TO_CLASS[prior_type](v)
                    if v.value != MISSING:
                        assert prior_type == v.c_type, \
                            ('Attempting to set attribute %s to type %s but it is already defined as type %s.') % (k, prior_type, v.c_type)
                    if k not in self.attribute_types:
                        if self.fout:
                            # Remove feature that violates the schema
                            # during streaming.
                            if k in line:
                                del line[k]
                        else:
                            self.attribute_types[k] = v.c_type
                            self.attributes.append(k)
                            schema_change = True
                    if isinstance(v, Nominal):
                        if self.fout:
                            # Remove feature that violates the schema
                            # during streaming.
                            if k not in self.attributes:
                                if k in line:
                                    del line[k]
                            elif v.value not in self.attribute_data[k]:
                                if k in line:
                                    del line[k]
                        else:
                            self.attribute_data.setdefault(k, set())
                            if v.value not in self.attribute_data[k]:
                                self.attribute_data[k].add(v.value)
                                schema_change = True
                    if v.cls:
                        if self.class_attr_name is None:
                            self.class_attr_name = k
                        else:
                            assert self.class_attr_name == k, \
                                ('Attempting to set class to "%s" when it has already been set to "%s"') % (k, self.class_attr_name)
                                    
                    # Ensure the class attribute is the last one listed,
                    # as that's assumed to be the class unless otherwise specified.
                    if self.class_attr_name:
                        try:
                            self.attributes.remove(self.class_attr_name)
                            self.attributes.append(self.class_attr_name)
                        except ValueError:
                            pass
                    
                if schema_change:
                    assert not self.fout, 'Attempting to add data that doesn\'t match the schema while streaming.'
                    
            if not schema_only:
                # Append line to data set.
                if self.fout:
                    line_str = self.write_line(line)
                    if line_str:
                        print(line_str, file=self.fout)
                else:
                    self.data.append(line)
        else:
            assert len(line) == len(self.attributes)
            self._parse_data(line)
