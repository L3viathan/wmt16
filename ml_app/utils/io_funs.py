import cPickle as pickle
import copy_reg, types
import os
import codecs

def is_file(fname):
    if os.path.isfile(fname):
        return True
    return False

def object_to_file(obj, fname):
    #register_instancemethod()
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)#protocl 0 default-text-based; binary mode for1,2,-1, HIGHGEST PROTOCOL is normal???

def file_to_object(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def register_instancemethod():
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def read_gz_file(path):
    zf = gzip.open(path, 'rb')
    contents = zf.read()
    zf.close()
    return contents

def gz_text_file_line_iter(path, encoding='utf8'):
    zf = gzip.open(path, 'rb')
    reader = codecs.getreader(encoding)
    contents = reader(zf)
    for line in contents.readlines():
        yield line
    zf.close()

def text_file_line_iter(path, encode='ascii'):
    #with codecs.open(path, 'rt', encoding=encode) as f:
    with open(path, 'rt') as f:
        for line in f:
            yield line 

def as_project_path(path):
    project_name = 'ml_app'
    return os.path.join(env_root(project_name), path)

def env_root(project_name):
    path, directory = os.path.split(os.path.abspath(__file__))

    while directory and directory != project_name:
        path, directory = os.path.split(path)

    if directory == project_name:
        return os.path.join(path, directory)
    else:
        raise Exception("Couldn't determine path to the project root.")
