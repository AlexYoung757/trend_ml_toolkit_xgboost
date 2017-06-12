#coding=utf-8

import hashlib


if __name__ == '__main__':

    x = hashlib.md5()
    x.update('hello, ')
    x.update('python')
    con0 = x.hexdigest()
    print(len(con0))
    con1 = hashlib.md5('hello, python').hexdigest()

    print(con0,con1)

    assert con0 == con1
    con2 = hashlib.md5('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx').hexdigest()
    print(len(con2))