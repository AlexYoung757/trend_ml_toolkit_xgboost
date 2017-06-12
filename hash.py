#coding=utf-8

import hashlib
import sha3

def get_config():
    config = dict()
    config['base_path'] = '/home/lili/opcode-07'
    config['re_path'] = '/home/lili'


if __name__ == '__main__':

    """
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
    """
    x = hashlib.shake_128('11111').hexdi