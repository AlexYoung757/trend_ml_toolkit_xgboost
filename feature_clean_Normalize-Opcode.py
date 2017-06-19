#coding=utf-8
import os
import argparse
import pprint

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputFolder', required=True)
    parser.add_argument('-o', '--outputFolder', required=True)
    parser.add_argument('-is', '--instructFile', required=True)
    parser.add_argument('-rs','--remove',default=1)
    parser.add_argument('-g','--group',default=1)
    return parser.parse_args()

def print_args(args):
    print args.inputFolder,args.outputFolder,args.instructFile,args.remove,args.group

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def handle_file(in_f_n):
    print("========================================")
    print(in_f_n)
    global args,instructions_prefixes,ops
    o_f_n = os.path.join(args.outputFolder,in_f_n)
    file_dir = os.path.split(o_f_n)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    # if not os.path.exists(o_f_n):

    legal_lines = []
    with open(in_f_n,'r') as in_f:
        lines = in_f.readlines()
        for line in lines:
            # print(line)
            tokens = ' '.join(line.split()).split(' ')
            # pprint.pprint(tokens[:5])
            # print(tokens)
            has_prefix = False
            if tokens[0] in instructions_prefixes:
                op_temp = tokens[1]
                has_prefix = True
            else:
                op_temp = tokens[0]
            # print 'op_temp : %s'%op_temp
            flag = False
            for op in ops:
                group_op = op.split(',')
                # 该组的操作符只有一个
                if len(group_op) == 1:
                    if op_temp == group_op[0]:
                        flag = True
                        break
                # 该组的操作符有多个
                else:
                    if op_temp in group_op:
                        flag = True
                        if int(args.group) == 1 and op_temp!=group_op[0]:
                            if has_prefix==True:
                                tokens[1] = group_op[0]
                            else:
                                tokens[0] = group_op[0]
                            line = ' '.join(tokens)
                        break
            if flag == True:
                # print('*  %s')%(line)
                line = line.replace('\r','').replace('\n','').replace(';','')
                legal_lines.append(line)
            else:
                print("*** %s is dirty")%op_temp
    # pprint.pprint(legal_lines[:3])
    final_legal_lines = []

    if int(args.remove) == 1:
        i = 0
        while(i<len(legal_lines)):
            final_legal_lines.append(legal_lines[i])
            j = i
            while(j<len(legal_lines) and legal_lines[j]==legal_lines[i]):
                j+=1
            i = j
        with open(o_f_n,mode='w') as o_f:
            for i in range(len(final_legal_lines)):
                o_f.writelines(final_legal_lines[i] + '\n')
                """
                if(i!=len(final_legal_lines)-1):
                    o_f.writelines(final_legal_lines[i])
                else:
                    o_f.writelines(final_legal_lines[i]+'\n')
                """
        return
    with open(o_f_n,mode='w') as o_f:
            for i in range(len(legal_lines)):
                o_f.writelines(legal_lines[i] + '\n')
                """
                if(i!=len(legal_lines)-1):
                    o_f.writelines(legal_lines[i])
                else:
                    o_f.writelines(legal_lines[i]+'\n')
                """
def handle_files():
    global input_files
    for f_n in input_files:
        handle_file(f_n)

def get_instructions():
    global ins_file_name
    with open(ins_file_name,mode='r') as ins_f:
        lines = ins_f.readlines()
    # pprint.pprint(lines[:10])
    # print(lines[1:10:1])
    # return lines
    ops = [op.replace('\n','').replace('\r','') for op in lines ]
    # pprint.pprint(ops)
    # print(len(ops))
    return ops
if __name__ == '__main__':
    args = arg_parser()
    # print_args(args)
    instructions_prefixes = ['lock','repne','repnz','rep','repe','repz']
    input_files = dirlist(args.inputFolder,[])
    ins_file_name = args.instructFile
    ops = get_instructions()
    # pprint.pprint(ops[:10])
    # exit()
    handle_files()

