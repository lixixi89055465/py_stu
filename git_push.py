# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 23:33
# @Author  : nanji
# @Site    : 
# @File    : git_push.py
# @Software: PyCharm 
# @Comment :
# !/usr/bin/python3
# -*-coding:utf-8-*-

###############################################################
## Name       : git_push
## Author     :
## Date       :
## Description:
##
##
###############################################################

import sys
import os
import re
import argparse


def main():
    fl_list = []

    print("STEP.1: git pull")
    os.system("git pull")

    print("STEP.2: git status")
    os.system("git status")
    hand = os.popen("git status").readlines()
    for line in hand:
        line = line.strip()
        res = re.search(r"modified:\s+([\/\.\w]+)", line)
        if res:
            fl = res.group(1)
            fl_list.append(fl)

    print("STEP.3: sel push file")
    i = 0
    for fl in fl_list:
        print("    {0}: {1}".format(i, fl))
        i += 1
    if len(fl_list) == 0:
        print("NO FILE NEED PUSH, EXIT")
        sys.exit(0)

    # inp = input("please sel 0 1 .. or all (default is all): ")
    # cmt = input("please input commit (default is update): ")
    #
    add_line = "git add ."
    cmt_line = "git commit -m 'comment'"
    psh_line = "git push"
    # if inp == "" or inp == "all":
    #     for fl in fl_list:
    #         add_line += fl + " "
    # else:
    #     num_list = inp.split(" ")
    #     for num in num_list:
    #         add_line += fl_list[int(num)] + " "
    #
    # if cmt == "":
    #     cmt = "update"
    # cmt_line += "\"{}\"".format(cmt)

    # print("STEP.4: git add/commit/push")
    os.system(add_line)
    os.system(cmt_line)
    os.system(psh_line)


if __name__ == "__main__":
    print("%s is working" % __file__)
    main()
