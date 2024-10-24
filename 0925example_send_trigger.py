#! /usr/bin/env python  
#  -*- coding:utf-8 -*-
#
# Author: FANG Junying, fangjunying@neuracle.cn
#
# Versions:
# 	v0.1: 2020-02-25, orignal
#
# Copyright (c) 2020 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

from neuracle_lib.triggerBox import TriggerBox,PackageSensorPara
import time
# from psychopy import  visual, event,core


if __name__ == '__main__':

#    isTriggerIn = True
    isTriggerBox = True



    if isTriggerBox:
        ## example send triggers by TriggerBox
        triggerbox = TriggerBox("COM4")


        for j in range(1,21):
            for i in range(1, 251):
                print('send trigger: {0}'.format(i))
                triggerbox.output_event_data(i)
                time.sleep(0.01)


        triggerbox.output_event_data(255)