import os

resnets = [18, 50, 101, 152]
checkpoints = [100, 500, 1000, 2000]

for resnet in resnets:
    for check in checkpoints:
        cmd = 'python roc_analysis.py --resnet %d --load %d' % (resnet, check)
        print(cmd)
        os.system(cmd)

