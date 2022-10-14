import os
import moxing as mox
OBS_PATH = 's3://bucket-cneast4/zhangqi'
HOME_PATH = '/cache'
CODE_PATH = 'mae-uniformity'

mox.file.copy_parallel(os.path.join(OBS_PATH, CODE_PATH), os.path.join(HOME_PATH, CODE_PATH))
mox.file.copy_parallel('s3://bucket-cneast4/zhangqi/mae-main/data/imagenet100','/cache/mae-uniformity/data/imagenet100')
import argparse
parser = argparse.ArgumentParser(description='Multi-Node Params')
parser.add_argument('--log', default='log1')
parser.add_argument('--beta', default=0)
parser.add_argument('--bs', default=128)
parser.add_argument('--tau',default=0.5)
env_args, unparsed = parser.parse_known_args()

os.system('cd /cache/'+ CODE_PATH + ' && bash ' +  'pretrain.sh '+str(env_args.beta)+' '+ str(env_args.bs)+ ' ' + str(env_args.tau))


CODE_PATH = 'mae-uniformity/temp_dir'
CODE_PATH2 = 'mae-uniformity/output_dir/'+env_args.log
mox.file.copy_parallel(os.path.join(HOME_PATH, CODE_PATH), os.path.join(OBS_PATH, CODE_PATH2))
