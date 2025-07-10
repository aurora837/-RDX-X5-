# cd tools/iot_car && python OneKeyConfig.py
import sys
import os
import time
import shutil
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../iot_car'))
import AccessToken

app_id = "your app id"
app_key = "your app key"
room_id = "your room id"
base_user_id = "your base user id"
instances = 4
build_dir='build'

proj_path = os.path.join(os.path.dirname(__file__), '../../')
build_path = os.path.join(proj_path, build_dir)
print("build_path:{}".format(build_path))
yuv_file_path = os.path.join(build_path, '1280X720X15XI420.yuv')
pcm_file_path = os.path.join(build_path, '48000-stereo-s16le.pcm')
exe_file_path = os.path.join(build_path, 'rtccli')
libVolcEngineRTC_so_file_path = os.path.join(build_path, 'libVolcEngineRTC.so')
libRTCFFmpeg_so_file_path = os.path.join(build_path, 'libRTCFFmpeg.so')
src_file_array =[yuv_file_path,pcm_file_path,exe_file_path,libVolcEngineRTC_so_file_path,libRTCFFmpeg_so_file_path]

config_file_path = os.path.join(build_path, 'config.json')
with open(config_file_path,'r')as fp:
    config_json_data = json.load(fp)
    #set room id
    config_json_data['app_id'] = app_id
    #set room id
    config_json_data['room_id'] = room_id    
for i in range(0,instances):
    user_id="{}_{}".format(base_user_id,i)
    #set user id
    config_json_data['user_id'] = user_id
    
    user_path = os.path.join(build_path,user_id)
    user_config_file_path=os.path.join(user_path,'config.json')
    isExist = os.path.exists(user_path)
    if isExist:
        shutil.rmtree(user_path)
    os.makedirs(user_path)
    #copy files
    for file in src_file_array:
        shutil.copy(file,user_path)
    #get token
    token = AccessToken.AccessToken(app_id, app_key, room_id, user_id)
    token.add_privilege(AccessToken.PrivSubscribeStream, 0)
    token.add_privilege(AccessToken.PrivPublishStream, int(time.time()) + 3600)
    token.expire_time(int(time.time()) + 3600)
    s = token.serialize()
    #set token
    config_json_data['token'] = s
    print("{}:{}".format(user_id,s))
    t = AccessToken.parse(s)
    print(t.verify(app_key))
    #set video device index
    config_json_data['video_device_index'] = i
    #save config.json
    with open(user_config_file_path,'a')as fp:
        json.dump(config_json_data,fp,sort_keys=True, indent=4, separators=(',', ':'))
