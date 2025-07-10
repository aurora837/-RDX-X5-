RTC 命令行Demo
=========

说明

Linux 命令行开源Demo，提供本地视频采集、本地音频采集、推视频文件流，音频流等。

需要安装依赖视频和音频的依赖OpenGl，PulseAudio

```
sudo apt-get install build-essential
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install pulseaudio libpulse-dev
```


最小依赖：
Cmake >= 3.13
构建方式

目录结构：
```
.
├── CMakeLists.txt  //cmake 工程配置文件
├── config.json //配置信息
├── README.md
├── resources //资源文件
│   ├── 1280X720X15XI420.yuv  //I420视频帧文件
│   └── 48000-stereo-s16le.pcm //PCM音频帧文件
├── src
│   ├── app_data_manager.cc  //全局数据管理类
│   ├── app_data_manager.h
│   ├── main.cc   //主函数
│   ├── rtc_engine_wrapper.cc //火山引擎包装类
│   ├── rtc_engine_wrapper.h
│   └── util
│       ├── argparser.cc //命令行参数解析类
│       ├── argparser.h
│       ├── json11   //json配置文件解析类
│       │   ├── json11.cpp
│       │   ├── json11.hpp
│       │   └── LICENSE.txt
│       ├── thread_loop.h //线程定时器
│       ├── util.cc  //
│       └── util.h
└── third_party
    ├── Linux 
    │   ├── VolcEngineRTC_arm  //arm sdk
    │   └── VolcEngineRTC_x86 //x86
    └── Windows
```

构建方式：
```
构建方式步骤一 构建Demo工程
打开终端窗口下，进入 QuickStart_Terminal_Demo 目录，执行 cmake -B build命令，在 build 目录下生成工程
步骤二 编译Demo工程
执行 cmake --build build 命令进行编译。
步骤三  获取 AppId 和临时 Token
关于在控制台获取 AppId 和临时 Token，参看获取 Appid。
临时 Token 生成时填写的房间 ID 和用户 ID 与 Demo 登陆页的房间 ID 和用户 ID 一致，若输入的房间 ID 或用户 ID 不一致，将无法进入正确房间与其他用户进行音视频通话。 
临时 Token 仅用于测试或跑通 Demo，你可以通过阅读密钥说明了解更多 Token 相关信息。
步骤四  修改Demo配置文件
进入build目录，修改config.json文件，程序启动时默认会从可执行程序所在路径加载该文件，进行rtc引擎相关的api调用，
{
    "app_id":"your app id",
    "token":"your token",
    "room_id":"your room id",
    "user_id":"your user id",
    "enable_audio":false,
    "enable_video":true,
    "enable_external_audio":false,
    "enable_external_video":false,
    "audio_file":"48000-stereo-s16le.pcm",
    "video_file":"1280X720X15XI420.yuv",
    "video_capture_config":{
        "width":1280,
        "height":720,
        "fps":30
    },
    "video_encoder_config":{
        "width":1280,
        "height":720,
        "fps":30,
        "max_bitrate":3000
    },
    "video_device_id":"print in console",
    "audio_device_id":"print in console"
}
1. 修改AppID。 使用在控制台获取的 AppId 覆盖 config.json 里  app_id  的值。
2. 修改TOKEN。 使用在控制台获取的临时 Token 覆盖 config.json 里 token 的值。
3. 修改房间ID，对应config.json 中 room_id 的值。
4. 修改用户ID，对应config.json 中 user_id的值。
步骤5 运行Demo
运行程序  ./rtccli 
默认config.json配置是关闭音频采集（enable_audio=false），开启视频内部采集(enable_video=true,enable_external_video=false)，
如果需要体验不同的rtc功能，可以修改配置文件，然后重新启动程序。config.json配置文件各字段功能含义如下表所示：
-------------------------------------------------------------------------------------
字段名称                |                   功能含义
-------------------------------------------------------------------------------------
app_id                |                   应用唯一标识
-------------------------------------------------------------------------------------
token                 |          服务器端对客户端用户身份进行鉴权
-------------------------------------------------------------------------------------
room_id               |                  房间ID
-------------------------------------------------------------------------------------
user_id               |                  用户id
-------------------------------------------------------------------------------------
enable_audio          |          打开或者关闭音频采集模块
-------------------------------------------------------------------------------------
enable_video          |          打开或者关闭视频采集模块
-------------------------------------------------------------------------------------
enable_external_audio | 是否开启外部音频采集。默认为false表示开启内部音频采集，这个也是SDK默认行为；
                      | true表示开启外部音频采集，对应调用setAudioSourceType，设置主流type类型为
                      | kAudioSourceTypeExternal。该字段只在打开音频采集模块功能即
                      | enable_video=true时生效。
---------------------------------------------------------------------------------------
enable_external_video | 是否开启外部视频采集。默认为false表示开启内部视频采集，这个也是SDK默认行为;
					  | true表示开启外部视频采集，对应调用setVideoSourceType，设置主流type类型为
					  | kVideoSourceTypeExternal。该字段只在打开视频采集模块功能即
					  |enable_video=true时生效。
----------------------------------------------------------------------------------------					  
audio_file            | 用于指定外部音频采集时使用到的pcm原始音频文件。当开启外部音频采集功能后，会读取
                      | 文件PCM数据，然后每隔10ms一次循环调用pushExternalAudioFrame将pcm数据编码后
					  | 推送给远端用户。
----------------------------------------------------------------------------------------
video_file            | 用于指定外部视频采集时使用到的yuv原始视频文件。当开启外部视频采集功能后，会读取
                      | 文件yuv文件数据，然后每隔1000/fps （其中fps为帧率，文件名称中会指定）毫秒循环调用
					  | pushExternalVideoFrame将原始视频数据编码后推送给远端用户。
-----------------------------------------------------------------------------------------
video_capture_config  | 设置视频采集相关参数，包括分辨率、帧率 。该字段只在视频内部采集开启情形下才会生效，
                      | 对应 setVideoCaptureConfig。
-----------------------------------------------------------------------------------------
video_encoder_config  | 设置视频编码相关参数，包括分辨率、帧率、码率，对应 setVideoEncoderConfig。 
					  | 码率默认推荐如下，码率范围1000-10000kbps
                      | - 1920*1080@60fps，推荐6000kbps
					  | - 1920*1080@30fps，推荐4000kbps
                      | - 1920*1080@15fps，推荐2500kbps
                      | - 1280*720@60fps，推荐4000kbps
                      | - 1280*720@30fps，推荐3000kbps，作为默认码率
                      | - 1280*720@15fps，推荐2000kbps
                      | - 640*480@30fps, 推荐1500kbps
                      | - 640*480@15fps, 推荐1000kbps 
------------------------------------------------------------------------------------------
audio_device_index    | 根据枚举音频设备时使用的索引，获取音频设备ID，调用setAudioCaptureDevice设置当前
                      | 音频内部采集时使用的音频设备.程序每次启动时会调用 enumerateAudioCaptureDevices 
					  | 枚举所有音频设备，并将设备索引、 ID 、名称打印并输出到终端,可以从终端日志中获取预设的
					  | 音频设备索引，写入配置文件，再次重启后生效。
-------------------------------------------------------------------------------------------
video_device_index    | 根据枚举视频设备时使用的索引，获取视频设备ID，调用setVideoCaptureDevice设置当前
                      | 视频内部采集时使用的视频设备 。程序每次启动时会调用 enumerateVideoCaptureDevices，
					  | 枚举所有视频设备，并将设备索引、 ID 、名称打印并输出到终端，可以从终端日志中获取预设的
					  | 视频设备索引，写入配置文件，再次重启后生效。
------------------------------------------------------------------------------------------
 启动多进程
 考虑到iot 车机场景应用，需要同时打开多个终端用户在相同房间推送多路流，而每次打开终端前都需要重复进行拷贝一份文件夹、获取Token等信息、修改配置文件这些重复操作。这里提供了一个一键配置python脚本文件，用于将这些重复性的配置准备工作用脚本一次性完成，你只需要分别打开终端，进入不同用户目录，启动进程即可。
该脚本文件存放在工程目录  tools/iot_car/OneKeyConfig.py 路径下，运行脚本前需要确保系统已经安装python 2.7+，具体操作步骤如下：
步骤1、2与上述相同
步骤3  修改tools/iot_car/OneKeyConfig.py  一键配置文件
必填的配置参数有：
1. app_id：在控制台上获取的应用 AppID。
2. app_key：在控制台上获取的 AppKey。
3. room_id：房间 ID。脚本默认启动多个用户进程需要在同一个房间。
4. base_user_id：基准用户 ID。执行python脚本后，会在cmake构建目录build底下，生成启动进程实例个数即 instances个目录，目录的命名方式是base_user_id基础上加上后缀 "_i"， 其中i 为索引下标，从0开始，instances-1结束。此外会将可执行程序、依赖库、资源文件拷贝到对应目录，并且会在该目录下生成一个配置文件config.json 其中user_id与目录命名方式保持一致。如设置base_user_id='user' 执行python脚本后，会在build目录下看到如下目录及文件：
build/
├── user_0
│   ├── 1280X720X15XI420.yuv
│   ├── 48000-stereo-s16le.pcm
│   ├── config.json
│   ├── libRTCFFmpeg.so
│   ├── libVolcEngineRTC.so
│   └── rtccli
├── user_1
│   ├── 1280X720X15XI420.yuv
│   ├── 48000-stereo-s16le.pcm
│   ├── config.json
│   ├── libRTCFFmpeg.so
│   ├── libVolcEngineRTC.so
│   └── rtccli
├── user_2
│   ├── 1280X720X15XI420.yuv
│   ├── 48000-stereo-s16le.pcm
│   ├── config.json
│   ├── libRTCFFmpeg.so
│   ├── libVolcEngineRTC.so
│   └── rtccli
└── user_3
    ├── 1280X720X15XI420.yuv
    ├── 48000-stereo-s16le.pcm
    ├── config.json
    ├── libRTCFFmpeg.so
    ├── libVolcEngineRTC.so
    └── rtccli
|-- 其它文件及文件夹
5. instances:   启动多进程实例个数
选填的参数：
6. build_dir: cmake工程构建目录，与步骤1生成的目录保持一致即可   
步骤4 执行一键配置python脚本 python tools/iot_car/OneKeyConfig.py
步骤5 打开终端，分别进入不同用户对应的目录，启动进程
如进入build/user_0 目录 运行命令行： ./rtccli
其他用户以此类推。如果需要对某个用户进程做特殊的修改，与前面介绍类似，修改配置文件config.json，重启进程即可。 