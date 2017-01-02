#!/bin/sh
#v4l2-ctl --device=/dev/video0 --set-fmt-video=width=320,height=180,pixelformat=1

# SD 240p resolution
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=352,height=240,pixelformat=1

# **** use RPI GPU and write text
ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512  -f v4l2 -i /dev/video0  -c:v h264_omx -maxrate 768k  -vf "format=yuv444p,drawbox=y=ih-h:color=black@0.9:width=40:height=12:t=max,drawtext=fontfile=OpenSans-Regular.ttf:textfile=/tmpfs/meta.txt:reload=1:fontsize=10:fontcolor=white:x=0:y=(h-th-2),format=yuv420p" -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61  

#ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512  -f v4l2 -i /dev/video0 -preset veryfast -tune zerolatency -vprofile baseline -c:v libx264 -vf "format=yuv444p,drawbox=y=ih-h:color=black@0.9:width=40:height=12:t=max,drawtext=fontfile=OpenSans-Regular.ttf:textfile=/tmp/meta.txt:reload=1:fontsize=10:fontcolor=white:x=0:y=(h-th-2),format=yuv420p" -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61

# ** stream plain video to server 
fmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512  -f v4l2 -i /dev/video0  -c:v h264_omx -maxrate 1024k  -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61

# stream plain video to server 
#ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512 -copytb 0 -f v4l2 -vcodec h264 -i /dev/video0  -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61

# stream plain video to server and save 20 YUV frames per second in memory filesystem /tmpfs
#ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512 -copytb 0 -f v4l2 -vcodec h264 -i /dev/video0  -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61 -vcodec rawvideo -vf fps=20  -f image2 "/tmpfs/out-%2d".rgb


# **** stream plain video to server and save last YUV frame in /tmpfs/thumb.yuv
# ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512  -f v4l2 -i /dev/video0  -c:v h264_omx -maxrate 768k  -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61   -vcodec rawvideo  -an -updatefirst 1 -y -f image2 /tmpfs/thumb.jpg

 /usr/local/bin/ffmpeg -r 30 -use_wallclock_as_timestamps 1 -thread_queue_size 512  -f v4l2 -i /dev/video0  -c:v h264 -maxrate 1024k  -threads 4 -r 30 -g 60 -f flv rtmp://newstaging.cometa.io:12345/src/74DA388EAC61
