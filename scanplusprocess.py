
import zmq
import struct
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import numpy as np
from collections import deque
from processframe import FrameProcessor
# 线程A：读取图像帧
def frame_producer():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:64179")

    # Subscribe to all topics
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("Listening for published messages...")

    metadata = None
    current_frame_number = 0
    while True:
        if pause_event.is_set():
            time.sleep(0.05)
            continue
        message = socket.recv()
        if len(message) == 40:
            # print("Received header")
            metadata = message
            # Unpack 5 doubles: 'd' = double (8 bytes), so '5d' = 5 doubles
            pixels_per_line, lines_per_frame, num_channels, timestamp, frame_number = struct.unpack('5d', metadata)
        if len(message) > 40:
            print("Received Frame")
            frame = message
            frame_buffer.append(np.frombuffer(frame, dtype=np.int16).reshape(int(pixels_per_line), 
                                                        int(lines_per_frame), 
                                                        int(num_channels), 
                                                        order = 'F'))
            # print("average pixel value: ", np.mean(np.frombuffer(frame, dtype=np.int16)))
            current_frame_number = current_frame_number + 1
        if current_frame_number > 10000: 
            break # or do keyboard interrupt once done
        time.sleep(0.01)

# 线程B：处理图像帧
def frame_processor(Params_post, filename_CNN, p):

    # image enhancement
    trt_path = 'Z:\LSR\DATA\checkpt\RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW\DeepIE_tensorRT_windows.trt'
    cnn_path = filename_CNN

    # cuda.init()
    # 初始化
    while True:
        with buffer_lock:
            if len(frame_buffer) >= FRAME_BUFFER_SIZE:
                print("开始初始化")
                pause_event.set()  # 暂停producer
                frame_list_afteradjust = []
                trace_list = []
                # fake a frame buffer
                import tifffile as tiff
                frame_buffer_fake = tiff.imread('block_001.tiff').astype(np.float32)
                # init FrameProcessor
                frame_processor = FrameProcessor(trt_path, cnn_path, 0, Params_post)
                # 初始化处理
                video_adjust, template, Masks, traces = frame_processor.process_frames_init(np.squeeze(np.array(frame_buffer_fake)),\
                                                                             batch_size=BATCH_SIZE,
                                                                            overlap_size=OVERLAP_SIZE)
                # 将处理后的帧存入缓冲区
                for frame in video_adjust:
                    frame_list_afteradjust.append(frame)

                break
        time.sleep(0.1)

    print("初始化完成")
    # 在线处理
    idx = len(frame_buffer)
    pause_event.clear()  # 恢复producer
    frame_list_afteradjust = []
    starttime = time.time()
    while True:
        with buffer_lock:
            if len(frame_buffer) >= idx + BATCH_SIZE + OVERLAP_SIZE * 2:
                frames_to_process = list(frame_buffer)[idx:idx + BATCH_SIZE + OVERLAP_SIZE * 2]
                idx += BATCH_SIZE
                # 在线处理
                video_adjust, traces = frame_processor.process_frames_online(np.squeeze(np.array(frames_to_process)), template, 
                                                            batch_size=BATCH_SIZE,
                                                            overlap_size=OVERLAP_SIZE, Masks=Masks)
                # 将处理后的帧存入缓冲区
                for frame in video_adjust:
                    frame_list_afteradjust.append(frame)
            else:
                # time.sleep(0.01)
                continue
            endtime = time.time()
            print(f"Processed {len(frame_list_afteradjust)} frames, time taken: {endtime - starttime:.2f} seconds")

if __name__ == "__main__":
    # 全局参数
    FRAME_HEIGHT = 512
    FRAME_WIDTH = 512
    FRAME_BUFFER_SIZE = 100
    BATCH_SIZE = 8
    OVERLAP_SIZE = 2

    Params_post={
                # minimum area of a neuron (unit: pixels).
                'minArea': 60, 
                # average area of a typical neuron (unit: pixels) 
                'avgArea': 180,
                # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
                'thresh_pmap': 180, 
                # values higher than "thresh_mask" times the maximum value of the mask are set to one.
                'thresh_mask': 0.4, 
                # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
                'thresh_COM0': 2, 
                # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
                'thresh_COM': 6, 
                # minimum IoU of two masks to be considered the same neuron
                'thresh_IOU': 0.5, 
                # minimum consume ratio of two masks to be considered the same neuron 
                'thresh_consume': 0.7, 
                # minimum consecutive number of frames of active neurons
                'cons': 4}
    filename_CNN = 'Z:\LSR\DATA\\2p_bench\suns\\0701\Weights\model_latest.pth'
    
    pause_event = threading.Event()

    # 使用 deque 实现共享缓冲区（线程安全操作要加锁）
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE + 1000)
    buffer_lock = threading.Lock()

    # 启动线程
    producer_thread = threading.Thread(target=frame_producer, daemon=True)
    processor_thread = threading.Thread(
        target=frame_processor,
        args=(Params_post, filename_CNN, None),
        daemon=True
    )

    producer_thread.start()
    processor_thread.start()

    # 保持主线程运行
    while True:
        time.sleep(1)








