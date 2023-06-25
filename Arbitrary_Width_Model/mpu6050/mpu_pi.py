import threading
import time
import torch
import torch.utils.data as Data
import numpy as np
import os
import smbus
from time import sleep
import tkinter as tk
from model.baseline_wisdm import *
from tkinter import *
# select the correct i2c bus for this revision of Raspberry Pi
revision = ([l[12:-1] for l in open('/proc/cpuinfo', 'r').readlines() if l[:8] == "Revision"] + ['0000'])[0]
bus = smbus.SMBus(1 if int(revision, 16) >= 4 else 0)

# MPU6050 constants
EARTH_GRAVITY_MS2 = 9.80665
SCALE_MULTIPLIER = 16384.0

PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')
class MPU6050:
    address = None

    def __init__(self, address=0x68):
        self.address = address
        self.setSampleRate(100)
        self.setRange(2)
        self.enableMeasurement()

    def enableMeasurement(self):
        bus.write_byte_data(self.address, PWR_MGMT_1, 0)

    def setSampleRate(self, rate):
        div = int(8000 / rate - 1)
	  bus.write_byte_data(self.address, SMPLRT_DIV, div)

    def setRange(self, range):
        if range == 2:
            bus.write_byte_data(self.address, ACCEL_CONFIG, 0x00)
        elif range == 4:
            bus.write_byte_data(self.address, ACCEL_CONFIG, 0x08)
        elif range == 8:
            bus.write_byte_data(self.address, ACCEL_CONFIG, 0x10)
        elif range == 16:
            bus.write_byte_data(self.address, ACCEL_CONFIG, 0x18)

    def getAxes(self, gforce=False):
        bytes = bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 6)

        x = bytes[0] << 8 | bytes[1]
        if x & 0x8000:
            x -= 65536

        y = bytes[2] << 8 | bytes[3]
        if y & 0x8000:
            y -= 65536

        z = bytes[4] << 8 | bytes[5]
        if z & 0x8000:
            z -= 65536

        x = x / SCALE_MULTIPLIER
        y = y / SCALE_MULTIPLIER
        z = z / SCALE_MULTIPLIER

        if gforce == False:
            x = x * EARTH_GRAVITY_MS2
            y = y * EARTH_GRAVITY_MS2
            z = z * EARTH_GRAVITY_MS2

        return {"x": x, "y": y, "z": z}


mpu6050_data = [0, 0, 0]
mpu6050_sum_data = []

def collect_data():
    mpu6050 = MPU6050()
    while True:
	  axes = mpu6050.getAxes(False)
        mpu6050_data[0] = axes['x']
        mpu6050_data[1] = axes['y']
        mpu6050_data[2] = axes['z']
        mpu6050_sum_data.append(mpu6050_data)
        if len(mpu6050_sum_data) > 200:
            time.sleep(0.02)
            del (mpu6050_sum_data[:-200])
def gui():
   model = tf.keras.models.load_model('./model_save/wisdm/net0.9818016378525932_129.h5')

    def run_model():
        sensor_data = np.array(adxl345_sum_data[-200:])
        data_x = sensor_data.reshape(-1, 1, 200, 3)  # (N, C, H, W) (7352, 1, 128, 9)
        inputs = tf.convert_to_tensor(data_x, dtype=tf.float32)

        outputs = model(inputs, training=False)
        preds = tf.argmax(outputs, axis=1)
        print(preds.numpy())
        return preds

    def changeImage(preds):
        if preds == 0:
            label_img.configure(image=img_gif0)
            preds = run_model()
            label_img.after(1, changeImage, preds)
        elif preds == 1:
            label_img.configure(image=img_gif1)
            preds = run_model()
            label_img.after(1, changeImage, preds)
        elif preds == 2:
            label_img.configure(image=img_gif2)
            preds = run_model()
            label_img.after(1, changeImage, preds)
        elif preds == 3:
            label_img.configure(image=img_gif3)
            preds = run_model()
            label_img.after(1, changeImage, preds)
        elif preds == 4:
            label_img.configure(image=img_gif4)
            preds = run_model()
	      label_img.after(1, changeImage, preds)
        elif preds == 5:
            label_img.configure(image=img_gif5)
            preds = run_model()
            label_img.after(1, changeImage, preds)
  
    while True:
        top = tk.Tk()

        top.title("HAR_demo")
        width = 260
        height = 380
        top.geometry(f'{width}x{height}')

        img_gif = tk.PhotoImage(file='./动作/问号.gif')
        img_gif0 = tk.PhotoImage(file='./动作/走.gif')
        img_gif1 = tk.PhotoImage(file='./动作/上楼.gif')
        img_gif2 = tk.PhotoImage(file='./动作/下楼.gif')
        img_gif3 = tk.PhotoImage(file='./动作/坐.gif')
        img_gif4 = tk.PhotoImage(file='./动作/站立.gif')
        img_gif5 = tk.PhotoImage(file='./动作/躺.gif')

        label_img = tk.Label(top, image=img_gif)
        label_img.place(x=30, y=30)  # 30  120
        preds = run_model()
        label_img.after(1, changeImage, preds)
        top.update_idletasks()

        ime_ = tk.Label(top, text="Human Activity Recognition").place(x=50, y=280)
        #time_2 = tk.Label(top, text=HAR_List[act]).place(x=170, y=280)
        #time_2.after(1, changeText, preds)
        button = tk.Button(top, text='Start')
        button.place(x=50, y=350)  # button.place(x=60, y=330)

        button_Clear = tk.Button(top, text='Stop')
        button_Clear.place(x=150, y=350)  # button_Clear.place(x=150, y=330)
        top.mainloop()


thread_1 = threading.Thread(target=collect_data, name="T1")
thread_2 = threading.Thread(target=gui, name="T2")

thread_1.start()
thread_2.start()