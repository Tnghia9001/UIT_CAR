import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO

# ------------- Add library ------------#
from keras.models import load_model
import utils
# --------------------------------------#

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)

# Tốc độ tối thiểu và tối đa của xe
MAX_SPEED = 35
MIN_SPEED = 25

# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED

# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        steering_angle = 0  # Góc lái hiện tại của xe
        speed = 0  # Vận tốc hiện tại của xe
        image = 0  # Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        # Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)

        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe

        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            # ------------------------------------------  Work space  ----------------------------------------------#
            image = utils.preprocess(image)
            image = np.array([image])
            # print('*****************************************************')
            steering_angle = float(model.predict(image, batch_size=1)) * 25

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giảm tốc độ
            else:
                speed_limit = MAX_SPEED

            sendBack_Speed = ((30 - 0.032*(steering_angle**2)) - speed) * 70

            # print(speed, steering_angle)
            # ------------------------------------------------------------------------------------------------------#
            # print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(steering_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':
    # -----------------------------------  Setup  ------------------------------------------#
    from model.LR_ASPP import LiteRASSP
    # Xây dựng model
    model = LiteRASSP((66, 200, 3)).build()
    model.load_weights('models_mobilenetv3/model-016.h5')
    # model = load_model('model-037.h5')
    # --------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
