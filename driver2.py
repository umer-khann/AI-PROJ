import msgParser
import carState
import carControl
import numpy as np
import tensorflow as tf  # type: ignore
import joblib

class Driver(object):
    """TORCS driver with scaling, reverse-on-stuck,
       and gear: solely NN prediction."""

    def __init__(self, stage):
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # Load model and scaler
        model_path  = "model/torcs_nn_model.keras"
        scaler_path = "preprocessing/processed/scaler.pkl"
        self.model  = tf.keras.models.load_model(model_path, compile=False)
        scaler      = joblib.load(scaler_path)

        # Extract scaler params
        self.scaler_mean  = tf.constant(scaler.mean_,  dtype=tf.float32)
        self.scaler_scale = tf.constant(scaler.scale_, dtype=tf.float32)

        # Buffers
        self.input_dim = self.model.input_shape[1]
        self.buf       = np.zeros((1, self.input_dim), dtype=np.float32)

        # Reverse logic params
        self.stuck_count        = 0
        self.stuck_threshold    = 20
        self.reverse_steps      = 60
        self.reverse_counter    = 0
        self.reverse_active     = False
        self.forward_recovery_steps   = 30
        self.forward_recovery_counter = 0

        @tf.function
        def predict_fn(x):
            x_norm = (x - self.scaler_mean) / self.scaler_scale
            p      = self.model(x_norm, training=False)[0]
            return p[0], p[1], p[2], p[3], p[4]

        self.predict_fn = predict_fn
        _ = self.predict_fn(tf.zeros((1, self.input_dim), dtype=tf.float32))

    def init(self):
        angles = [0] * 19
        for i in range(5):
            angles[i]    = -90 + 15 * i
            angles[18 - i] = 90 - 15 * i
        for i in range(5, 9):
            angles[i]    = -20 + 5 * (i - 5)
            angles[18 - i] = 20 - 5 * (i - 5)
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        # Warm-up lap
        if self.state.getCurLapTime() < 0:
            self.control.setAccel(1.0)
            self.control.setGear(1)
            return self.control.toMsg()

        speed = self.state.speedX

        # Forward recovery
        if self.forward_recovery_counter > 0:
            self.forward_recovery_counter -= 1
            ignore_stuck = True
        else:
            ignore_stuck = False

        # Stuck detection
        if not self.reverse_active and not ignore_stuck:
            self.stuck_count = self.stuck_count + 1 if speed < 1.0 else 0
        if not self.reverse_active and self.stuck_count > self.stuck_threshold:
            self.reverse_active = True
            self.reverse_counter = self.reverse_steps
            self.stuck_count = 0

        # Reverse mode
        if self.reverse_active:
            if self.reverse_counter > 0:
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
                self.control.setClutch(0.0)
                self.control.setGear(-1)
                self.control.setSteer(0.0)
                self.reverse_counter -= 1
                return self.control.toMsg()
            else:
                self.reverse_active = False
                self.forward_recovery_counter = self.forward_recovery_steps
                self.control.setAccel(0.0)
                self.control.setBrake(0.0)
                self.control.setGear(1)
                self.control.setSteer(0.0)
                return self.control.toMsg()

        # Build feature buffer
        idx = 0
        for v in (self.state.angle, speed, self.state.speedY,
                  self.state.speedZ, self.state.rpm, self.state.trackPos):
            self.buf[0, idx] = v
            idx += 1
        for v in self.state.track[:18]:
            self.buf[0, idx] = v
            idx += 1
        while idx < self.input_dim:
            self.buf[0, idx] = 0.0
            idx += 1

        # Inference
        accel_t, brake_t, clutch_t, gear_t, steer_t = self.predict_fn(tf.constant(self.buf))

        # Debug print of predictions
        print(f"[Predictions] Accel: {accel_t.numpy():.3f}, Brake: {brake_t.numpy():.3f}, "
              f"Clutch: {clutch_t.numpy():.3f}, Gear: {gear_t.numpy():.3f}, Steer: {steer_t.numpy():.3f}")

        accel  = float(accel_t.numpy())
        brake  = float(brake_t.numpy())
        clutch = float(clutch_t.numpy())
        pred_gear = int(np.clip(np.round(gear_t.numpy()), 1, 6))
        steer  = float(steer_t.numpy())

        # Controls
        self.control.setAccel(np.clip(accel, 0, 1))
        self.control.setBrake(np.clip(brake, 0, 1))
        self.control.setClutch(np.clip(clutch, 0, 1))
        self.control.setSteer(np.clip(steer, -1, 1))
        self.control.setGear(pred_gear)

        return self.control.toMsg()

    def onShutDown(self): pass
    def onRestart(self): pass
