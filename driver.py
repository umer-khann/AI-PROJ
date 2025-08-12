import msgParser
import carState
import carControl
import numpy as np
import tensorflow as tf # type: ignore
import joblib

class Driver(object):
    """TORCS driver with scaling, PD-steering, smoothing, edge-detected RPM gear shifting, and reverse-on-stuck."""

    def __init__(self, stage):
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # Load model and scaler
        model_path  = "model/torcs_nn_model.keras"
        scaler_path = "preprocessing/processed/scaler.pkl"
        self.model  = tf.keras.models.load_model(model_path, compile=False)
        scaler      = joblib.load(scaler_path)

        # Extract StandardScaler parameters
        self.scaler_mean  = tf.constant(scaler.mean_,  dtype=tf.float32)
        self.scaler_scale = tf.constant(scaler.scale_, dtype=tf.float32)

        # Input dimension and buffer
        self.input_dim = self.model.input_shape[1]
        self.buf       = np.zeros((1, self.input_dim), dtype=np.float32)

        # Steering PD gains and smoothing
        self.steer_kp      = 0.15
        self.steer_kd      = 0.05
        self.steer_alpha   = 0.2
        self.prev_trackpos = 0.0
        self.prev_steer    = 0.0

        # RPM thresholds for gear shifting with hysteresis
        self.gear_up_rpm    = 6500
        self.gear_down_rpm  = 3000
        self.prev_gear      = 1
        self.prev_rpm       = 0.0

        # Stuck detection & reverse logic
        self.stuck_count         = 0
        self.stuck_threshold     = 20  # ticks before reverse
        self.reverse_steps       = 30  # ticks to reverse
        self.reverse_counter     = 0
        self.reverse_active      = False

        # Forward recovery after reverse
        self.forward_recovery_steps    = 30
        self.forward_recovery_counter  = 0

        @tf.function
        def predict_fn(x):
            x_norm = (x - self.scaler_mean) / self.scaler_scale
            p      = self.model(x_norm, training=False)[0]
            return p[0], p[1], p[2], p[4]

        self.predict_fn = predict_fn
        # Warm up graph
        _ = self.predict_fn(tf.zeros((1, self.input_dim), dtype=tf.float32))

    def init(self):
        angles = [0]*19
        for i in range(5):
            angles[i]      = -90 + 15*i
            angles[18-i]   =  90 - 15*i
        for i in range(5,9):
            angles[i]      = -20 + 5*(i-5)
            angles[18-i]   =  20 - 5*(i-5)
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        # Warm-up lap
        if self.state.getCurLapTime() < 0:
            self.control.setAccel(1.0)
            self.control.setGear(1)
            self.prev_rpm = 0.0
            return self.control.toMsg()

        speed = self.state.speedX
        # Handle forward recovery
        if self.forward_recovery_counter > 0:
            self.forward_recovery_counter -= 1
            ignore_stuck = True
        else:
            ignore_stuck = False

        # Stuck detection
        if not self.reverse_active and not ignore_stuck:
            if speed < 1.0:
                self.stuck_count += 1
            else:
                self.stuck_count = 0

        if not self.reverse_active and self.stuck_count > self.stuck_threshold:
            self.reverse_active      = True
            self.reverse_counter     = self.reverse_steps
            self.stuck_count         = 0

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
                # end reverse, start forward recovery
                self.reverse_active          = False
                self.forward_recovery_counter = self.forward_recovery_steps
                self.prev_gear               = 1

        # Build input buffer
        idx = 0
        for v in (self.state.angle, speed,
                  self.state.speedY, self.state.speedZ,
                  self.state.rpm, self.state.trackPos):
            self.buf[0, idx] = v; idx += 1
        for v in self.state.track[:18]:
            self.buf[0, idx] = v; idx += 1
        while idx < self.input_dim:
            self.buf[0, idx] = 0.0; idx += 1

        # Inference
        accel, brake, clutch, steer_raw = self.predict_fn(tf.constant(self.buf))
        self.control.setAccel(float(tf.clip_by_value(accel,0,1).numpy()))
        self.control.setBrake(float(tf.clip_by_value(brake,0,1).numpy()))
        self.control.setClutch(float(tf.clip_by_value(clutch,0,1).numpy()))

        # Steering PD + smoothing
        raw_s = float(tf.clip_by_value(steer_raw,-1,1).numpy())
        tp    = self.state.trackPos
        p_term = -self.steer_kp * tp
        d_term = -self.steer_kd * (tp - self.prev_trackpos)
        desired = np.clip(raw_s + p_term + d_term, -1, 1)
        steer_out = self.steer_alpha * desired + (1-self.steer_alpha)*self.prev_steer
        self.control.setSteer(float(steer_out))
        self.prev_trackpos = tp
        self.prev_steer    = steer_out

        # RPM edge-detected gear logic
        rpm   = self.state.rpm
        gear  = self.prev_gear
        # rising edge past up threshold
        if self.prev_rpm <= self.gear_up_rpm and rpm > self.gear_up_rpm and gear < 6:
            gear += 1
        # falling edge below down threshold
        elif self.prev_rpm >= self.gear_down_rpm and rpm < self.gear_down_rpm and gear > 1:
            gear -= 1
        self.control.setGear(gear)
        self.prev_gear = gear
        self.prev_rpm  = rpm

        return self.control.toMsg()

    def onShutDown(self): pass
    def onRestart(self): pass
