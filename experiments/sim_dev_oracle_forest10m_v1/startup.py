"""Bounded recovery for the isolated, non-integrating SIM startup."""
from query_minco.integration.yopo_startup_v3.readiness import Readiness

SENSORS = ('odom', 'imu', 'depth')

class Startup(Readiness):
    def __init__(self, limit, engine, clock, log):
        super().__init__(True, True, 'equilibrium_hover')
        self.limit, self.engine, self.clock, self.log = limit, engine, clock, log
        self.wait = None
        self.spent = 0.
        self.floor = {}
        self.generation = 0
        self.ack_floor = 0.

    def enter_wait(self, reason):
        plant = self.samples.get('plant')
        if (self.mode not in ('HELD', 'PREPARE', 'ACK') or self.last_steps
                or getattr(self.engine, 'arm_sent', False)
                or (plant and plant[2]['phase'] not in ('HELD', 'PREPARE', 'ACK'))):
            self.abort(reason)
            return
        if self.wait is not None:
            return
        self.wait = self.clock()
        self.generation += 1
        self.log('WAIT_FOR_SENSORS', None, {'reason': reason, 'generation': self.generation})
        with self.engine.lock:
            self.engine.actual = self.engine.actual_pair = self.engine.committed = self.engine.proposed = None
            self.engine.state_pending = True
            self.engine.arm_request_stamp = None
        if self.mode == 'ACK':
            self.transition('PREPARE')
        self.acks.clear()
        self.ack_floor = self.last_sent
        for k in (*SENSORS, 'request', 'raw'):
            previous = self.samples.pop(k, None)
            if previous:
                self.floor[k] = previous[0]
            self.counts.pop(k, None)

    def observe(self, kind, stamp, now, data):
        if self.mode == 'ABORT':
            return False
        if self.wait is not None and kind == 'plant' and (data.get('phase') in ('ARMED','ABORT') or data.get('integration_steps', 0)):
            self.abort('PLANT_PHASE_DURING_SENSOR_WAIT')
            return False
        if kind in self.floor and stamp < self.floor[kind]:
            self.abort(kind + '_BACKWARD_STAMP')
            return False
        if kind in self.floor and stamp == self.floor[kind]:
            return False
        if kind in SENSORS and self.mode in ('HELD', 'PREPARE', 'ACK'):
            prev = self.samples.get(kind)
            if now-stamp > .2 or (prev and (now-prev[1] > .2 or stamp-prev[0] > .2)):
                if prev and stamp < prev[0]:
                    self.abort(kind + '_BACKWARD_STAMP')
                    return False
                self.enter_wait(kind + '_STARTUP_GAP')
                old = self.samples.pop(kind, None)
                if old:
                    self.floor[kind] = old[0]
                    self.counts.pop(kind, None)
                if self.mode == 'ABORT' or now-stamp > .2:
                    return False
                if stamp <= self.floor.get(kind, 0.):
                    return False
        return super().observe(kind, stamp, now, data)

    def sensors_ready(self, now):
        return all(k in self.samples and 0 <= now-self.samples[k][0] <= .2
                   and 0 <= now-self.samples[k][1] <= .2 for k in SENSORS)

    def tick(self, now):
        if self.mode in ('HELD', 'PREPARE', 'ACK'):
            for k in SENSORS:
                if k in self.samples and (now-self.samples[k][0] > .2 or now-self.samples[k][1] > .2):
                    self.enter_wait(k + '_STALE')
                    break
        if self.wait is not None and self.mode != 'ABORT':
            elapsed = self.clock()-self.wait
            if self.spent+elapsed >= self.limit:
                self.abort('STARTUP_SENSOR_WAIT_TIMEOUT')
                return self.mode
            # Plant transport and phase remain authoritative throughout recovery.
            plant = self.samples.get('plant')
            if plant and (now-plant[0] > .2 or now-plant[1] > .2):
                self.abort('plant_STALE')
                return self.mode
            if not self.sensors_ready(now) or not all(self.counts.get(k, 0) >= 5 for k in SENSORS):
                return self.mode
            self.spent += elapsed
            self.wait = None
            self.log('STARTUP_SENSORS_RECOVERED', None, {'generation': self.generation})
        if self.mode == 'PREPARE' and self.generation:
            plant = self.samples.get('plant')
            if plant and plant[2]['phase'] == 'ACK' and plant[2]['final_command_stamp'] <= self.ack_floor:
                # The native Plant keeps reporting its last accepted final during recovery.
                # Do not treat that old command as a new prepare confirmation.
                if now-plant[0] <= .2 and now-plant[1] <= .2:
                    return self.mode
        return super().tick(now)

    def forward_allowed(self, now):
        allowed = super().forward_allowed(now)
        return self.wait is None and allowed
