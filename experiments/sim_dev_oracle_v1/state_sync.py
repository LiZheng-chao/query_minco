"""SIM-only pairing of received native world-frame Odom/IMU. No retiming."""
from collections import deque
import copy
import math
import numpy as np
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation
from query_minco.integration.yopo_execution_v4.trajectory import digest

MAX_SKEW = .02
MAX_AGE = .2

class StateInvalid(ValueError):
    pass

def validate_sample(kind, data, stamp, now):
    try:
        if not math.isfinite(stamp) or not math.isfinite(now):
            raise StateInvalid('STATE_TIMESTAMP_NONFINITE')
        if now < stamp:
            raise StateInvalid('STATE_FUTURE_' + kind.upper())
        if now - stamp > MAX_AGE:
            raise StateInvalid('STATE_STALE_' + kind.upper())
        if data.get('frame') != 'world':
            raise StateInvalid('STATE_FRAME_' + kind.upper())
        keys = ('p', 'v') if kind == 'odom' else ('a_world',)
        for key in keys:
            a = np.asarray(data[key], dtype=float)
            if a.shape != (3,):
                raise StateInvalid('STATE_SHAPE_' + key)
            if not np.isfinite(a).all():
                raise StateInvalid('STATE_NONFINITE_' + key)
        if kind == 'odom':
            rotation(data['q'])
    except StateInvalid:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise StateInvalid('STATE_SOURCE_OR_QUATERNION_' + kind.upper())

class StatePairs:
    def __init__(self):
        self.buffers = {k: deque(maxlen=32) for k in ('odom', 'imu')}
        self.invalid = None
        self.last_pair = None

    def add(self, kind, data, stamp, now):
        if self.invalid:
            raise StateInvalid(self.invalid)
        try:
            if kind not in self.buffers:
                raise StateInvalid('STATE_SOURCE')
            validate_sample(kind, data, stamp, now)
            q = self.buffers[kind]
            if q and stamp < q[-1]['stamp']:
                raise StateInvalid('STATE_ROLLBACK_' + kind.upper())
            if q and stamp == q[-1]['stamp']:
                if data != q[-1]['data']:
                    raise StateInvalid('STATE_DUPLICATE_CONFLICT_' + kind.upper())
                return False
            q.append(dict(data=copy.deepcopy(data), stamp=stamp, receipt=now,
                          source='NATIVE_ROS_' + kind.upper()))
            return True
        except StateInvalid as ex:
            self.invalid = str(ex)
            raise

    def select(self, now):
        if self.invalid:
            raise StateInvalid(self.invalid)
        latest = {k: q[-1]['stamp'] if q else None for k, q in self.buffers.items()}
        context = dict(raw_latest_stamps=latest, raw_latest_skew=(abs(latest['odom']-latest['imu']) if all(t is not None for t in latest.values()) else None), source='RECEIVED_NATIVE_ODOM_IMU')
        for kind, q in self.buffers.items():
            if q:
                try:
                    validate_sample(kind, q[-1]['data'], q[-1]['stamp'], now)
                    if now < q[-1]['receipt']:
                        raise StateInvalid('STATE_RECEIPT_FUTURE_' + kind.upper())
                except StateInvalid as ex:
                    self.invalid = str(ex)
                    raise
        if not all(self.buffers.values()):
            return dict(context, status='STATE_PAIR_PENDING', reason='MISSING_STREAM')
        for odom in reversed(self.buffers['odom']):
            if not 0 <= now-odom['stamp'] <= MAX_AGE:
                continue
            matches = [i for i in self.buffers['imu'] if 0 <= now-i['stamp'] <= MAX_AGE and abs(odom['stamp']-i['stamp']) <= MAX_SKEW]
            if not matches:
                continue
            imu = min(matches, key=lambda i: (abs(odom['stamp']-i['stamp']), -i['stamp']))
            key = (odom['stamp'], imu['stamp'])
            if self.last_pair is not None and key[0] < self.last_pair[0]:
                continue
            if self.last_pair is not None and key[0] == self.last_pair[0]:
                # Never resubmit the same Odom state epoch or change its acceleration.
                if key != self.last_pair:
                    # A later, closer IMU cannot rewrite this submitted Odom epoch.
                    # Retain its ORIGINAL still-fresh received pair, not a false PENDING.
                    retained = next((i for i in matches if i['stamp'] == self.last_pair[1]), None)
                    if retained is None:
                        continue
                    imu = retained
                    key = self.last_pair
                status = 'STATE_PAIR_DUPLICATE'
            else:
                status = 'STATE_PAIR_READY'
            state = dict(odom['data'], a=list(imu['data']['a_world']))
            identity = digest(dict(odom_stamp=key[0], imu_stamp=key[1], state=state, source=context['source']))
            if status == 'STATE_PAIR_READY':
                self.last_pair = key
            return dict(context, status=status, reason='MATCHED', state=state, odom_stamp=key[0], imu_stamp=key[1], odom_receipt=odom['receipt'], imu_receipt=imu['receipt'], state_pair_id=identity, skew=abs(key[0]-key[1]))
        return dict(context, status='STATE_PAIR_PENDING', reason='NO_FRESH_PAIR_WITHIN_20MS')
