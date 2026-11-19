"""Bounded development journal, with file IO outside the callback lock."""
from queue import Queue, Full
from threading import Thread
import json
import time
from query_minco.integration.yopo_execution_v4.lifecycle import Journal

class AsyncJournal(Journal):
    def __init__(self, path):
        super().__init__(path)
        self.queue = Queue(maxsize=4096)
        self.failure = None
        self.writer = Thread(target=self.write, daemon=True)
        self.writer.start()

    def write(self):
        try:
            while True:
                event = self.queue.get()
                if event is None:
                    break
                self.file.write(json.dumps(event, allow_nan=False)+'\n')
                if self.queue.empty():
                    self.file.flush()
            if self.failure:
                self.file.write(json.dumps({'kind':'AUDIT_FAILURE','reason':self.failure})+'\n')
            self.file.flush()
        except Exception as exc:
            self.failure = 'AUDIT_WRITER_FAILURE: '+repr(exc)

    def append(self, event):
        if self.closing or self.failure:
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except Full:
            self.failure = 'AUDIT_QUEUE_OVERFLOW'
            return False

    def close(self, unregister=()):
        with self.lock:
            self.closing = True
        errors = []
        for fn in unregister:
            try:
                fn()
            except Exception as exc:
                errors.append(repr(exc))
        try:
            self.queue.put(None, timeout=2.)
            self.writer.join(2.)
        except Full:
            errors.append('AUDIT_DRAIN_TIMEOUT')
        if self.writer.is_alive():
            return errors+['AUDIT_DRAIN_TIMEOUT']
        if self.failure:
            errors.append(self.failure)
        self.file.write(json.dumps(dict(kind="JOURNAL_CLOSED",monotonic=time.monotonic(),late_callbacks=self.late_callbacks,cleanup_errors=errors))+"\n")
        self.file.close()
        return errors
