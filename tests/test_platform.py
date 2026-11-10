"""Platform details of the runner: checking that a lock's owner is alive must not signal it.

On Windows ``os.kill(pid, 0)`` is no probe: signal 0 is CTRL_C_EVENT, which interrupts every
process of the console. A test run would stop here if the runner still used it.
"""

import os
import subprocess
import sys
import unittest

from mechanet.runner import _pid_alive


class ProcessCheckTests(unittest.TestCase):
    def test_a_process_is_seen_alive_or_gone_without_signalling_it(self):
        self.assertTrue(_pid_alive(os.getpid()))
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertTrue(_pid_alive(child.pid))
        finally:
            child.kill()
            child.wait()
        self.assertFalse(_pid_alive(child.pid))

    def test_an_unreadable_owner_keeps_the_lock(self):
        for value in ("not a pid", None, -5):
            self.assertTrue(_pid_alive(value))


if __name__ == "__main__":
    unittest.main()
