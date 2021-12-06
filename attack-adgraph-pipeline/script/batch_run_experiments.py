import subprocess
import time

INTERVAL = 25

browser_id = 0
for i in range(3000, 5000, INTERVAL):
    cmd = "sh run_pipeline.sh attack %d %d all %d" % (i, i + INTERVAL, browser_id)
    print("Issuing the command to Shell: %s" % cmd)
    subprocess.Popen(cmd, shell=True)
    browser_id += 1
    time.sleep(1.5)
