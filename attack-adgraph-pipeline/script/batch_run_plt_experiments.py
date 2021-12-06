import subprocess
import time


for i in range(97):
    cmd = "python3 plt_load_page.py --id %d --plt-list %s" % (i, "~/plt.csv")
    print("Issuing the command to Shell: %s" % cmd)
    subprocess.Popen(cmd, shell=True)
    time.sleep(1.5)
