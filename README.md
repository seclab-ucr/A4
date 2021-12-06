# Eluding ML-based Adblockers With Actionable Adversarial Examples

Hello! Thank you for evaluating our artifacts. We appreciate your time a lot and thus provide this detailed documentation for walkthrough purposes. Please follow the instructions below to reproduce the main results in the paper (Table 2 in the paper). 

**System Requirements**

We have only tested our artifacts on **Ubuntu 18.04.5 LTS**. We believe they are also compatible with newer versions of Ubuntu with minimum dependency adjustments, but have not tested yet. Most Python dependencies can be installed by `pip3 install -r requirements.txt` inside the `attack-adgraph-pipeline` repository (some extra packages might be needed depending on your system configurations). 

**Reproduction Steps**

**Step #1 -- download repositories**

After cloning this repo, you are supposed to observe the following directory structure, with all required files already in place.
```
├── AdGraphAPI
├── AdGraph-Ubuntu-16.04
├── attack-adgraph-pipeline
├── map_local_list_cent.csv
├── map_local_list.csv
├── map_local_list_dist.csv
├── mitmproxy
└── rendering_stream
```

**Step #2 -- set up proxies**

Recall that our attack is launched directly against webpages, which is technically implemented via HTML rewrites (i.e., structural and URL feature perturbations). For simulating real client-server environments (similar to what the deployers of A4 will face) and avoiding excessive engineering efforts, we rewrite local HTML files and pretend to be hosting websites that attempt to cloak ads in them. However, if we directly load local HTML files, several issues might arise (e.g., relative URLs in webpages cannot be interpreted correctly due to changed hostname). To tackle this challenge, we set up a MITM proxy that redirects requests to their originally intended server hostname so that resources can be loaded correctly. As discussed in Section 3.4 in the paper, we have two “mapping-back strategies'' that require two different proxies (due to different versions of the same webpage). In addition to them, the original (unmodified) version of the webpage also needs to be loaded and rendered at the beginning of handling each webpage. 

In the meantime, in order to realize concurrency for reducing the time consumption of your evaluations, we have prepared relevant code design in our pipeline so that multiple instances can be run simultaneously. Please follow this [mini tutorial](https://github.com/shitongzhu/mitmproxy#development-setup) to set up the virtual environment required for MITM proxy. After starting tmux (since we will need to run 11 * 2 + 2 = 24 proxies in different Shell sessions) with at least 24 panes, let us start all the proxy servers (assuming you are under `\home` directory) by following these three sub-steps (port numbers in `[XXXX-YYYY]` mean you will need to enumerate all integers in that range in different panes, including both ends): 

1. for proxies handling requests from unmodified webpages, type `cd ~/mitmproxy/ && . venv/bin/activate && mitmproxy --map-local-file ~/map_local_list.csv -p [6666-6667]`;
2. for proxies handling requests from centrally perturbed (“Centralized strategy” in Figure 5 in the paper) webpages, type `cd ~/mitmproxy/ && . venv/bin/activate && mitmproxy --map-local-file ~/map_local_list_cent.csv -p [7777-7787] --use-modified`;
3. for proxies handling requests from distributionally perturbed (“Distributed strategy” in Figure 5 in the paper) webpages, type `cd ~/mitmproxy/ && . venv/bin/activate && mitmproxy --map-local-file ~/map_local_list_dist.csv -p [7797-7807] --use-modified`. 

**Step #3 -- run the pipeline**

Now we are ready to run the attack pipeline. Please first enter the `script` directory under `attack-adgraph-pipeline` by typing `cd attack-adgraph-pipeline/script`. Then, start the Python script that launches the attack pipeline in batches: `python3 batch_run_experiments.py`. Note that this script will only launch the “`All`” (i.e., including all perturbable features) variant of A4 attack (as explained in Section 5.1) as we consider it as the main results in the paper. 

**Step #4 -- analyze the results**

Normally, the attack pipeline will take up to 48 hours (with the concurrency over 24 proxies) to finish. After all processes finish, we can now analyze the generated logs to compare their results with what was reported in the paper. Specifically, all logs generated during the execution of the attack pipeline will be dumped into the folder of `attack-adgraph-pipeline/report`. Let us use some simple commands to summarize the results in log files. 

First, we should merge all logs into one: `cat aug_pgd_attack* > all.log`; and then we can use `cat all.log | grep "SUCCESS" | wc -l` and `cat all.log | grep "FAIL" | wc -l` to count the numbers of successful cases and failed cases, respectively. Eventually, we simply use the formula `success_rate = num_success / (num_success + num_fail)` to calculate the evasion rate which can be directly compared with the results in Table 2 in the paper. It is worth noting that the `success_rate` can be slightly deviated from the number presented in the paper, because of the time lag between when we generated the dataset (a few months ago) and now.
