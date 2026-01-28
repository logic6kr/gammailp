import re
import csv
from collections import defaultdict

# get the current folder 
import os
current_folder = os.path.dirname(os.path.abspath(__file__))
# get father folder 
project_folder = os.path.dirname(current_folder)
project_folder = os.path.dirname(project_folder)
# print(project_folder)

input_file = f"{project_folder}/cache/timeuf.md"
# input_file = "/root/code/gammaILP/cache/ILP_share_2.md"
output_file = f"{project_folder}/cache/timeuf.csv"

with open(input_file, "r") as f:
    content = f.read()
content = '\n'+content
namespaces = re.split(r'seconds', content)
results = []
task_counts = defaultdict(int)

for ns in namespaces[:]:
    # Extract cuda seed (case-insensitive, allow both CUDA and cuda)
    seed_match = re.search(r'cuda[_ ]seed[:=] ?([0-9]+)', ns, re.IGNORECASE)
    cuda_seed = seed_match.group(1) if seed_match else ""

    # Extract TEST RECALL value
    # recall_match = re.search(r'TEST RECALL\s*=\s*([0-9.]+)', ns)
    # Change the regex to allow either '=' or a blank space between 'TEST RECALL' and the value
    recall_match = re.search(r'TEST RECALL\s*[= ]\s*([0-9.]+)', ns)
    test_recall = recall_match.group(1) if recall_match else ""
    running_time_match = re.search(r'Running time:\s*([0-9.]+)\s*', ns)
    running_time = running_time_match.group(1) if running_time_match else ""

    # Extract task name (last line of block, strip comments)
    lines = [line.strip() for line in ns.strip().split('\n') if line.strip()]
    try:
        task_name = lines[0].split()[0] if lines else ""
    except:
        pass

    # Index for the same task name
    task_counts[task_name] += 1
    idx = task_counts[task_name]

    results.append({
        "task_name": task_name,
        "index": idx,
        "test_recall": test_recall,
        "cuda_seed": cuda_seed,
        "running_time": running_time
    })

with open(output_file, "w", newline="") as csvfile:
    fieldnames = ["task_name", "index", "test_recall", "cuda_seed","running_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Saved results to {output_file}")

# compute the average running time under the same task_name, and then insert to the csv file 
avg_times = defaultdict(list)
with open(output_file, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        task_name = row["task_name"]
        running_time = row["running_time"]
        if running_time:
            avg_times[task_name].append(float(running_time))
        else:
            avg_times[task_name].append(0.0)

with open(output_file, "a", newline="") as csvfile:
    fieldnames = ["task_name", "index", "test_recall", "cuda_seed","running_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for task_name, times in avg_times.items():
        if times:
            avg_time = sum(times) / len(times)
        else:
            avg_time = 0.0
        writer.writerow({
            "task_name": task_name,
            "index": "avg",
            "test_recall": "",
            "cuda_seed": "",
            "running_time": f"{avg_time:.4f}"
        })
        

