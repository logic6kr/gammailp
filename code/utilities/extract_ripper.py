import re

def extract_accuracies(logfile_path):
    results = []
    with open(logfile_path, "r") as f:
        lines = f.readlines()
    task = None
    cluster_numbers = None
    dt_acc = None
    ripper_acc = None
    for line in lines:
        if line.startswith("Namespace("):
            # Extract task and cluster_numbers
            task_match = re.search(r"task='([^']+)'", line)
            cluster_match = re.search(r"cluster_numbers=(\d+)", line)
            if task_match:
                task = task_match.group(1)
            if cluster_match:
                cluster_numbers = int(cluster_match.group(1))
        if "Accuracy of decision tree classifier:" in line:
            dt_acc = float(line.strip().split(":")[-1])
        if "Accuracy of ripper classifier:" in line:
            ripper_acc = float(line.strip().split(":")[-1])
            # Save result when both accuracies are found
            results.append({
                "task": task,
                "cluster_numbers": cluster_numbers,
                "decision_tree_accuracy": dt_acc,
                "ripper_accuracy": ripper_acc
            })
    return results

if __name__ == "__main__":
    import csv
    logfile = "/root/code/gammaILP/cache/kandinsky_onered_4to1/ripper_c45.md"
    results = extract_accuracies(logfile)
    # Save as CSV file
    csv_file = "/root/code/gammaILP/cache/kandinsky_onered_4to1/ripper_c45_accuracies.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "cluster_numbers", "decision_tree_accuracy", "ripper_accuracy"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results to {csv_file}")