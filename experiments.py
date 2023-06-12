import subprocess

if __name__ == "__main__":
    experiments = [
        # "./experiments/sequential/gru4rec.py",
        # "./experiments/sequential/gru4rec_extended.py",
        "./experiments/general/bpr.py",
        "./experiments/general/pop.py",
        "./experiments/sequential/gru4rec_extended.py"
    ]
    for experiment in experiments:
        subprocess.run(f"python {experiment}")
    print("Done!!!")