import subprocess


if __name__ == '__main__':
    subprocess.run(
        "python3 main.py " +
        "-cuda -model ./results/layers/2/199/model.pth " +
        "-layers 2 -mode q " +
        "-output_dir ./results/q/layers2", shell=True)
