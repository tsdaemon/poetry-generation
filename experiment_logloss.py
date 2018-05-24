import subprocess


if __name__ == '__main__':
    # processes = []
    layers = range(3, 4)
    for layer in layers:
        subprocess.run(
            "python3 main.py " +
            "-cuda -max_epoch 300 " +
            "-layers {} ".format(layer) +
            "-output_dir ./results/layers/{}".format(layer), shell=True)

        # _, tmp_file = tempfile.mkstemp()
        # f = open(tmp_file, 'w')
        # p = subprocess.Popen(, stdout=f, shell=False)
        # print("Started, outfile: {}".format(f.name))
        # processes.append((p, f))

    # for p, f in processes:
    #     p.wait()
    #     f.close()
