import time
import os
import subprocess


depths = [3]
cpu_samples = [256, 512, 1024]


def main():
    # cpu_timing = open('cpu_timing.txt', 'w')
    gpu_timing = open('gpu_timing.txt', 'w')
    scenes = os.listdir('./../important_assets/scenes/')
    for scene in scenes:
        scene_path = './../important_assets/scenes/' + scene
        for depth in depths:
            for cpu_sample in cpu_samples:
                gpu_sample = cpu_sample*3
                cpu_render_path = f'./../important_assets/Renders/{scene}_{depth}_{cpu_sample}_cpu.png'
                gpu_render_path = f'./../important_assets/Renders/{scene}_{depth}_{gpu_sample}_gpu.png'
                start = time.time()
                subprocess.run(f'../build/bin/ray -r {depth} {scene_path} {cpu_render_path} -s {cpu_sample}', shell=True)
                cpu_time = time.time() - start
                start = time.time()
                subprocess.run(f'../build/bin/ray -r {depth} {scene_path} {gpu_render_path} -s {gpu_sample} -g', shell=True)
                gpu_time = time.time() - start
                print(f'{scene_path}: {depth} - CPU w/ {cpu_sample}: {cpu_time} seconds, GPU w/ {gpu_sample}: {gpu_time} seconds')
                # cpu_timing.write(f'{scene_path} - CPU|{cpu_sample}: {cpu_time}\n')
                gpu_timing.write(f'{scene_path} - GPU|{gpu_sample}: {gpu_time}\n')
    # cpu_timing.close()
    gpu_timing.close()

if __name__ == '__main__':
    main()