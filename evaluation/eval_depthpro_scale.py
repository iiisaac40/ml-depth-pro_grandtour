import os
import glob
import random

SCHEDULE = True
submit_dir ="/cluster/home/haozhu1/Thesis/result/.eval_depthpro"

if not os.path.exists(submit_dir):
    os.makedirs(submit_dir)


directory = "/cluster/scratch/haozhu1/depth_data/eval_images/KITTI"
available_dates = os.listdir(directory)

depth_alignment = 'FALSE'

# csv_pattern = 'depthPro_noAlignment_metric_KITTI' if depth_alignment == 'FALSE' else 'depthPro_Alignment_metric_KITTI'

# txt_file = os.path.join('/mnt/KITTI', "val_pairs.txt")
# csv_file = txt_file.replace('pairs.txt', f'{csv_pattern}.csv')

txt_file = '/mnt/txt_files/'
for max_depth in [80]:
    for accum_frames in [150]:
        csv_pattern = f'depthPro_accum{str(accum_frames)}_maxdepth{max_depth}'
    
        test_txt = os.path.join(txt_file, f'test_{str(accum_frames)}_files.txt')
        csv_file = test_txt.replace('files.txt', f'{csv_pattern}.csv')
        print(f"csv_file: {csv_file}")
    
        # test_txt = os.path.join('/mnt/KITTI', "val_pairs.txt")
        # csv_file = test_txt.replace('pairs.txt', f'{csv_pattern}.csv')
    
    
        master_port = random.randint(10000, 20000)
        content = f"""#!/bin/bash

#SBATCH --account=es_hutter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12288m
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=13312
#SBATCH --tmp=90000
#SBATCH --output="/cluster/home/haozhu1/Thesis/.out/depthpro_eval_grandtour_out.log"
#SBATCH --error="/cluster/home/haozhu1/Thesis/.out/depthpro_eval_grandtour_out.log"
#SBATCH --open-mode=truncate

mkdir -p $TMPDIR/GrandTour
tar -xf /cluster/scratch/haozhu1/Thesis/container/grandtour_depth_benchmark2.tar -C $TMPDIR
tar -xf /cluster/scratch/haozhu1/depth_data/updated_images/GrandTour/GrandTour.tar  -C $TMPDIR/GrandTour

# tar -xf /cluster/scratch/haozhu1/depth_data/eval_images/KITTI.tar -C $TMPDIR/GrandTour
# sed -i 's|/cluster/scratch/haozhu1/depth_data/eval_images|/mnt|g' $TMPDIR/GrandTour/KITTI/val_pairs.txt

module load stack/2024-04 gcc/8.5.0 cuda/12.1.1 eth_proxy

apptainer exec --nv --containall --writable --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu  \
  --env HF_HOME=/mnt/.cache/huggingface \
  --env TRANSFORMERS_CACHE=/mnt/.cache/huggingface \
  --env XDG_CACHE_HOME=/mnt/.cache \
  --env MPLCONFIGDIR=/mnt/.config/matplotlib \
  --bind $TMPDIR/GrandTour:/mnt/ \
  $TMPDIR/grandtour_depth_benchmark2.sif \
  /bin/bash -c "
  export HOME=/home && export KLEINKRAM_ACTIVE=ACTIVE && \
  source /opt/conda/etc/profile.d/conda.sh && \
  conda activate grandtour && pip install timm==0.9.10 && \
  source /opt/ros/noetic/setup.bash && \
  source /home/grand_tour_depth_benchmark/catkin_ws/devel/setup.bash && \
  python -c 'import torch; print(torch.cuda.device_count())' && ls /home/grand_tour_depth_benchmark/third_parties/ml-depth-pro_grandtour/checkpoints/ && ls /mnt && \
  python /home/grand_tour_depth_benchmark/third_parties/ml-depth-pro_grandtour/evaluation/eval.py \
  --max_depth {max_depth} --depth_alignment {depth_alignment} \
  --dataset_txt_path {test_txt} --dataset_root_dir /mnt/GrandTour \
  --csv_file {csv_file}  --port {master_port} --dataset grandtour
  "

cp -r $TMPDIR/GrandTour/GrandTour/visualizations /cluster/scratch/haozhu1/depth_data/updated_images
   
exit 0
        """

script_path = os.path.join(submit_dir, "eval_depthpro_kitti.sh")
with open(script_path, "w") as file:
    file.write(content)

if SCHEDULE:
    os.system(f"sbatch {script_path}")



