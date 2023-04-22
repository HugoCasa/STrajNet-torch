import submitit
import logging
import datetime

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Basic parameters
SAVE_DIR = "./weights"
FILES_DIR = "./waymo110/preprocessed_data"
CHECKPOINT_PATH = None
TB_DIR = "./tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Training parameters
LR = 1e-4
BATCH_SIZE = 8  # batch size per GPU => 8 * 2 * 2 = 32 batch size
EPOCHS = 10

LOCAL = False

# SLURM parameters
LOG_DIR = "./logs"
N_NODES = 4
GPUS_PER_NODE = 2
CPUS_PER_NODE = 40
MEM_PER_NODE = 150


from trainer import Trainer


def main():
    trainer = Trainer(
        lr=LR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        save_dir=SAVE_DIR,
        files_dir=FILES_DIR,
        tb_dir=TB_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        local=LOCAL,
    )

    if LOCAL:
        LOG.info("Running locally")
        trainer()
        return 0

    executor = submitit.AutoExecutor(folder=LOG_DIR)
    executor.update_parameters(
        name="train_strajnet_torch",
        nodes=N_NODES,
        mem_gb=MEM_PER_NODE,
        gpus_per_node=GPUS_PER_NODE,
        tasks_per_node=GPUS_PER_NODE,
        cpus_per_task=CPUS_PER_NODE // GPUS_PER_NODE,  # 40 total on one node
        timeout_min=60 * 72,  # 72 hours
        slurm_partition="gpu",
        slurm_qos="gpu",
        slurm_gres=f"gpu:{GPUS_PER_NODE}",
        slurm_additional_parameters={
            "requeue": True,
        },
    )

    job = executor.submit(trainer)

    LOG.info(f"Submitted job_id: {job.job_id}")

    return job


if __name__ == "__main__":
    main()
