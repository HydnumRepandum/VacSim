#!/bin/bash
#SBATCH -J YOUR_JOB_NAME                            # Job name
#SBATCH --output=OUTPUT_PATH       # Output file
#SBATCH --nodes=1                             # Request 1 node (single node)
#SBATCH --gres=gpu:4                     # Request 4 GPUs on the single node
#SBATCH --ntasks-per-node=1                   # 1 task per node
#SBATCH --mem=200GB                           # Request 200 GB of memory
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --partition=YOUR_PARTITION_NAME             

# Parse input arguments, expecting -s as the first argument and -m as the second
server_id="$1" # This is a mapping from ID to server ports
model_type="$2"


# Check if server_id is set, otherwise, exit with an error
if [ -z "$server_id" ]; then
  echo "Error: server_id argument is required. Use -s <server_id> to specify it."
  exit 1
fi

# Check if model_type is set, otherwise, exit with an error
if [ -z "$model_type" ]; then
  echo "Error: model_type argument is required. Use -m <model_type> to specify it (0, 1, or 2)."
  exit 1
fi

# Validate model type argument (must be 0, 1, or 2)

# Determine the model to use based on the model_type argument
model="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Default model for type 0
if [ "$model_type" -eq 0 ]; then
  model="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Model for type 0
elif [ "$model_type" -eq 1 ]; then
  model="meta-llama/Meta-Llama-3-8B-Instruct"   # Model for type 1
elif [ "$model_type" -eq 2 ]; then
  model="failspy/Meta-Llama-3-8B-Instruct-abliterated-v3" # Model for type 2 
elif [ "$model_type" -eq 3 ]; then
  model="meta-llama/Llama-3.2-3B-Instruct"
elif [ "$model_type" -eq 4 ]; then
  model="meta-llama/Llama-3.2-1B-Instruct"
elif [ "$model_type" -eq 5 ]; then
  model="microsoft/Phi-3.5-mini-instruct"
elif [ "$model_type" -eq 6 ]; then
  model="Qwen/Qwen2.5-7B-Instruct"
else
  echo "Error: Invalid model_type argument. Use 0, 1, or 2 to specify the model type."
  exit 1
fi


# List of nodes to exclude
EXCLUDE_NODES=("n03")

# Activate the environment
source ~/.bashrc
source activate sim

# Define a dictionary of server IDs to ports
declare -A int_to_ports
int_to_ports[0]="49172 55050 60050 60100"
int_to_ports[1]="32773 49177 49178 49179"
int_to_ports[2]="49180 55060 60060 60110"
int_to_ports[3]="32774 49181 49182 49183"
int_to_ports[4]="49184 55070 60070 60120"
int_to_ports[5]="32775 49185 49186 49187"
int_to_ports[6]="49188 55080 60080 60130"
int_to_ports[7]="32776 49189 49190 49191"
int_to_ports[8]="49192 55090 60090 60140"
int_to_ports[9]="32777 49193 49194 49195"
int_to_ports[10]="49196 55100 60100 60150"
int_to_ports[11]="32778 49197 49198 49199"
int_to_ports[12]="49200 55110 60110 60160"
int_to_ports[13]="32779 49201 49202 49203"
int_to_ports[14]="49204 55120 60120 60170"

# Validate the server ID
if [[ -z "${int_to_ports[$server_id]}" ]]; then
  echo "Error: Server ID $server_id not found in the port mapping."
  exit 1
fi

# Extract the ports for the selected server ID
PORTS=(${int_to_ports[$server_id]})

# Get the list of GPU IDs for this node
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))

# Check if the number of GPUs matches the number of ports
if [ ${#GPU_IDS[@]} -lt ${#PORTS[@]} ]; then
  echo "Error: Not enough GPUs available for the ports on this node."
  exit 1
fi

# Set up log file to capture which ports and models are running
LOG_FILE="outputs/running_servers_${SLURM_JOB_ID}.txt"
echo "Logging running servers to $LOG_FILE"
echo "Server ID: $server_id" > "$LOG_FILE"
echo "Model: $model" >> "$LOG_FILE"
echo "Node: $NODE_NAME" >> "$LOG_FILE"
echo "Assigned ports: ${PORTS[@]}" >> "$LOG_FILE"
echo "Assigned GPUs: ${GPU_IDS[@]}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Starting servers..." >> "$LOG_FILE"

# Loop through the ports and start a server on each with a different GPU
for i in "${!PORTS[@]}"
do
  PORT=${PORTS[$i]}
  GPU=${GPU_IDS[$i]}
  echo "Starting server on port $PORT with GPU $GPU and model $model" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
      --model $model --guided-decoding-backend lm-format-enforcer --max-model-len 6144 \
      --tensor-parallel-size 1 --port $PORT &

  # Log each running server's details to the file
  echo "Running server on port $PORT with GPU $GPU and model $model" >> "$LOG_FILE"
done

# Wait for all servers to start
wait

# Final log statement indicating completion
echo "All servers are running on node $NODE_NAME" >> "$LOG_FILE"
