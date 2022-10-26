import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import shutil
from azureml.core import Environment
from azureml.core import ScriptRunConfig

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

exp = Experiment(workspace=ws, name='py-hands')
script_folder = './'



# choose a name for your cluster
cluster_name = "hd-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_NC6', 
        max_nodes=1)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

# can poll for a minimum number of nodes and for a specific timeout. 
# if no min node count is provided it uses the scale settings for the cluster
compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())

compute_targets = ws.compute_targets
for name, ct in compute_targets.items():
    print(name, ct.type, ct.provisioning_state)

# the training logic is in the tf_mnist.py file.
#shutil.copy('./py_hands_job.py', script_folder)

#tf_env = Environment.get(ws, name='AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu')
tf_env = Environment.get(ws, name='AzureML-tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu')

src = ScriptRunConfig(source_directory=script_folder,
                      script='py_hands_job.py',
                      #arguments=args,
                      compute_target=compute_target,
                      environment=tf_env)

run = exp.submit(src)