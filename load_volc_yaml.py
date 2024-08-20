import os

import yaml


def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def save_dict_to_yaml(dict_value: dict, save_path: str):
    with open(save_path, "w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


now_work_dir = os.getcwd()
mount_path = now_work_dir.split("/", maxsplit=2)[2]
r_yaml = read_yaml_to_dict(os.environ["EXAMPLE_YAML_PATH"])
r_yaml["TaskRoleSpecs"] = [
    {"RoleName": "worker", "RoleReplicas": int(os.environ["RoleReplicas"]), "Flavor": "ml.hpcpni2l.28xlarge"}
]
# r_yaml["TensorBoardStorage"] = {"SubPath": os.path.join(mount_path, os.environ["tensorboard_folder"]), "Type": "Vepfs"}
# r_yaml["EnableTensorBoard"] = True
r_yaml["ResourceQueueName"] = os.environ["ResourceQueueName"]
r_yaml["Entrypoint"] = f"cd {now_work_dir} && " + os.environ["Entrypoint"]
r_yaml["Tags"] = [os.environ["Tags"]]
r_yaml["TaskName"] = os.environ["JOB_NAME"]
save_dict_to_yaml(r_yaml, os.environ["saved_yaml_path"])