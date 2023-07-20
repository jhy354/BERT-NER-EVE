import os
from customize_service import CustomizeService


# 待预测文件路径
test_path = "./datasets/cner/infer.char.bmes"
service = CustomizeService(model_name="pytorch_model.bin", model_path="./outputs/cner_output/bert/")

post_data = {"input_txt": {os.path.basename(test_path): open(test_path, "rb")}}
print('post_data:',post_data)
data = service._preprocess(post_data)
data = service._inference(data)
# data = service._postprocess(data)

# with open("./datasets/cner/res.char.bmes", "w", encoding="utf-8") as f:
#     f.writelines(data.get("result"))
