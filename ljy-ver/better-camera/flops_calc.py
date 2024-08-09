from eyecontactcnn.model import model_static
from ptflops import get_model_complexity_info
USE_CUDA= False
model_weight = ".\eyecontactcnn\data\model_weights.pkl"
model = model_static(model_weight,USE_CUDA=USE_CUDA)
model_dict = model.state_dict()
model.to("cpu")
flops, params = get_model_complexity_info(model, (3,224,224),as_strings=True,print_per_layer_stat=True)
print("%s %s" % (flops,params))