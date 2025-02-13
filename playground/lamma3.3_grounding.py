import pickle 

path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa4/llava_vqa_yes_gsam_grounding_random_1000/explanations/explanations_404552.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)
    
# print(data.keys())
# if  key starts with 'response' then it is the response from the model, print it 
for key in data.keys():
    if key.startswith('response'):
        print(data[key])
        
        print(data[key]['decoded_outputs'])
        # print('\n
