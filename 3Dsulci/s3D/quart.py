import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import quantus
from quantus import *
from typing import Union

# Choose hardware acceleration if available
def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"



class quart:

    def __init__(self,store,model,x_b,y_b,a_b, Faithfulness='off',Robustness='off',Complexity='off',Randomisation='off',Localisation='off'):

        self.device = torch.device(choose_device())

        if Complexity=='on':
            score,score1=self.C(model,x_b,y_b,a_b)
            self.save(score,(store+'complexity'))
            self.save(score1,(store+'effectivecomplexity'))

        if Faithfulness=='on':
            score,score1=self.F(model,x_b,y_b,a_b)
            self.save(score,(store+'faithfullnesscorel'))
            self.save(score1,(store+'faithfullness_est'))       
 
        if Robustness=='on':
            score1,score2=self.R(model,x_b,y_b,a_b)
            #self.save(score,(store+'RIS'))
            self.save(score1,(store+'Consistensy'))
            self.save(score2,(store+'MaxSensitivety'))


        if Randomisation=='on':
            result=self.Ra(model,x_b,y_b,a_b)
            self.save(result,(store+'randomisation'))
        if Localisation=='on':
            score,score1=self.L(model,x_b,y_b,a_b)
            self.save(score,(store+'topk'))
            self.save(score1,(store+'Auc'))
        if (Faithfulness=='off')and(Robustness=='off')and(Complexity=='off')and(Randomisation=='off')and(Localisation=='off'):
            score=self.quntus_petrub(model,x_b,y_b,a_b,samp=10,ratio=0.2)
            self.save(score,(store+'petrubation'))
            




    def quntus_petrub(self,model,x_b,y_b,a_b,samp=8,ratio=0.25):
        device = torch.device(choose_device())
        print('we runwith :' ,device)
        scores_total=[]
        for i in range(x_b.shape[3]):
            metric = quantus.MaxSensitivity(nr_samples=samp,lower_bound=ratio, norm_numerator=quantus.norm_func.fro_norm, norm_denominator=quantus.norm_func.fro_norm, perturb_func=quantus.uniform_noise,similarity_func=quantus.difference)
            x=x_b[:,:,:,i,:]
            a=a_b[:,:,:,i,:]
            print(x.shape,a.shape)
            x=np.transpose(x, (0,3,2,1))
            scores = metric(model=model,x_batch=x,y_batch=y_b,a_batch=a,device=device)
            score_total.append(scores)
        scores_total=np.array(scores_total)
        print(scores_total.shape)
        scores_t=np.mean(scores_total)
        print(scores_t.shape)
        return scores_t


    def F(self,model,x_b,y_b,a_b):
        device=self.device
       # Return faithfulness correlation scores in an one-liner - by calling the metric instance.
        scores_total=[]
        scores_total2=[]

        xt=np.transpose(x_b, (4,0,1,2,3))
        at=np.transpose(a_b, (4,0,1,2,3))
        yt=np.transpose(y_b, (1,0))
        print(xt.shape,yt.shape,at.shape)
        for i in range(xt.shape[1]):
            x=xt[:,i,:,:,:]
            a=at[:,i,:,:,:]
            y=yt[:,i]
            print(x.shape)
            x=np.reshape(x,[x.shape[0],x.shape[1],x.shape[2],x.shape[3],1])
            a=np.reshape(a,[a.shape[0],1,a.shape[1],a.shape[2],a.shape[3]])
            y=np.reshape(y,[y.shape[0],1])
            print(x.shape,a.shape)
            #scores=quantus.FaithfulnessCorrelation(nr_runs=50,subset_size=1472,perturb_baseline="black",perturb_func=None,similarity_func=quantus.similarity_func.correlation_pearson,abs=False,return_aggregate=False,)(model=model,x_batch=x, y_batch=y,a_batch=a,channel_first=True,device=device)
            scores=0
            scores_total.append(scores)

       # Return faithfulness estimate scores in an one-liner - by calling the metric instance.
            scores2=quantus.FaithfulnessEstimate(perturb_func=quantus.perturb_func.baseline_replacement_by_indices,similarity_func=quantus.similarity_func.correlation_pearson,features_in_step=184,perturb_baseline="black",)(model=model, x_batch=x, y_batch=y,a_batch=a,channel_first=True,device=device,)
            #scores2=0
            scores_total2.append(scores2)
        scores_t=np.array(scores_total)
        scores_t2=np.array(scores_total2)
        print(scores_t.shape)

        return scores_t,scores_t2



    def R(self,model,x_b,y_b,a_b,samp=8):
        device=self.device
        # Return Relative Input Stability scores in an one-liner - by calling the metric instance.
        #scores_total=[]
        scores_total2=[]
        scores_total3=[]
        xt=np.transpose(x_b, (4,0,1,2,3))
        at=np.transpose(a_b, (4,0,1,2,3))
        yt=np.transpose(y_b, (1,0))
        print(xt.shape,yt.shape,at.shape)
        for i in range(xt.shape[1]):
            x=xt[:,i,:,:,:]
            a=at[:,i,:,:,:]
            y=yt[:,i]
            print(x.shape)
            x=np.reshape(x,[x.shape[0],x.shape[1],x.shape[2],x.shape[3],1])
            a=np.reshape(a,[a.shape[0],1,a.shape[1],a.shape[2],a.shape[3]])
            y=np.reshape(y,[y.shape[0],1])
            print(x.shape,a.shape)
            #score=quantus.Continuity(patch_size=30,nr_steps=10,perturb_baseline="uniform",similarity_func=quantus.similarity_func.correlation_spearman,)(model=model, x_batch=x, y_batch=y,a_batch=a,channel_first=True,device=device,)
        # Return Consistency scores in an one-liner - by calling the metric instance.
            #score2=quantus.Consistency(discretise_func=quantus.discretise_func.top_n_sign,return_aggregate=False,)(model=model, x_batch=x, y_batch=y,a_batch=a,channel_first=True, device=device,)
        # Return max sensitivity scores in an one-liner - by calling the metric instance.
            score3=quantus.MaxSensitivity(nr_samples=10,lower_bound=0.2,norm_numerator=quantus.norm_func.fro_norm,norm_denominator=quantus.norm_func.fro_norm,perturb_func=quantus.perturb_func.uniform_noise,similarity_func=quantus.similarity_func.difference,)(model=model, x_batch=x, y_batch=y,a_batch=None ,explain_func=quantus.explain,explain_func_kwargs={"method": "Saliency"},channel_first=True,device=device,)
            score2=quantus.AvgSensitivity(nr_samples=10,lower_bound=0.2,norm_numerator=quantus.norm_func.fro_norm,norm_denominator=quantus.norm_func.fro_norm,perturb_func=quantus.perturb_func.uniform_noise,similarity_func=quantus.similarity_func.difference,)(model=model, x_batch=x, y_batch=y,a_batch=None,explain_func=quantus.explain,explain_func_kwargs={"method": "Saliency"},channel_first=True,device=device,)

            #score_total.append(scores)
            score_total2.append(scores2)
            score_total3.append(scores3)
        #scores_t=np.mean(scores_total)
        scores_t2=np.array(scores_total2)
        scores_t3=np.array(scores_total3)
        print(scores_t2.shape)

        return scores_t2,scores_t3




    def L(self,model,x_b,y_b,a_b):
        device=self.device
        score=quantus.TopKIntersection()(model=model, x_batch=x_b, y_batch=y_b,a_batch=a_b,device=device,)
        score1=quantus.AUC()(model=model, x_batch=x_b, y_batch=y_b,a_batch=a_b,device=device,)

        return score,score1


    def Ra(self,model,x_b,y_b,a_b):
        device=self.device
        result={method: quantus.ModelParameterRandomisation(layer_order="bottom_up",similarity_func=quantus.similarity_func.correlation_spearman,normalise=True,)(model=model, x_batch=x_b, y_batch=y_b,a_batch=a_b,device=device,)}          
        return result
   

    def C(self,model,x_b,y_b,a_b):

        scores_total2=[]
        scores_total3=[]

        device=self.device
        print(x_b.shape,y_b.shape)
        xt=np.transpose(x_b, (4,0,1,2,3))
       	at=np.transpose(a_b, (4,0,1,2,3))	
        yt=np.transpose(y_b, (1,0))
        print(xt.shape,yt.shape,at.shape)
        for i in range(xt.shape[1]):
            x=xt[:,i,:,:,:]
            a=at[:,i,:,:,:]
            y=yt[:,i]
            print(x.shape)  
            x=np.reshape(x,[x.shape[0],1,x.shape[1],x.shape[2],x.shape[3]])
            a=np.reshape(a,[a.shape[0],1,a.shape[1],a.shape[2],a.shape[3]])
            
            y=np.reshape(y,[y.shape[0],1])
            score1=quantus.Complexity(disable_warnings=True)(model=model, x_batch=x, y_batch=y,a_batch=a,channel_first=True,device=device,)
            #score2=quantus.EffectiveComplexity(eps=1e-5,)(model=model, x_batch=x, y_batch=y,a_batch=a,channel_first=True,device=device,)
            score2=0
            scores_total2.append(score1)
            scores_total3.append(score2)
        sc1=np.array(scores_total2)
        sc2=np.array(scores_total3)
        print(sc1.shape)
        return sc1,sc2



    def save(self,result,store):
        np.savetxt(store, result, delimiter=",")

      
