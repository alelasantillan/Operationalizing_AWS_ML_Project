# Operationalizing AWS ML Project
As part of requirements for the AWS MLEND

## Udacity Project 4: AWS Machine Learning Engineer Nanodegree Program (2021-2022)

In this proyect we are requested to accomplish five steps towards applying accuired skills
about most importan services that AWS offer to the operative part of training and deploying 
Machine Learning models.

The project consists in 5 Steps<br/>
Step 1: Train and deploy a model on a Sagemaker notebook (one of the modalities that Sagemaker offer amongst several more).<br/>
Step 2: Perform a similar task on an EC2 instance.</br>
Step 3: Create a Lambda function that will consume your model inference capabilites via endpoints.</br>
Step 4: Set up concurrence for your lambda function and auto-scaling for your deployed endpoint.</br>


## Step 0: Project Set Up and Installation
You must be logged to a AWS account and search for SageMaker to get into the SageMaker console. 
Launch a Sagemaker Notebook Instance.
Created a ml.t2.medium instance named Operationalizing-AWS-ML-Project.
The cost of this instance is not published, but certainly is  less than $0.05/hour, which is the price for ml.t3.medium according AWS: https://aws.amazon.com/sagemaker/pricing/.
This instance will allow me to perform code debugging without incurring in great costs. In general the computing resources are consumed by the processes launched by the notebook rather than the notebook itself, so no much is required in this instance.
Once the project is running, I will consider increasing capacity along with the stress testing I will eventually perform.
In previous experiences I've used also ml.m5.large ($0.115) with sucess, but this time I will take more care of resources used since in the former project I ended too close to the limit.
<br/>
<img src="snapshots/" width="50%">
<br/>



The files to upload are: 
`train_adn_deploy.ipynb`, running each cell will execute the whole process and three auxiliary python scripts that are called by the notebook:
`hpo.py`, contains the prediction model, the training loop as well as the validation and testing tasks in step 2 for hyperparameter optimization.
`train_model.py`, esentially identical to hpo.py, but with the hooks of SageMaker module that performs debugging of the model in step 3.
endpoint_inference.py responsible for invoking the endpoint created by the notebook and return the prediction.

Open the Jupyter Notebook and select the kernel as follows:
<br/>
<img src="images/kernel.png" width="50%">
<br/>
<br/>
and the instance:<br/>
<img src="images/instance.png" width="50%">
<br/>




## Step 1: Train and deploy a model on a Sagemaker notebook 

Notebook Instance setup

1.
Created a ml.t2.medium instance named Operationalizing-AWS-ML-Project.
The cost of this instance is not published, but certainly is  less than $0.05/hour, which is the price for ml.t3.medium according AWS: https://aws.amazon.com/sagemaker/pricing/.
This instance will allow me to perform code debugging without incurring in great costs. In general the computing resources are consumed by the processes launched by the notebook 
rather than the notebook itself, so no much is required in this instance.
Once the project is running, I will consider increasing capacity along with the stress testing I will eventually perform.
In previous experiences I've used also ml.m5.large ($0.115) with sucess, but this time I will take more care of resources used since in the former project I ended too close to the limit.

2.
I uploaded the train_and_deploy-solution.ipynb and hpo.py to run the Hyperparameter Optimization part.

3.
Created a bucket named "udacitysolution-alela" and changed the notebook to use that bucket.
Run the train_and_deploy-solution.ipynb first cells and created the images folders into the bucket

4.
Run the different cells of the notebook to peform Hyperparameter optimization.
I reserved the values of the optimization to perform the training of 
the model. At this point using the smdebug module, web perform debugging of the model 
to avoid the following problems that can show up in any training:
overfitting, vanishin gradients, poor weight initialization or overtraining.
Once the model is trained this way, we create another identical model but with multi instance.
The multi instance training resulted in:
Training seconds: 4221
whereas the single instance just:
Training seconds: 1339

5.
We deployed two endpoints for inference in both single instance and multi instance and peformed the prediction for the same data and we obtained different results as well ad different inference times.
We kept the logs of both invocations to see if there is some sensitive difference but inference times were similar. We should instead perform a lot of requests to see how the endpoints latency behaves in case of higher throughput.


6.
finally we kept the final version of the notebook, which is the one in this repo and we deleted the endpoints and stop the notebook instance. 


## Step 2: Perform a similar task on an EC2 instance.

EC2 Instance setup

1.
I've launched a simple instance ml.t3.medium on EC2 and connected to the instance once it was availabla. 
We choose first to launch a t2.micro since it's free tier, but later it turn out that for amazon deep learning free tier is not available and when installing torch by doing:
pip install torch
there was a memory problem.
If I consider it insufficient, I will retry with a larger instance. Anyway the load is not in the EC2, as it was not on the notebook in sagemaker, but in the jobs launched for hpo and training.:w

To compare:
In the sagemaker task we used ml.t2.medium for the notebook (very light work) and two ml.m5.xlarge for the trainings and ml.m5.large for inferences.
The total costs of performing the tasks with sagemaker were $4.03
The total costs of EC2 using same combination of resources were $

2.
created the dir TrainedModels and downloaded and unzipped there the file:
https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
using wget and unzip commands

3.
Created the file solution.py and I pasted the contents of the scrip ec2train1.py

4.
Run the solution.py and took a screenshot of the model into the TrainedModels directory
After inspecting the code in solution.py I can see that it performs the same tasks that were
performed in the notebook of step 1 (train_and_deploy-solution.ipynb. It was adapted to work 
in a typical linux distro but with some changes as follows:

The code resembles the one used in hpo.py, but it has no smdebug module to perform the final debugging, so the result will be less effective.
As well, hyperparameters are fixed, so there is no hyperparameter optimization. 
Also, this code does not perform the deploy of the endpoint. 
All that will have to be worked later.


Execution time start 6:57 7:24 ended


## Step 3: Create a Lambda function that will consume your model inference capabilites via endpoints.
1. 
we had to re create the endpoint we deleted yeasterday.
We have the models for the endpoint configuration we 
 created for both multi-instance and single-instance
We went to endpoints and created the enpoint using the multi-instance model



## Step 4: Set up concurrence for your lambda function and auto-scaling for your deployed endpoint.
we run into an error when testing.
To solve that we added the policies for sagemakerfullaccess and s3fullaccess to the execution role.


{"url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" }

Test Event Name
test-lambda

Response
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "type-result": "<class 'str'>",
  "COntent-Type-In": "<__main__.LambdaContext object at 0x7f2b8cba16d0>",
  "body": "[[-1.9230748414993286, 2.0606367588043213, -2.975001096725464, 2.7361202239990234, 2.503251314163208, 0.9743241667747498, -0.5436190962791443, -0.552527904510498, -6.05665397644043, 1.721237063407898, 2.425633668899536, 1.1596161127090454, 0.23340359330177307, 1.1787495613098145, -1.682868480682373, 0.04093865305185318, -3.0608670711517334, -1.0202009677886963, 0.5772457122802734, 2.3727235794067383, 1.725218415260315, -0.061895087361335754, -0.507676362991333, -3.4325389862060547, -2.3014237880706787, -6.103825092315674, 2.229447841644287, -1.3559074401855469, 0.214759960770607, -0.4876475930213928, 2.2932322025299072, -1.8778115510940552, -2.646745443344116, -0.4442533254623413, -1.4839411973953247, 0.018974244594573975, 0.5177682042121887, -0.4669727385044098, -0.7576069235801697, -1.7498409748077393, 1.264542818069458, 1.4518052339553833, 2.3175251483917236, 0.6133086681365967, 2.315450429916382, -2.0466291904449463, 1.5150392055511475, -0.7387037873268127, 0.36982569098472595, -0.008169978857040405, 1.6424503326416016, -2.1007399559020996, -2.434668779373169, -0.008525431156158447, -4.083022117614746, -0.004357367753982544, -2.7408454418182373, -0.48699548840522766, -3.6401796340942383, -1.6554871797561646, -1.546553373336792, -1.7185020446777344, -2.040588855743408, -4.549625396728516, -2.8638806343078613, -2.800689697265625, 1.9408971071243286, -0.13130821287631989, -1.1525102853775024, -1.2089629173278809, 3.5017807483673096, -2.380239725112915, -1.1759413480758667, -1.4099303483963013, -2.453126907348633, -2.063462018966675, -3.7739031314849854, -1.0499930381774902, 1.1904747486114502, -1.380522608757019, 1.0462281703948975, -5.719635009765625, 0.3957575261592865, 1.9719444513320923, -4.87777853012085, -5.302970886230469, -0.5300191640853882, -3.840769052505493, -2.534928798675537, 1.4356107711791992, -3.2125887870788574, 0.5481349229812622, -3.9147355556488037, -1.0964760780334473, 1.0998213291168213, 0.7646914124488831, -1.695220708847046, -1.466501235961914, -3.896580696105957, -4.843873023986816, -9.731215476989746, -2.6416592597961426, 1.6428296566009521, -2.0402002334594727, -0.4763360619544983, -0.48405128717422485, -2.3959767818450928, 1.7910501956939697, 3.1037676334381104, 1.1668872833251953, -0.15623413026332855, 0.057766884565353394, -3.2649059295654297, -1.6079130172729492, -3.212416410446167, 0.8164639472961426, -0.5811924934387207, 2.154860258102417, -2.4100425243377686, 0.4797557294368744, -0.26428431272506714, -1.8739787340164185, -0.9146941304206848, 0.05285979062318802, -6.775721073150635, 0.4937722682952881, -3.6679766178131104, 1.0705149173736572, -0.2790021300315857, -3.6403069496154785, -7.05629825592041, -2.225168466567993, -5.770076274871826]]"
}

Function Logs
START RequestId: bedc566e-5914-40dd-8375-87253071f404 Version: $LATEST
Context::: <__main__.LambdaContext object at 0x7f2b8cba16d0>
EventType:: <class 'dict'>
END RequestId: bedc566e-5914-40dd-8375-87253071f404
REPORT RequestId: bedc566e-5914-40dd-8375-87253071f404	Duration: 1064.61 ms	Billed Duration: 1065 ms	Memory Size: 128 MB	Max Memory Used: 68 MB

Request ID
bedc566e-5914-40dd-8375-87253071f404






## Final words:
AWS was able to let us quickly create this model and deploy it in a way that is easily scalable and secure as shown in Step 3 and Step 4. This is one of the strongest features of the AWS Sagemaker for quick and professional Machine Learning solutions.
